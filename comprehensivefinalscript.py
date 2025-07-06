
# --------------------------------------------------------------
# GLB TO 3DM CONVERTER
# --------------------------------------------------------------

import pygltflib
import rhino3dm
import os
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
import matplotlib.pyplot as plt
# import csv
import pandas as pd
from collections import OrderedDict
import json
from shapely.ops import unary_union # For combining polygons on a floor
import plotly.io as pio
pio.renderers.default = 'browser'

def convert_glb_to_3dm(glb_path, output_3dm_path):
    """
    Loads a GLB file, extracts its mesh data, and saves it as a 3dm file.
    Only mesh geometry will be converted. Materials, animations, etc., are not translated.
    """

    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    print(f"Loading GLB file from: {glb_path}")
    gltf = pygltflib.GLTF2().load(glb_path)

    model_3dm = rhino3dm.File3dm()

    # Iterate through GLTF meshes and add them to the 3dm model
    for mesh_idx, gltf_mesh in enumerate(gltf.meshes):
        print(f"Processing mesh: {gltf_mesh.name if gltf_mesh.name else f'Mesh_{mesh_idx}'}")
        
        for primitive in gltf_mesh.primitives:
            # Get vertex positions
            accessor_pos = gltf.accessors[primitive.attributes.POSITION]
            buffer_view_pos = gltf.bufferViews[accessor_pos.bufferView]
            buffer_pos = gltf.buffers[buffer_view_pos.buffer]

            # Extract vertices
            vertices_bytes = gltf.get_data_from_buffer_uri(buffer_pos.uri)[
                buffer_view_pos.byteOffset : buffer_view_pos.byteOffset + buffer_view_pos.byteLength
            ]
            vertices = np.frombuffer(vertices_bytes, dtype=np.float32).reshape(-1, 3)

            # Detect unit scale
            scale_factor = 1
            print(f"Detected scale factor: {scale_factor} (applied to convert to meters)")

            rhino_mesh = rhino3dm.Mesh()

            for v in vertices:
                rhino_mesh.Vertices.Add(v[0] * scale_factor, v[1] * scale_factor, v[2] * scale_factor)

            # Get indices (faces)
            if primitive.indices is not None:
                accessor_indices = gltf.accessors[primitive.indices]
                buffer_view_indices = gltf.bufferViews[accessor_indices.bufferView]
                buffer_indices = gltf.buffers[buffer_view_indices.buffer]

                indices_bytes = gltf.get_data_from_buffer_uri(buffer_indices.uri)[
                    buffer_view_indices.byteOffset : buffer_view_indices.byteOffset + buffer_view_indices.byteLength
                ]
                
                # Determine dtype for indices (UINT8, UINT16, UINT32)
                if accessor_indices.componentType == pygltflib.UNSIGNED_BYTE:
                    indices_dtype = np.uint8
                elif accessor_indices.componentType == pygltflib.UNSIGNED_SHORT:
                    indices_dtype = np.uint16
                elif accessor_indices.componentType == pygltflib.UNSIGNED_INT:
                    indices_dtype = np.uint32
                else:
                    print(f"Warning: Unsupported index component type: {accessor_indices.componentType}. Skipping faces for this primitive.")
                    continue

                indices = np.frombuffer(indices_bytes, dtype=indices_dtype)

                # glTF uses flat arrays for indices, assuming triangles (mode 4)
                if primitive.mode == pygltflib.TRIANGLES:
                    for i in range(0, len(indices), 3):
                        rhino_mesh.Faces.AddFace(int(indices[i]), int(indices[i+1]), int(indices[i+2]))
                else:
                    print(f"Warning: Skipping primitive with unsupported mode: {primitive.mode}. Only triangles (mode 4) are fully supported for faces.")
                    continue
            else:
                print(f"Warning: Primitive has no indices. Assuming sequential triangles, but this might not be correct for complex GLBs.")
                for i in range(0, len(vertices) - 2, 3):
                    rhino_mesh.Faces.AddFace(i, i+1, i+2)

            # Optional: Calculate normals
            rhino_mesh.Normals.ComputeNormals()
            rhino_mesh.Compact()

            model_3dm.Objects.AddMesh(rhino_mesh)
    
    print(f"Saving 3dm file to: {output_3dm_path}")
    model_3dm.Write(output_3dm_path, 0)
    print("Conversion complete!")

    print("Sample vertices:", vertices[:5])

# --- Define paths ---
# Prompt the user for the GLB file path
while True:
    glb_input_path = input("Please enter the full path to your GLB file: ").strip().strip('"').strip("'")
    glb_file_path = os.path.abspath(os.path.expanduser(glb_input_path))

    # glb_input_path = input("Please enter the full path to your GLB file (e.g., C:\\Users\\papad\\Documents\\GitHub\\Octopusie\\Reference Files\\my_model.glb): ")
    # glb_file_path = os.path.abspath(os.path.expanduser(glb_input_path.strip())) # Clean and absolute path
    if os.path.exists(glb_file_path):
        if glb_file_path.lower().endswith(".glb"):
            break
        else:
            print("Error: The provided file is not a .glb file. Please enter a valid GLB file path.")
    else:
        print(f"Error: File not found at '{glb_file_path}'. Please check the path and try again.")

# Derive the output 3dm file path from the GLB file path
directory = os.path.dirname(glb_file_path)
file_name_without_ext = os.path.splitext(os.path.basename(glb_file_path))[0]
output_3dm_file_path = os.path.join(directory, f"{file_name_without_ext}.3dm")

# --- Run the conversion ---
try:
    convert_glb_to_3dm(glb_file_path, output_3dm_file_path)
    print(f"\nConversion successful! The 3DM file is saved at: {output_3dm_file_path}")

except FileNotFoundError as e:
    print(e)
    print("Please ensure the GLB file path is correct.")
except Exception as e:
    print(f"An error occurred during conversion: {e}")


# --------------------------------------------------------------
# MASHALLA FOR FORCED COLUMNS - Now uses the dynamically generated 3DM file
# --------------------------------------------------------------

# File path - now dynamically set from the conversion output
rhino_path = output_3dm_file_path

if not os.path.exists(rhino_path):
    raise FileNotFoundError(f"File not found: {rhino_path}")

# Load model
model = rhino3dm.File3dm.Read(rhino_path)

# Extract geometries and their bounding boxes
building_floor_footprints = []
all_mesh_bboxes = []
roof_meshes_info = []
max_z = 0.0

Z_FLATNESS_TOLERANCE = 0.1

for obj in model.Objects:
    geom = obj.Geometry
    if geom.ObjectType == rhino3dm.ObjectType.Mesh:
        bbox = geom.GetBoundingBox()
        
        bbox_x_dim = bbox.Max.X - bbox.Min.X
        bbox_y_dim = bbox.Max.Y - bbox.Min.Y
        bbox_z_dim = bbox.Max.Z - bbox.Min.Z

        if bbox_z_dim < Z_FLATNESS_TOLERANCE and bbox_x_dim > 0.1 and bbox_y_dim > 0.1:
            base_pts = [
                [bbox.Min.X, bbox.Min.Y],
                [bbox.Max.X, bbox.Min.Y],
                [bbox.Max.X, bbox.Max.Y],
                [bbox.Min.X, bbox.Max.Y],
                [bbox.Min.X, bbox.Min.Y]
            ]
            poly = Polygon(base_pts)
            if poly.area > 1e-3:
                building_floor_footprints.append(poly)
                roof_meshes_info.append((obj.Attributes.Id, bbox, poly))

        all_mesh_bboxes.append(bbox)
        
        max_z = max(max_z, bbox.Max.Z)

if not building_floor_footprints:
    raise RuntimeError("No meaningful building floor footprints (meshes flat in Z with area) found in the model.")

print(f"Detected {len(building_floor_footprints)} building floor footprints. Max Z height: {max_z:.2f}m")
print(f"Total meshes contributing to height calculation: {len(all_mesh_bboxes)}")
print(f"Detected {len(roof_meshes_info)} potential roof meshes for comparison.")

# --------------------------------------------------------------
# ===== PERIMETER =====
# --------------------------------------------------------------

# Find and print roofs that are peaks (taller than all directly touching roofs)
print("\n--- Analyzing Roof Heights ---")

wall_thickness = 0.3  # meters (30 cm)
print(f"Using default wall thickness: {wall_thickness} m")


combined_building_polygon = MultiPolygon(building_floor_footprints)

try:
    exterior_perimeter = combined_building_polygon.buffer(wall_thickness, join_style=1)
except Exception as e:
    print(f"Could not buffer the building outline. Error: {e}")
    exterior_perimeter = None

if exterior_perimeter and exterior_perimeter.geom_type == 'MultiPolygon':
    exterior_perimeter = max(exterior_perimeter.geoms, key=lambda p: p.area)

perimeter_line_coords = []
if exterior_perimeter:
    if exterior_perimeter.geom_type == 'Polygon':
        perimeter_line_coords = list(exterior_perimeter.exterior.coords)
    elif exterior_perimeter.geom_type == 'MultiPolygon':
        perimeter_line_coords = list(exterior_perimeter.geoms[0].exterior.coords)
    else:
        print("Warning: The buffered perimeter is not a Polygon or MultiPolygon. Cannot extract line coordinates.")

# --- End perimeter section ---

detected_rooms = sorted([(poly, poly.area) for poly in building_floor_footprints], key=lambda x: -x[1])
if not detected_rooms:
    raise RuntimeError("No valid rooms detected after filtering by area. Check your Rhino model geometry.")


# --------------------------------------------------------------
# ===== FLOOR DETECTION =====
# --------------------------------------------------------------

floor_height = 2.5  # meters per floor
num_floors = max(1, int(round(max_z / floor_height)))
print(f"Automatically calculated number of floors: {num_floors} (based on max height {max_z:.2f}m and {floor_height}m per floor)")

# Assume floor_height and Z_FLATNESS_TOLERANCE are defined

# --- Step 1: Categorize Floor Footprints by Height/Floor Level ---
floor_footprints_by_level = {}
floor_z_levels = set() # To store the z-coordinates of each detected floor

for obj in model.Objects:
    geom = obj.Geometry
    if geom.ObjectType == rhino3dm.ObjectType.Mesh:
        bbox = geom.GetBoundingBox()
        
        bbox_x_dim = bbox.Max.X - bbox.Min.X
        bbox_y_dim = bbox.Max.Y - bbox.Min.Y
        bbox_z_dim = bbox.Max.Z - bbox.Min.Z

        if bbox_z_dim < Z_FLATNESS_TOLERANCE and bbox_x_dim > 0.1 and bbox_y_dim > 0.1:
            base_pts = [
                [bbox.Min.X, bbox.Min.Y],
                [bbox.Max.X, bbox.Min.Y],
                [bbox.Max.X, bbox.Max.Y],
                [bbox.Min.X, bbox.Max.Y],
                [bbox.Min.X, bbox.Min.Y]
            ]
            poly = Polygon(base_pts)
            if poly.area > 1e-3:
                # Estimate floor level based on min Z of the bbox
                # You might need to refine this to snap to discrete floor levels
                
                # A more robust way might be to cluster Z values
                # For simplicity, let's just use the rounded Z for grouping for now
                approx_z = round(bbox.Max.Z / floor_height) * floor_height # Snap to nearest floor_height multiple
                floor_z_levels.add(approx_z)
                
                if approx_z not in floor_footprints_by_level:
                    floor_footprints_by_level[approx_z] = []
                floor_footprints_by_level[approx_z].append(poly)

# Sort the Z levels to process floors in order
sorted_floor_z_levels = sorted(list(floor_z_levels))
print(f"Detected Z levels for floors: {sorted_floor_z_levels}")


# --------------------------------------------------------------
# ===== CANTILEVER DETECTION =====
# --------------------------------------------------------------


# --- REVISED CANTILEVER DETECTION LOGIC ---
CANTILEVER_AREA_THRESHOLD = 0.5 # m^2, adjust as needed

print("\n--- REVISING Cantilever Analysis (Ground Floor vs. First Floor) ---")

detected_cantilevers = []

forced_cantilever_corner_points = []



# Store regions where columns should NOT be placed from ground to first floor
cantilever_no_column_zones_ground_to_first = [] 

CANTILEVER_CHECK_BUFFER = 0.05 # meters, a small buffer for point-in-polygon checks

# These columns will start from the first floor level instead of the ground.
columns_to_skip_ground_to_first_span = set()

# Ensure we have at least a ground floor and a "first floor" above it
if len(sorted_floor_z_levels) >= 2:
    ground_floor_z = sorted_floor_z_levels[0] # Assuming the lowest Z is the ground floor
    first_floor_above_ground_z = sorted_floor_z_levels[1] # This is the "first floor" to check for cantilevers

    ground_floor_polygons = floor_footprints_by_level.get(ground_floor_z, [])
    first_floor_polygons = floor_footprints_by_level.get(first_floor_above_ground_z, [])

    if not ground_floor_polygons:
        print(f"No ground floor polygons found at Z level {ground_floor_z:.2f}. Cannot check for first floor cantilevers.")
    elif not first_floor_polygons:
        print(f"No first floor polygons found at Z level {first_floor_above_ground_z:.2f}. No cantilevers to detect.")
    else:
        # Create a combined footprint for the ground floor
        combined_ground_floor_footprint = unary_union(ground_floor_polygons)
        
        print(f"\nChecking First Floor (Z={first_floor_above_ground_z:.2f}m) against Ground Floor (Z={ground_floor_z:.2f}m) for cantilevers.")

        for first_floor_poly in first_floor_polygons:
            # Calculate the part of the first floor polygon that extends beyond the ground floor
            cantilever_part = first_floor_poly.difference(combined_ground_floor_footprint)

            if not cantilever_part.is_empty and cantilever_part.geom_type in ['Polygon', 'MultiPolygon']:
                cantilever_area = 0
                if cantilever_part.geom_type == 'Polygon':
                    cantilever_area = cantilever_part.area
                else: # MultiPolygon
                    for geom in cantilever_part.geoms:
                        if geom.geom_type == 'Polygon':
                            cantilever_area += geom.area
                
                if cantilever_area > CANTILEVER_AREA_THRESHOLD:
                    detected_cantilevers.append({
                        "UpperFloorZ": round(first_floor_above_ground_z, 2),
                        "LowerFloorZ": round(ground_floor_z, 2),
                        "CantileverArea": round(cantilever_area, 2),
                        "CantileverGeometry": cantilever_part,
                        "UpperFloorPolygonCenter": [round(first_floor_poly.centroid.x, 2), round(first_floor_poly.centroid.y, 2)]
                    })
                    print(f"  Detected cantilever at First Floor Z={first_floor_above_ground_z:.2f}m with area: {cantilever_area:.2f} m²")
                    print(f"    Originating polygon center: X={first_floor_poly.centroid.x:.2f}, Y={first_floor_poly.centroid.y:.2f}")
else:
    print("\nNot enough floor levels (less than 2) to check for first floor cantilevers.")

if detected_cantilevers:
    print("\n--- Summary of Detected First Floor Cantilevers ---")
    for cantilever in detected_cantilevers:
        print(f"  - Upper Floor Z: {cantilever['UpperFloorZ']}m, Lower Floor Z: {cantilever['LowerFloorZ']}m, Cantilever Area: {cantilever['CantileverArea']} m²")
else:
    print("\nNo significant first floor cantilevers detected based on the defined thresholds.")

    
for cantilever in detected_cantilevers:
    cantilever_geom = cantilever["CantileverGeometry"]
    ground_walls = floor_footprints_by_level[ground_floor_z]

    if cantilever_geom.geom_type == 'Polygon':
        polys = [cantilever_geom]
    elif cantilever_geom.geom_type == 'MultiPolygon':
        polys = list(cantilever_geom.geoms)
    else:
        continue

    for poly in polys:
        coords = list(poly.exterior.coords)

        for i in range(len(coords) - 1):
            pt1, pt2 = coords[i], coords[i+1]
            edge = LineString([pt1, pt2])

            # If this edge touches any wall from the ground floor
            for wall in ground_walls:
                if wall.buffer(0.05).intersects(edge):
                    # Force columns at both endpoints of the edge (i.e., the corners)
                    forced_cantilever_corner_points.append(pt1)
                    forced_cantilever_corner_points.append(pt2)
                    break  # Only need one wall to validate the edge


# --------------------------------------------------------------
# ===== NO COLUMN ZONES =====
# --------------------------------------------------------------


# --- Identify "No-Column" Zones for Cantilevers (Ground to First Floor) ---
if detected_cantilevers and len(sorted_floor_z_levels) >= 2:
    ground_floor_z = sorted_floor_z_levels[0]
    combined_ground_floor_footprint = unary_union(floor_footprints_by_level.get(ground_floor_z, []))

    if not combined_ground_floor_footprint.is_empty:
        print("\n--- Identifying Cantilever 'No-Column' Zones ---")
        for cantilever_info in detected_cantilevers:
            cantilever_geom = cantilever_info['CantileverGeometry']
            
            # The perimeter of the cantilever geometry
            cantilever_perimeter = cantilever_geom.exterior if cantilever_geom.geom_type == 'Polygon' else unary_union([g.exterior for g in cantilever_geom.geoms if g.geom_type == 'Polygon'])

            if cantilever_perimeter.is_empty:
                continue

            # The part of the cantilever perimeter that does NOT touch the ground floor footprint
            exposed_perimeter_candidate = cantilever_perimeter.difference(combined_ground_floor_footprint.buffer(CANTILEVER_CHECK_BUFFER))

            # Filter out small artifacts and keep only LineString or MultiLineString components
            no_column_lines = []
            if exposed_perimeter_candidate.geom_type == 'LineString':
                if exposed_perimeter_candidate.length > 0.1: # Only consider significant lengths
                    no_column_lines.append(exposed_perimeter_candidate)
            elif exposed_perimeter_candidate.geom_type == 'MultiLineString':
                for line in exposed_perimeter_candidate.geoms:
                    if line.length > 0.1:
                        no_column_lines.append(line)
            elif exposed_perimeter_candidate.geom_type == 'Polygon': # Should ideally not be a polygon unless a very thin part
                if exposed_perimeter_candidate.area > 0.01: # Small area threshold
                    no_column_lines.append(exposed_perimeter_candidate.exterior) # Use its exterior as the line

            for line in no_column_lines:
                # Create a thin rectangular zone along this line
                try:
                    no_column_zone = line.buffer(CANTILEVER_CHECK_BUFFER, cap_style=3) # cap_style=3 for square ends
                    cantilever_no_column_zones_ground_to_first.append(no_column_zone)
                except Exception as e:
                    print(f"Warning: Could not create no-column zone from line. Error: {e}")
        
        # Combine all no-column zones into a single MultiPolygon for efficient checking
        if cantilever_no_column_zones_ground_to_first:
            cantilever_no_column_zones_ground_to_first_combined = unary_union(cantilever_no_column_zones_ground_to_first)
            print(f"  Combined {len(cantilever_no_column_zones_ground_to_first)} no-column zones.")
        else:
            cantilever_no_column_zones_ground_to_first_combined = None
            print("  No significant cantilever no-column zones identified.")
    else:
        cantilever_no_column_zones_ground_to_first_combined = None
        print("  Ground floor footprint is empty, cannot define cantilever no-column zones.")
else:
    cantilever_no_column_zones_ground_to_first_combined = None
    print("  No detected cantilevers or not enough floor levels to define no-column zones.")

# --------------------------------------------------------------
# ===== COLUMN PLACEMENT =====
# --------------------------------------------------------------

# Structural logic
MaxS = 6.0
MinS = 3.0

columns_2d_points = [] # Store raw (x,y) points for columns
beams_2d_lines = []    # Store raw ((x1,y1),(x2,y2)) for beams for 2D plot

added_column_xy = set()
columns_to_skip_ground_to_first_span = set()

# Find and print roofs that are peaks (taller than all directly touching roofs)
print("\n--- Analyzing Roof Heights ---")


# --------------------------------------------------------------
# ===== DOMINANT ROOFS =====
# --------------------------------------------------------------

INTERSECTION_BUFFER_ROOF = 0.1

dominant_roofs_identified = []

for i, (roof1_id, roof1_bbox, roof1_poly_2d) in enumerate(roof_meshes_info):
    roof1_max_z = roof1_bbox.Max.Z
    is_dominant_roof = True
    touching_lower_neighbors = []
    touching_equal_or_higher_neighbors = []

    for j, (roof2_id, roof2_bbox, roof2_poly_2d) in enumerate(roof_meshes_info):
        if i == j:
            continue

        roof2_max_z = roof2_bbox.Max.Z

        intersection_geometry = roof1_poly_2d.buffer(INTERSECTION_BUFFER_ROOF).intersection(roof2_poly_2d.buffer(INTERSECTION_BUFFER_ROOF))
        
        is_touching = not intersection_geometry.is_empty and \
                      intersection_geometry.geom_type in ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']
        
        if is_touching:
            if roof1_max_z <= roof2_max_z + 1e-6:
                is_dominant_roof = False
                touching_equal_or_higher_neighbors.append({"Id": str(roof2_id), "Height": round(roof2_max_z, 3)})
            else:
                touching_lower_neighbors.append({"Id": str(roof2_id), "Height": round(roof2_max_z, 3)})
        
    if is_dominant_roof and (touching_lower_neighbors or touching_equal_or_higher_neighbors):
        dominant_roofs_identified.append({
            "RhinoObjectId": str(roof1_id),
            "Height": round(roof1_max_z, 3),
            "Location_Min_X": round(roof1_bbox.Min.X, 3),
            "Location_Min_Y": round(roof1_bbox.Min.Y, 3),
            "Polygon": roof1_poly_2d,
            "TouchingLowerNeighbors": touching_lower_neighbors,
            "TouchingEqualOrHigherNeighbors": touching_equal_or_higher_neighbors
        })

if dominant_roofs_identified:
    print("\nRoofs identified as strictly taller than all their directly touching neighbors:")
    for roof_info in dominant_roofs_identified:
        if not roof_info['TouchingEqualOrHigherNeighbors']:
            print(f"  Roof ID: {roof_info['RhinoObjectId']} (Height: {roof_info['Height']}m)")
            print(f"    Location (Min XY): ({roof_info['Location_Min_X']}, {roof_info['Location_Min_Y']})")
            if roof_info['TouchingLowerNeighbors']:
                neighbor_details = ", ".join([f"ID: {n['Id']} (H: {n['Height']}m)" for n in roof_info['TouchingLowerNeighbors']])
                print(f"    Touching Lower Neighbors: {neighbor_details}")
            else:
                print(f"    No directly touching lower neighbors found (might be isolated or higher than implied).")
else:
    print("\nNo roofs found that are strictly taller than all their directly touching neighbors.")

# --------------------------------------------------------------
# ===== PERIMETER LINE =====
# --------------------------------------------------------------
while True:
    try:
        wall_thickness = 0.3
        if wall_thickness <= 0:
            raise ValueError
        break
    except ValueError:
        print("Please enter a valid positive number for wall thickness.")

combined_building_polygon = MultiPolygon(building_floor_footprints)

try:
    exterior_perimeter = combined_building_polygon.buffer(wall_thickness, join_style=1)
except Exception as e:
    print(f"Could not buffer the building outline. Error: {e}")
    exterior_perimeter = None

if exterior_perimeter and exterior_perimeter.geom_type == 'MultiPolygon':
    exterior_perimeter = max(exterior_perimeter.geoms, key=lambda p: p.area)

perimeter_line_coords = []
if exterior_perimeter:
    if exterior_perimeter.geom_type == 'Polygon':
        perimeter_line_coords = list(exterior_perimeter.exterior.coords)
    elif exterior_perimeter.geom_type == 'MultiPolygon':
        perimeter_line_coords = list(exterior_perimeter.geoms[0].exterior.coords)
    else:
        print("Warning: The buffered perimeter is not a Polygon or MultiPolygon. Cannot extract line coordinates.")

# Force columns at the corners of dominant roof footprints
print("\n--- Forcing columns at dominant roof corners ---")
for roof_info in dominant_roofs_identified:
    if not roof_info['TouchingEqualOrHigherNeighbors']:
        poly_2d = roof_info['Polygon']
        minx, miny, maxx, maxy = poly_2d.bounds
        corners_to_force = [
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy)
        ]
        for cx, cy in corners_to_force:
            rounded_cx = round(cx, 5)
            rounded_cy = round(cy, 5)
            if (rounded_cx, rounded_cy) not in added_column_xy:
                columns_2d_points.append((rounded_cx, rounded_cy))
                added_column_xy.add((rounded_cx, rounded_cy))
                print(f"  Forced column at dominant roof corner: ({rounded_cx}, {rounded_cy})")

# Iterate through detected rooms/floor footprints
for approx_z, floor_polygons in floor_footprints_by_level.items():
    current_floor_z = approx_z # Get the Z-level for the current set of polygons

    for room_poly in floor_polygons: # Iterate through individual room polygons on this floor
        minx, miny, maxx, maxy = room_poly.bounds
        width, height = maxx - minx, maxy - miny
        
        divisions_x = max(1, int(np.ceil(width / MaxS)))
        divisions_y = max(1, int(np.ceil(height / MaxS)))
        
        x_points_grid = np.linspace(minx, maxx, divisions_x + 1)
        y_points_grid = np.linspace(miny, maxy, divisions_y + 1)
        
        # Add interior grid columns (this part is fine, columns go to ground unless skipped)
        for x in x_points_grid:
            for y in y_points_grid:
                col_pt = Point(x, y)
                rounded_x = round(x, 5)
                rounded_y = round(y, 5)
                
                if room_poly.contains(col_pt) or room_poly.buffer(1e-6).contains(col_pt):
                    if (rounded_x, rounded_y) not in added_column_xy:
                        if all(np.linalg.norm(np.array((rounded_x, rounded_y)) - np.array(exist_col_xy)) >= MinS for exist_col_xy in added_column_xy):
                            columns_2d_points.append((rounded_x, rounded_y))
                            added_column_xy.add((rounded_x, rounded_y))

                    # Check for cantilever no-column zones for the vertical span
                    if cantilever_no_column_zones_ground_to_first_combined:
                        if cantilever_no_column_zones_ground_to_first_combined.intersects(col_pt.buffer(CANTILEVER_CHECK_BUFFER)):
                            columns_to_skip_ground_to_first_span.add((rounded_x, rounded_y))

        # Add columns at corners of the room polygon (this part is fine)
        for corner_x, corner_y in room_poly.exterior.coords:
            corner_pt_shapely = Point(corner_x, corner_y)
            rounded_corner_x = round(corner_x, 5)
            rounded_corner_y = round(corner_y, 5)
            
            if (rounded_corner_x, rounded_corner_y) not in added_column_xy:
                if all(np.linalg.norm(np.array((corner_x, corner_y)) - np.array(exist_col_xy)) >= MinS * 0.5 for exist_col_xy in added_column_xy):
                    columns_2d_points.append((rounded_corner_x, rounded_corner_y))
                    added_column_xy.add((rounded_corner_x, rounded_corner_y))
            
            # Check for cantilever no-column zones for the vertical span
            if cantilever_no_column_zones_ground_to_first_combined:
                if cantilever_no_column_zones_ground_to_first_combined.intersects(corner_pt_shapely.buffer(CANTILEVER_CHECK_BUFFER)):
                    columns_to_skip_ground_to_first_span.add((rounded_corner_x, rounded_corner_y))

        # IMPORTANT: NEW BEAM GENERATION LOGIC - APPLY CONDITION HERE
        if abs(current_floor_z - ground_floor_z) > 1e-4: # Or 1e-5, depending on precision needed            # Horizontal beams
            for y_fixed in y_points_grid:
                points_on_line = []
                for x_coord in x_points_grid:
                    p = Point(x_coord, y_fixed)
                    if room_poly.buffer(1e-6).contains(p):
                        points_on_line.append((x_coord, y_fixed))
                
                if len(points_on_line) > 1:
                    for i in range(len(points_on_line) - 1):
                        beams_2d_lines.append((points_on_line[i], points_on_line[i+1]))

            # Vertical beams
            for x_fixed in x_points_grid:
                points_on_line = []
                for y_coord in y_points_grid:
                    p = Point(x_fixed, y_coord)
                    if room_poly.buffer(1e-6).contains(p):
                        points_on_line.append((x_fixed, y_coord))
                
                if len(points_on_line) > 1:
                    for i in range(len(points_on_line) - 1):
                        beams_2d_lines.append((points_on_line[i], points_on_line[i+1]))


# Combine all base columns
all_base_columns = list(added_column_xy)

# --- Utility function for wall height ---
def get_wall_height(x, y, mesh_bboxes, global_max_z):
    pt = Point(x, y)
    relevant_bboxes = []
    for bbox in mesh_bboxes:
        bbox_poly = Polygon([
            [bbox.Min.X, bbox.Min.Y],
            [bbox.Max.X, bbox.Min.Y],
            [bbox.Max.X, bbox.Max.Y],
            [bbox.Min.X, bbox.Max.Y],
            [bbox.Min.X, bbox.Min.Y]
        ])
        if bbox_poly.buffer(1e-4).contains(pt):
            relevant_bboxes.append(bbox)

    if not relevant_bboxes:
        return global_max_z 

    max_relevant_z = 0.0
    for bbox in relevant_bboxes:
        max_relevant_z = max(max_relevant_z, bbox.Max.Z)
        
    return max_relevant_z if max_relevant_z > 0 else global_max_z
# --------------------------------------------------------------
# --- DATA GENERATION AND CSV/JSON EXPORT ---
# --------------------------------------------------------------

print("--- Generating data and Exporting CSVs & JSONs ---")

EXPORT_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(glb_file_path)), "structural_data") # Dynamically set export path

os.makedirs(EXPORT_SAVE_PATH, exist_ok=True)
print(f"Ensuring directory exists: {os.path.abspath(EXPORT_SAVE_PATH)}")

column_lines = []
beam_lines = []
floor_height = 2.5  # Ideal floor height reference

for fx, fy in forced_cantilever_corner_points:
    rounded_fx = round(fx, 5)
    rounded_fy = round(fy, 5)
    if (rounded_fx, rounded_fy) not in added_column_xy:
        columns_2d_points.append((rounded_fx, rounded_fy))
        added_column_xy.add((rounded_fx, rounded_fy))
        print(f"🔵 Forced column at cantilever corner touching ground wall: ({rounded_fx}, {rounded_fy})")


node_coords = []
node_dict = OrderedDict()

def add_node(pt):
    key = tuple(np.round(pt, 5))
    if key not in node_dict:
        node_id = f"N{len(node_dict)}"
        node_dict[key] = node_id
        node_coords.append([node_id] + list(key))
    return node_dict[key]

# --------------------------------------------------------------
# ===== CORRECT FLOOR SEGMENTATION =====
# --------------------------------------------------------------

for x, y in all_base_columns:
    local_height = get_wall_height(x, y, all_mesh_bboxes, max_z)
    skip_ground_to_first = (round(x, 5), round(y, 5)) in columns_to_skip_ground_to_first_span

    num_floors = max(1, int(round(local_height / floor_height)))
    actual_floor_height = local_height / num_floors
    z_levels = [round(i * actual_floor_height, 5) for i in range(num_floors + 1)]

    for i in range(len(z_levels) - 1):
        start_z = z_levels[i]
        end_z = z_levels[i + 1]
        if skip_ground_to_first and abs(start_z) < 1e-4:
            continue

        id1 = add_node((x, y, start_z))
        id2 = add_node((x, y, end_z))
        column_lines.append((id1, id2))

unique_beam_tuples_3d = set()
beam_lines = []

for (x1, y1), (x2, y2) in beams_2d_lines:
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    local_height = get_wall_height(mid_x, mid_y, all_mesh_bboxes, max_z)

    num_floors = max(1, int(round(local_height / floor_height)))
    actual_floor_height = local_height / num_floors
    z_levels = [round(i * actual_floor_height, 5) for i in range(1, num_floors + 1)]

    skip1 = (round(x1, 5), round(y1, 5)) in columns_to_skip_ground_to_first_span
    skip2 = (round(x2, 5), round(y2, 5)) in columns_to_skip_ground_to_first_span

    for z in z_levels:
        if abs(z) < 1e-4:  # ⬅️ THIS LINE SKIPS GROUND LEVEL BEAMS
            continue
        if abs(z) < 1e-4 and (skip1 or skip2):
            continue

        id1 = add_node((x1, y1, z))
        id2 = add_node((x2, y2, z))
        ordered_nodes = tuple(sorted((id1, id2)))
        if ordered_nodes not in unique_beam_tuples_3d:
            unique_beam_tuples_3d.add(ordered_nodes)
            beam_lines.append((id1, id2))


# --------------------------------------------------------------
# ===== EXPORT CSV AND JSON =====
# --------------------------------------------------------------

# --------------------------------------------------------------
# --- DATA GENERATION AND JSON-ONLY EXPORT ---
# --------------------------------------------------------------

print("--- Generating data and exporting JSONs only ---")
EXPORT_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(glb_file_path)), "structural_data")
os.makedirs(EXPORT_SAVE_PATH, exist_ok=True)
print(f"Ensuring directory exists: {os.path.abspath(EXPORT_SAVE_PATH)}")

# --- Export nodes.json only ---
nodes_json_path = os.path.join(EXPORT_SAVE_PATH, "nodes.json")
nodes_json_data = [
    {"ID": n[0], "X": n[1], "Y": n[2], "Z": n[3]}
    for n in node_coords
]
with open(nodes_json_path, "w") as f:
    json.dump(nodes_json_data, f, indent=4)
print(f"✅ nodes.json written to {nodes_json_path}")

# --- Export columns.json only ---
columns_json_path = os.path.join(EXPORT_SAVE_PATH, "columns.json")
columns_json_data = []
for i, (i_node_id, j_node_id) in enumerate(column_lines):
    A = next(item for item in nodes_json_data if item["ID"] == i_node_id)
    B = next(item for item in nodes_json_data if item["ID"] == j_node_id)
    length = round(np.linalg.norm(
        (B["X"]-A["X"], B["Y"]-A["Y"], B["Z"]-A["Z"])
    ), 3)
    columns_json_data.append({
        "ID": f"C{i}",
        "i_node": i_node_id,
        "j_node": j_node_id,
        "length": length
    })
with open(columns_json_path, "w") as f:
    json.dump(columns_json_data, f, indent=4)
print(f"✅ columns.json written to {columns_json_path}")

# --- Export beams.json only ---
beams_json_path = os.path.join(EXPORT_SAVE_PATH, "beams.json")
beams_json_data = []
for i, (i_node_id, j_node_id) in enumerate(beam_lines):
    A = next(item for item in nodes_json_data if item["ID"] == i_node_id)
    B = next(item for item in nodes_json_data if item["ID"] == j_node_id)
    length = round(np.linalg.norm(
        (B["X"]-A["X"], B["Y"]-A["Y"], B["Z"]-A["Z"])
    ), 3)
    beams_json_data.append({
        "ID": f"B{i}",
        "i_node": i_node_id,
        "j_node": j_node_id,
        "length": length
    })
with open(beams_json_path, "w") as f:
    json.dump(beams_json_data, f, indent=4)
print(f"✅ beams.json written to {beams_json_path}")



# --- Combine JSON files into combinedjson.json ---
combined_json_path = os.path.join(EXPORT_SAVE_PATH, "combinedjson.json")

# Load individual JSON files
with open(nodes_json_path, 'r') as f:
    nodes_data = json.load(f)

with open(columns_json_path, 'r') as f:
    columns_data = json.load(f)

with open(beams_json_path, 'r') as f:
    beams_data = json.load(f)

# Combine into one dictionary
combined_data = {
    "nodes": nodes_data,
    "columns": columns_data,
    "beams": beams_data
}

# Write to combined JSON file
with open(combined_json_path, 'w') as f:
    json.dump(combined_data, f, indent=4)

print(f"✅ combinedjson.json written to {combined_json_path}")



# --------------------------------------
# Correct Json Viewer - Now uses the dynamically generated 3DM file and CSV paths
# --------------------------------------


try:
    with open(nodes_json_path, 'r') as f:
        nodes = json.load(f)
    df_nodes_loaded   = pd.DataFrame(nodes)

    with open(columns_json_path, 'r') as f:
        cols = json.load(f)
    df_columns_loaded = pd.DataFrame(cols)

    with open(beams_json_path, 'r') as f:
        bms = json.load(f)
    df_beams_loaded   = pd.DataFrame(bms)

    print("✅ JSON files loaded successfully for update process.")
except FileNotFoundError:
    print("Error: Make sure 'nodes.json', 'columns.json', and 'beams.json' are in the specified path.")
    exit()


if not os.path.exists(rhino_path):
    print(f"Error: Rhino file not found at {rhino_path}. Cannot identify peak roofs for re-processing.")
    exit()

model = rhino3dm.File3dm.Read(rhino_path)

# Generate 3D visualization
import plotly.graph_objects as go # Added this import

fig_3d = go.Figure()

# Add columns
for index, row in df_columns_loaded.iterrows():
    n1_coords = df_nodes_loaded[df_nodes_loaded['ID'] == row['i_node']][['X', 'Y', 'Z']].values[0]
    n2_coords = df_nodes_loaded[df_nodes_loaded['ID'] == row['j_node']][['X', 'Y', 'Z']].values[0]
    fig_3d.add_trace(go.Scatter3d(
        x=[n1_coords[0], n2_coords[0]],
        y=[n1_coords[1], n2_coords[1]],
        z=[n1_coords[2], n2_coords[2]],
        mode='lines',
        line=dict(color='blue', width=5),
        name=f'Column {row["ID"]}',
        hoverinfo='text',
        text=f'Column: {row["ID"]}<br>Length: {row["length"]:.2f}m'
    ))

# Add beams
for index, row in df_beams_loaded.iterrows():
    n1_coords = df_nodes_loaded[df_nodes_loaded['ID'] == row['i_node']][['X', 'Y', 'Z']].values[0]
    n2_coords = df_nodes_loaded[df_nodes_loaded['ID'] == row['j_node']][['X', 'Y', 'Z']].values[0]
    
    line_color = 'green'
    line_width = 3

    fig_3d.add_trace(go.Scatter3d(
        x=[n1_coords[0], n2_coords[0]],
        y=[n1_coords[1], n2_coords[1]],
        z=[n1_coords[2], n2_coords[2]],
        mode='lines',
        line=dict(color=line_color, width=line_width),
        name=f'Beam {row["ID"]}',
        hoverinfo='text',
        text=f'Beam: {row["ID"]}<br>Length: {row["length"]:.2f}m'
    ))

# Add nodes as markers
fig_3d.add_trace(go.Scatter3d(
    x=df_nodes_loaded['X'],
    y=df_nodes_loaded['Y'],
    z=df_nodes_loaded['Z'],
    mode='markers',
    marker=dict(size=4, color='black'),
    name='Nodes',
    hoverinfo='text',
    text=df_nodes_loaded['ID']
))

# Update layout for better visualization
fig_3d.update_layout(
    title='3D Structural Model with Columns (Blue), Beams (Green), and Peak Roof Beams (Red)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data' # Ensure uniform scaling
    ),
    height=800
)

fig_3d.show()

# ─────────────────────────────────────────────────────────────────
# Pynite FEM Analysis
# ─────────────────────────────────────────────────────────────────
from Pynite.Visualization import Renderer
from Pynite.FEModel3D import FEModel3D

# 1) INITIALIZE MODEL
model = FEModel3D()

# 2) ADD NODES (convert m → in)
UNIT_CONVERSION = 39.37  # meters to inches
for _, row in df_nodes_loaded.iterrows():
    nid = str(row['ID'])
    model.add_node(
        nid,
        row['X'] * UNIT_CONVERSION,
        row['Y'] * UNIT_CONVERSION,
        row['Z'] * UNIT_CONVERSION
    )

# 3) MATERIAL & SECTION
f_c   = 4.0                          # ksi
E_c   = 57000 * (f_c ** 0.5)         # ksi
G_c   = E_c / (2 * (1 + 0.2))        # ksi
rho_c = (150 / 1728) * 1e-3          # kip/in^3
model.add_material('RCC', E_c, G_c, f_c, rho_c)

A, Iy, Iz, J = 8.84, 49.0, 14.4, 0.95
model.add_section('W14X30', A, Iy, Iz, J)

# 4) ADD MEMBERS
for _, row in df_beams_loaded.iterrows():
    bid = str(row['ID'])
    model.add_member(bid,
                     str(row['i_node']),
                     str(row['j_node']),
                     'RCC', 'W14X30')

for _, row in df_columns_loaded.iterrows():
    cid = str(row['ID'])
    model.add_member(cid,
                     str(row['i_node']),
                     str(row['j_node']),
                     'RCC', 'W14X30')

# 5) SUPPORTS (fixed at Z=0)
for _, row in df_nodes_loaded.iterrows():
    fixed = (row['Z'] == 0)
    model.def_support(
        str(row['ID']),
        fixed, fixed, fixed,  # Tx,Ty,Tz  
        fixed, fixed, fixed   # Rx,Ry,Rz
    )

# 6) MEMBER LENGTHS lookup
import numpy as np
beam_lengths = {}
column_lengths = {}

for _, row in df_beams_loaded.iterrows():
    mid = str(row['ID'])
    iN, jN = str(row['i_node']), str(row['j_node'])
    xi, yi, zi = model.nodes[iN].X, model.nodes[iN].Y, model.nodes[iN].Z
    xj, yj, zj = model.nodes[jN].X, model.nodes[jN].Y, model.nodes[jN].Z
    beam_lengths[mid] = float(np.sqrt((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2))

for _, row in df_columns_loaded.iterrows():
    mid = str(row['ID'])
    iN, jN = str(row['i_node']), str(row['j_node'])
    xi, yi, zi = model.nodes[iN].X, model.nodes[iN].Y, model.nodes[iN].Z
    xj, yj, zj = model.nodes[jN].X, model.nodes[jN].Y, model.nodes[jN].Z
    column_lengths[mid] = float(np.sqrt((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2))

# 7) DISTRIBUTED LOADS
length_lookup = {**beam_lengths, **column_lengths}
for mid in model.members:
    L = length_lookup[mid]
    if mid in beam_lengths:
        w_DL, w_LL = -0.02, -0.015
    else:
        w_DL, w_LL = -0.01, -0.008
    model.add_member_dist_load(mid, 'FZ', w_DL, w_LL, 0, L, case='DL')
    model.add_member_dist_load(mid, 'FZ', w_LL, w_LL, 0, L, case='LL')

# 8) LOAD COMBINATIONS
model.add_load_combo('0.9DL+1.0W', factors={'DL': 1.2, 'W': 1.0})
model.add_load_combo('1.2DL+1.0W', factors={'DL': 0.9, 'W': 1.0})

# 9) ANALYZE
model.analyze()

# 10) DEFORMED SHAPE
rndr = Renderer(model)
rndr.deformed_shape = True
rndr.deformed_scale = 50
rndr.combo_name     = '1.2DL+1.0W'
rndr.show_loads     = True
rndr.show_reactions = True
rndr.window_width   = 750
rndr.window_height  = 750
rndr.render_model()

# 11) SLENDERNESS CHECKS
print("\nCOLUMN SLENDERNESS (L/r):")
sec = model.sections['W14X30']
r_y = np.sqrt(sec.Iy / sec.A)
r_z = np.sqrt(sec.Iz / sec.A)
r   = min(r_y, r_z)
for _, row in df_columns_loaded.iterrows():
    cid = str(row['ID'])
    L   = column_lengths[cid]
    sl  = L / r
    ok  = "PASS" if sl <= 200 else "FAIL"
    print(f"  Col {cid}: L/r={sl:.1f} ({ok})")

print("\nBEAM SLENDERNESS (L/r):")
for _, row in df_beams_loaded.iterrows():
    bid = str(row['ID'])
    L   = beam_lengths[bid]
    sl  = L / r
    ok  = "PASS" if sl <= 300 else "FAIL"
    print(f"  Beam {bid}: L/r={sl:.1f} ({ok})")

# 12) STIFFNESS & DISPLACEMENTS
K = model.K(combo_name='1.2DL+1.0W', log=True, check_stability=True, sparse=True).toarray()
print(f"\nGlobal K matrix ({K.shape[0]}×{K.shape[1]}):")
labels = [f"{nid}_{dof}" for nid in model.nodes for dof in ('DX','DY','DZ','RX','RY','RZ')]
import pandas as pd
K_df = pd.DataFrame(K, index=labels, columns=labels)
print(K_df)

print("\nNODAL DISPLACEMENTS:")
for nid, node in model.nodes.items():
    dx = node.DX.get('1.2DL+1.0W', 0.0)
    dy = node.DY.get('1.2DL+1.0W', 0.0)
    dz = node.DZ.get('1.2DL+1.0W', 0.0)
    print(f"  {nid}: DX={dx:.6e} in, DY={dy:.6e} in, DZ={dz:.6e} in")
