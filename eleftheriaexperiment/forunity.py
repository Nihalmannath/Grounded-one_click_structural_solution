# 1. Imports for GLB conversion
import trimesh
import rhino3dm
import numpy as np

# Imports for structural analysis and visualization
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pyvista as pv

# --- GLB to 3DM Conversion Section ---

# Define GLB input path and 3dm output path
glb_path = "Cuboid_5m.glb" # Ensure this GLB file exists in the same directory
out_3dm = "for_Structure.3dm"

# Check if the GLB file exists before proceeding
if not os.path.exists(glb_path):
    raise FileNotFoundError(f"GLB file not found: {glb_path}")

print(f"Loading GLB file: {glb_path}")
# 1. load GLB – force a scene so we get all meshes + transforms
scene = trimesh.load(glb_path, force='scene')

# 2. prepare an empty 3DM container
doc = rhino3dm.File3dm()

for name, tm in scene.geometry.items():
    # Apply the mesh’s transform (if part of a node)
    xform = scene.graph.get(name)[0]
    if xform is not None:
        tm.apply_transform(xform)

    rh_mesh = rhino3dm.Mesh()

    # vertices
    for v in tm.vertices:
        rh_mesh.Vertices.Add(*v)

    # faces (tri + quad)
    for f in tm.faces:
        if len(f) == 3:
            rh_mesh.Faces.AddFace(f[0], f[1], f[2])
        else:
            rh_mesh.Faces.AddFace(f[0], f[1], f[2], f[3])

    # optional: vertex colours
    if tm.visual.kind == "vertex" and tm.visual.vertex_colors.shape[1] >= 3:
        for c in tm.visual.vertex_colors:
            rh_mesh.VertexColors.Add(int(c[0]), int(c[1]), int(c[2]), 255)

    rh_mesh.Normals.ComputeNormals()
    rh_mesh.Compact()

    # add to the 3dm document
    attrs = rhino3dm.ObjectAttributes()
    attrs.Name = name
    doc.Objects.AddMesh(rh_mesh, attrs)

# 3. write the finished file (version 7 archive)
doc.Write(out_3dm, 7)
print(f"Successfully converted {glb_path} to {out_3dm}")

# --- End GLB to 3DM Conversion Section ---


# Now, use the newly created 3dm file as the input for the structural analysis
rhino_path = out_3dm

# Load model (this will now load the converted 3dm file)
model = rhino3dm.File3dm.Read(rhino_path)

# --- Geometry Extraction (No Layer Assumptions) ---
building_volumes = []
max_z = 0.0
wall_breps = []

for obj in model.Objects:
    geom = obj.Geometry

    # We only care about Breps for building volumes.
    # Note: If the GLB only contains meshes, we'll need to convert them to Breps or
    # derive bounding box from mesh directly. For simplicity, assuming the
    # generated 3dm might contain Breps if original GLB did, or we handle Meshes as well.
    # For now, let's try to convert Mesh to Brep if possible or get bbox directly from Mesh.
    bbox_candidate = None
    if geom.ObjectType == rhino3dm.ObjectType.Brep:
        bbox_candidate = geom.GetBoundingBox()
    elif geom.ObjectType == rhino3dm.ObjectType.Mesh:
        # For meshes, we can get the bounding box directly
        bbox_candidate = geom.GetBoundingBox()
        # Optionally, convert mesh to Brep if more accurate polygon representation is needed.
        # This can be computationally intensive and might not always result in a valid Brep.
        # For now, we proceed with the mesh's bbox to define the footprint.
        # If the structural logic heavily relies on Brep properties beyond bbox,
        # further mesh-to-Brep conversion or direct mesh processing would be needed.

    if bbox_candidate and bbox_candidate.IsValid:
        # Heuristic: Filter for reasonably sized bounding boxes to be considered building geometry.
        # This helps ignore small details or artifacts. Adjust thresholds if needed.
        dx = bbox_candidate.Max.X - bbox_candidate.Min.X
        dy = bbox_candidate.Max.Y - bbox_candidate.Min.Y
        dz = bbox_candidate.Max.Z - bbox_candidate.Min.Z

        if dx > 1 and dy > 1 and dz > 1: # Example thresholds for a "building" size
            base_pts = [
                [bbox_candidate.Min.X, bbox_candidate.Min.Y],
                [bbox_candidate.Max.X, bbox_candidate.Min.Y],
                [bbox_candidate.Max.X, bbox_candidate.Max.Y],
                [bbox_candidate.Min.X, bbox_candidate.Max.Y],
                [bbox_candidate.Min.X, bbox_candidate.Min.Y] # Ensure polygon is closed
            ]
            poly = Polygon(base_pts)

            if poly.is_valid and not poly.is_empty:
                building_volumes.append(poly)
                # Store the original geometry and its bbox for wall visualization
                wall_breps.append({'geometry': geom, 'bbox': bbox_candidate})
                max_z = max(max_z, bbox_candidate.Max.Z)

# Critical check: a building volume must be found to proceed
if not building_volumes:
    raise RuntimeError("No valid building geometry found. Please ensure your GLB model contains identifiable building volumes (e.g., large Meshes or Breps).")

# Ask for number of floors
while True:
    try:
        num_floors = int(input("How many floors does the building have? (e.g., 2): "))
        if num_floors < 1:
            raise ValueError
        break
    except ValueError:
        print("Please enter a valid positive integer for the number of floors.")

# Room sorting (useful if the building is composed of multiple parts)
detected_rooms = sorted([(poly, poly.area) for poly in building_volumes], key=lambda x: -x[1])

# Structural logic
MaxS = 6.0 # Maximum span for a beam
MinS = 3.0 # Minimum spacing between columns

columns = []
beams = []
existing_columns = [] # Tracks positions of all placed columns to ensure minimum spacing

for room_poly, _ in detected_rooms:
    minx, miny, maxx, maxy = room_poly.bounds
    width, height = maxx - minx, maxy - miny

    # Calculate divisions for the structural grid
    divisions_x = max(1, int(np.ceil(width / MaxS))) # Ensure at least one division if width > 0
    divisions_y = max(1, int(np.ceil(height / MaxS))) # Ensure at least one division if height > 0

    x_points = np.linspace(minx, maxx, divisions_x + 1)
    y_points = np.linspace(miny, maxy, divisions_y + 1)

    # Generate grid points for columns within the room
    for x in x_points:
        for y in y_points:
            candidate_col = (x, y)
            point = Point(candidate_col)
            # Check if the candidate point is within the room polygon
            if room_poly.contains(point) or room_poly.exterior.distance(point) < 0.01: # Small buffer for boundary points
                # Add column if it's not too close to an already existing one
                if all(np.linalg.norm(np.array(candidate_col) - np.array(existing_col)) >= MinS for existing_col in existing_columns):
                    columns.append(candidate_col)
                    existing_columns.append(candidate_col) # Add to existing for future checks

    # Generate beams along the grid lines
    # Vertical beams
    for x in x_points:
        beams.append(((x, miny), (x, maxy)))
    # Horizontal beams
    for y in y_points:
        beams.append(((minx, y), (maxx, y)))

    # Ensure columns at corners if they weren't naturally placed by the grid due to spacing
    for corner_x, corner_y in list(room_poly.exterior.coords):
        corner_pt = (corner_x, corner_y)
        # Check if a column is already very close to this corner
        if all(np.linalg.norm(np.array(corner_pt) - np.array(exist_col)) >= MinS * 0.5 for exist_col in existing_columns):
            columns.append(corner_pt)
            existing_columns.append(corner_pt)


# All columns are generated columns
all_base_columns = columns

---

### 2D Visualization

fig, ax = plt.subplots(figsize=(10, 10))

# Room outlines
for poly, _ in detected_rooms:
    px, py = poly.exterior.xy
    ax.plot(px, py, 'k-', linewidth=1)

# Generated columns
if all_base_columns:
    gx, gy = zip(*all_base_columns)
    ax.scatter(gx, gy, c='blue', s=80, label='Generated Columns', zorder=5)

intermediate_label_added = False
roof_label_added = False

for (x1, y1), (x2, y2) in beams:
    if num_floors > 1:
        for floor in range(1, num_floors):  # intermediate floors
            ax.plot([x1, x2], [y1, y2], color='orange', linestyle=':', linewidth=1,
                            label='Intermediate Floor Beams' if not intermediate_label_added else "")
            intermediate_label_added = True

    ax.plot([x1, x2], [y1, y2], color='green', linestyle='--', linewidth=1.5,
                    label='Roof Beams' if not roof_label_added else "")
    roof_label_added = True

ax.set_aspect('equal', 'box')
ax.legend()
plt.title("2D Generated Column and Beam Grid")
ax.grid(True)
plt.show()

---

### 3D Visualization

def get_wall_height(x, y, wall_data):
    """
    Determines the height of the building at a given (x,y) point by finding the
    closest bounding box from the detected building volumes.
    """
    pt = np.array([x, y])
    closest_wall_bbox = None
    closest_dist = float('inf')

    for wall in wall_data:
        poly = wall['polygon'] # Note: `wall_data` now stores polygons and bboxes directly.
        dist = poly.exterior.distance(Point(x, y))
        if dist < closest_dist:
            closest_dist = dist
            closest_wall_bbox = wall['bbox']

    if closest_wall_bbox:
        return closest_wall_bbox.Max.Z
    else:
        return max_z # Fallback to the overall maximum Z if no specific wall is found (shouldn't happen often)


plotter = pv.Plotter(title="3D Structural System")

# Columns as vertical cylinders
for x, y in all_base_columns:
    wall_height = get_wall_height(x, y, wall_breps)
    # Create cylinders from Min.Z to Max.Z of the building height at that point
    cylinder = pv.Cylinder(center=(x, y, wall_height / 2), direction=(0, 0, 1),
                            radius=0.1, height=wall_height)
    plotter.add_mesh(cylinder, color='blue')

# Beams as thinner cylinders
beam_radius = 0.04

for (x1, y1), (x2, y2) in beams:
    # Determine the effective height for beams based on the lowest point of the two beam ends
    h1 = get_wall_height(x1, y1, wall_breps)
    h2 = get_wall_height(x2, y2, wall_breps)
    effective_building_height = min(h1, h2)

    # Place beams at each floor level
    for floor in range(1, num_floors + 1):
        # Calculate Z-coordinate for the current floor
        z = effective_building_height / num_floors * floor

        # Avoid placing beams outside the actual building height if the building has varying rooflines
        if z > h1 or z > h2:
            continue

        start = np.array([x1, y1, z])
        end = np.array([x2, y2, z])

        # Calculate beam direction, length, and center
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6: # Skip if points are too close
            continue
        direction = direction / length
        center = (start + end) / 2

        # Create and add the beam cylinder to the plotter
        beam = pv.Cylinder(center=center, direction=direction, radius=beam_radius, height=length)

        # Color beams differently for the roof level
        color = 'green' if floor == num_floors else 'orange'
        plotter.add_mesh(beam, color=color)

# Visualize the actual building geometry (from the converted 3dm)
def mesh_rhino_geometry(rhino_geom_obj, mesh_type=rhino3dm.MeshType.Any):
    """Converts a rhino3dm.GeometryBase object to a list of PyVista meshes."""
    meshes_to_plot = []
    if isinstance(rhino_geom_obj, rhino3dm.Brep):
        for face in rhino_geom_obj.Faces:
            try:
                m = face.GetMesh(mesh_type)
                if m: meshes_to_plot.append(m)
            except Exception:
                continue
    elif isinstance(rhino_geom_obj, rhino3dm.Mesh):
        meshes_to_plot.append(rhino_geom_obj) # If it's already a mesh, just add it

    pv_meshes = []
    for mesh in meshes_to_plot:
        pts = [(v.X, v.Y, v.Z) for v in mesh.Vertices]
        faces = []
        for f in mesh.Faces:
            if len(f) == 4:
                idxs = (f[0], f[1], f[2], f[3])
            else:
                idxs = (f[0], f[1], f[2])
            faces.append((len(idxs),) + idxs)
        faces_flat = [i for face in faces for i in face]
        pv_meshes.append(pv.PolyData(pts, faces_flat))
    return pv_meshes

# Iterate through the objects in the model for visualization
for wall_info in wall_breps: # wall_breps now stores the original geometry object
    geom_to_viz = wall_info['geometry']
    pv_meshes = mesh_rhino_geometry(geom_to_viz)
    for pv_mesh in pv_meshes:
        plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.3)

# Final setup for the 3D viewer
plotter.show_grid()
plotter.show()