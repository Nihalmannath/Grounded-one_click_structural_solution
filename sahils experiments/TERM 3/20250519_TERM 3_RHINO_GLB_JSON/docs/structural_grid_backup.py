#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Structural Grid Generator
------------------------
This script generates 3D structural grids from mesh files,
with support for:
- Vertical columns
- Horizontal beams
- Filtering to keep elements inside geometry only
- JSON export of structural elements
- Interactive grid adjustment
"""

import os
import sys
import numpy as np
import trimesh
import pyvista as pv
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk


class StructuralGridGenerator:
    def __init__(self):
        self.mesh = None
        self.voxel_pitch = None  # Spacing between grid points
        self.num_floors = 3      # Default number of horizontal divisions (floors)
        self.grid_points = None  # All grid points
        self.columns = []        # Vertical column lines (point pairs)
        self.beams = []          # Horizontal beam lines (point pairs)
        self.internal_beams = [] # Internal cross beams
        self.diagonals = []      # Diagonal bracing connections
        self.plotter = None
        self.filtered_points = None  # Points inside mesh
        
        # Grid adjustment parameters
        self.grid_offset_x = 0.0  # Grid offset in X direction
        self.grid_offset_y = 0.0  # Grid offset in Y direction
        self.grid_spacing_x = None  # Grid spacing in X direction (if different from voxel_pitch)
        self.grid_spacing_y = None  # Grid spacing in Y direction (if different from voxel_pitch)
        self.is_grid_moved = False  # Flag to track if grid has been moved
          def load_mesh(self, filepath):
        """Load a mesh file using trimesh, with added support for GLB files."""
        try:
            # Check if it's a GLB file
            if filepath.lower().endswith('.glb'):
                # For GLB files, we need to extract the scene and merge all meshes
                scene = trimesh.load(filepath)
                
                # If it's a scene, extract all meshes and merge them
                if isinstance(scene, trimesh.Scene):
                    meshes = []
                    for name, geometry in scene.geometry.items():
                        if isinstance(geometry, trimesh.Trimesh):
                            # Apply the transform from the scene
                            transform = scene.graph.get(name)[0]
                            geometry.apply_transform(transform)
                            meshes.append(geometry)
                            
                    # Combine all meshes into a single mesh
                    if meshes:
                        self.mesh = trimesh.util.concatenate(meshes)
                    else:
                        print("No valid meshes found in GLB file")
                        return False
                else:
                    # If it's already a mesh, use it directly
                    self.mesh = scene
            else:
                # For other file formats, use the standard loader
                self.mesh = trimesh.load(filepath, force='mesh', process=True)
                
            print(f"Loaded mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces")
            
            # Compute basic mesh statistics
            if not self.mesh.is_watertight:
                print("Warning: Mesh is not watertight. Point filtering may be less accurate.")
            
            print(f"Mesh volume: {self.mesh.volume:.2f} cubic units")
            print(f"Mesh bounding box: {self.mesh.bounds}")
            
            return True
        except Exception as e:
            print(f"Error loading mesh: {str(e)}")
            return False
            
    def generate_grid(self, offset_x=None, offset_y=None, spacing_x=None, spacing_y=None):
        """Generate a grid of points within the mesh's bounding box with optional offsets and custom spacing."""
        if self.mesh is None:
            print("No mesh loaded!")
            return False
            
        # Apply offsets if provided
        self.grid_offset_x = offset_x if offset_x is not None else self.grid_offset_x
        self.grid_offset_y = offset_y if offset_y is not None else self.grid_offset_y
        self.grid_spacing_x = spacing_x if spacing_x is not None else self.grid_spacing_x
        self.grid_spacing_y = spacing_y if spacing_y is not None else self.grid_spacing_y
            
        # Calculate grid spacing if not specified
        if self.voxel_pitch is None:
            # Default to 1/20th of the largest dimension
            bounds = self.mesh.bounds
            dims = bounds[1] - bounds[0]
            max_dim = np.max(dims)
            self.voxel_pitch = max_dim / 20
        
        # Get effective spacing values (use custom if set, otherwise default)
        effective_spacing_x = self.grid_spacing_x if self.grid_spacing_x is not None else self.voxel_pitch
        effective_spacing_y = self.grid_spacing_y if self.grid_spacing_y is not None else self.voxel_pitch
            
        # Get mesh bounds for grid extents
        min_corner, max_corner = self.mesh.bounds
        
        # Calculate number of grid points in each dimension
        nx = int(np.ceil((max_corner[0] - min_corner[0]) / effective_spacing_x)) + 1
        ny = int(np.ceil((max_corner[1] - min_corner[1]) / effective_spacing_y)) + 1
        nz = self.num_floors  # Use specified number of floors for Z
        
        # Create evenly spaced grid with offsets
        x = np.linspace(min_corner[0], max_corner[0], nx) + self.grid_offset_x
        y = np.linspace(min_corner[1], max_corner[1], ny) + self.grid_offset_y
        z = np.linspace(min_corner[2], max_corner[2], nz)
        
        # Create 3D grid points
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.grid_points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
        
        # Store grid dimensions for structural element generation
        self.grid_dims = (nx, ny, nz)
        
        print(f"Generated grid with {len(self.grid_points)} points, dimensions: {nx}x{ny}x{nz}")
        print(f"Grid offsets: X={self.grid_offset_x:.2f}, Y={self.grid_offset_y:.2f}")
        print(f"Grid spacing: X={effective_spacing_x:.2f}, Y={effective_spacing_y:.2f}")
        
        # Set flag to indicate grid has been moved if offsets or custom spacing are used
        if self.grid_offset_x != 0 or self.grid_offset_y != 0 or self.grid_spacing_x is not None or self.grid_spacing_y is not None:
            self.is_grid_moved = True
            
        return True
        
    def filter_points(self):
        """Filter grid points to keep only those inside or near the mesh."""
        if self.mesh is None or self.grid_points is None:
            print("Mesh or grid points not available!")
            return False
            
        # Use different methods based on mesh properties
        if self.mesh.is_watertight:
            # For watertight meshes, check if points are contained
            inside_mask = self.mesh.contains(self.grid_points)
        else:
            # For non-watertight meshes, use proximity-based approach
            closest_points, distances, _ = trimesh.proximity.closest_point(self.mesh, self.grid_points)
            threshold = self.mesh.scale * 0.01  # Small fraction of mesh scale
            inside_mask = distances < threshold
            
        self.filtered_points = self.grid_points[inside_mask]
        
        # Store filtered point indices for faster structure generation
        self.filtered_indices = np.where(inside_mask)[0]
        
        print(f"Filtered grid to {len(self.filtered_points)} points inside the mesh")
        return True
    
    def filter_points_by_floor(self):
        """Organize filtered points into separate floors for grid generation."""
        if self.filtered_points is None or len(self.filtered_points) == 0:
            print("No filtered points available!")
            return False
            
        # Get grid dimensions and floor height
        _, _, nz = self.grid_dims
        min_corner, max_corner = self.mesh.bounds
        floor_height = (max_corner[2] - min_corner[2]) / (nz - 1) if nz > 1 else 0
        
        # Organize points by floor
        self.floor_points = [[] for _ in range(nz)]
        self.floor_indices = [[] for _ in range(nz)]
        
        # Assign points to floors
        for idx in self.filtered_indices:
            point = self.grid_points[idx]
            # Calculate floor based on Z coordinate
            floor_idx = int(round((point[2] - min_corner[2]) / floor_height)) if floor_height > 0 else 0
            floor_idx = min(max(0, floor_idx), nz - 1)  # Clamp to valid range
            
            self.floor_points[floor_idx].append(point)
            self.floor_indices[floor_idx].append(idx)
            
        # Report points per floor
        for i, points in enumerate(self.floor_points):
            print(f"Floor {i+1}: {len(points)} points")
            
        return True
    
    def generate_structures(self):
        """Generate vertical columns and horizontal beams in the grid."""
        if self.filtered_points is None or len(self.filtered_points) == 0:
            print("No filtered points available!")
            return False
            
        # Clear existing structures
        self.columns = []
        self.beams = []
        
        # Get grid dimensions
        nx, ny, nz = self.grid_dims
        
        # Helper function to get index in the 3D grid
        def get_grid_index(i, j, k):
            return i * ny * nz + j * nz + k
            
        # Helper function to check if a point is in filtered set
        point_set = set(map(tuple, self.filtered_points.round(decimals=6)))
        
        def is_point_filtered(point):
            return tuple(np.round(point, decimals=6)) in point_set
        
        # Generate columns (vertical connections between floors)
        for i in range(nx):
            for j in range(ny):
                column_points = []
                for k in range(nz):
                    idx = get_grid_index(i, j, k)
                    if idx in self.filtered_indices:
                        column_points.append(self.grid_points[idx])
                        
                # Create column segments if we have more than one point
                for p in range(len(column_points) - 1):
                    self.columns.append((column_points[p], column_points[p+1]))
        
        # Generate beams (horizontal connections on each floor)
        for k in range(nz):  # For each floor
            for i in range(nx):
                for j in range(ny-1):  # Connect in Y direction
                    idx1 = get_grid_index(i, j, k)
                    idx2 = get_grid_index(i, j+1, k)
                    if idx1 in self.filtered_indices and idx2 in self.filtered_indices:
                        self.beams.append((self.grid_points[idx1], self.grid_points[idx2]))
            
            for i in range(nx-1):  # Connect in X direction
                for j in range(ny):
                    idx1 = get_grid_index(i, j, k)
                    idx2 = get_grid_index(i+1, j, k)
                    if idx1 in self.filtered_indices and idx2 in self.filtered_indices:
                        self.beams.append((self.grid_points[idx1], self.grid_points[idx2]))
        
        print(f"Generated {len(self.columns)} columns and {len(self.beams)} beams")
        return True
        
    def visualize(self, interactive=True):
        """Visualize the mesh, grid points, columns, and beams with interactive controls for grid movement."""
        if self.mesh is None:
            print("No mesh loaded!")
            return False
            
        try:
            # Create plotter
            self.plotter = pv.Plotter()
            
            # Add the mesh as translucent
            vertices = self.mesh.vertices
            faces = np.hstack([[3, f[0], f[1], f[2]] for f in self.mesh.faces])
            pv_mesh = pv.PolyData(vertices, faces)
            self.plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.3, show_edges=True)
            
            # Add all grid points (even before filtering) with lower opacity
            if self.grid_points is not None and len(self.grid_points) > 0:
                grid_array = np.array(self.grid_points, dtype=float)
                all_points = pv.PolyData(grid_array)
                self.plotter.add_mesh(all_points, color='lightblue', opacity=0.3, 
                                    render_points_as_spheres=True, point_size=6, 
                                    name="grid_points")  # Add a name for retrieval
            
            # Add filtered grid points
            if self.filtered_points is not None and len(self.filtered_points) > 0:
                # Make sure filtered_points is proper 2D array
                points_array = np.array(self.filtered_points, dtype=float)
                if points_array.ndim > 2:
                    print(f"Warning: filtered_points has {points_array.ndim} dimensions, reshaping to 2D")
                    points_array = points_array.reshape(-1, 3)
                points = pv.PolyData(points_array)
                self.plotter.add_mesh(points, color='red', render_points_as_spheres=True, point_size=10,
                                     name="filtered_points")  # Add a name for retrieval
            
            # Add columns (vertical structural elements)
            for i, col in enumerate(self.columns):
                try:
                    # Ensure col is a tuple of two valid 3D points
                    if len(col) != 2:
                        print(f"Warning: skipping invalid column with {len(col)} points")
                        continue
                        
                    col_start = np.array(col[0], dtype=float)
                    col_end = np.array(col[1], dtype=float)
                    
                    # Verify both points are proper 3D points (1D arrays)
                    if col_start.ndim > 1:
                        col_start = col_start.flatten()[:3]
                    if col_end.ndim > 1:
                        col_end = col_end.flatten()[:3]
                        
                    # Create 2D array with start and end points
                    line_points = np.vstack([col_start, col_end])
                    line = pv.Line(line_points[0], line_points[1])
                    self.plotter.add_mesh(line, color='blue', line_width=5, name=f"column_{i}")
                except Exception as e:
                    print(f"Error adding column: {e}")
                    continue
                    
            # Add beams (horizontal structural elements)
            for i, beam in enumerate(self.beams):
                try:
                    # Ensure beam is a tuple of two valid 3D points
                    if len(beam) != 2:
                        print(f"Warning: skipping invalid beam with {len(beam)} points")
                        continue
                        
                    beam_start = np.array(beam[0], dtype=float)
                    beam_end = np.array(beam[1], dtype=float)
                    
                    # Verify both points are proper 3D points (1D arrays)
                    if beam_start.ndim > 1:
                        beam_start = beam_start.flatten()[:3]
                    if beam_end.ndim > 1:
                        beam_end = beam_end.flatten()[:3]
                        
                    # Create 2D array with start and end points
                    line_points = np.vstack([beam_start, beam_end])
                    line = pv.Line(line_points[0], line_points[1])
                    self.plotter.add_mesh(line, color='green', line_width=3, name=f"beam_{i}")
                except Exception as e:
                    print(f"Error adding beam: {e}")
                    continue
            
            # Add bounding box and axes
            self.plotter.add_bounding_box()
            self.plotter.add_axes()
            
            # Add interactive grid movement controls if requested
            if interactive:
                # Add sliders for grid adjustment
                def update_offset_x(offset):
                    self.update_grid_position(offset_x=offset)
                    
                def update_offset_y(offset):
                    self.update_grid_position(offset_y=offset)
                    
                def update_spacing_x(spacing):
                    self.update_grid_position(spacing_x=spacing)
                    
                def update_spacing_y(spacing):
                    self.update_grid_position(spacing_y=spacing)
                
                # Add sliders with current values
                min_corner, max_corner = self.mesh.bounds
                mesh_width = max_corner[0] - min_corner[0]
                mesh_depth = max_corner[1] - min_corner[1]
                
                # Set slider ranges based on mesh dimensions
                offset_range = [-mesh_width/2, mesh_width/2]
                spacing_range = [self.voxel_pitch/2, self.voxel_pitch*2]
                
                self.plotter.add_slider_widget(
                    update_offset_x, 
                    [offset_range[0], offset_range[1]], 
                    value=self.grid_offset_x,
                    title="Grid Offset X",
                    pointa=(0.1, 0.1), 
                    pointb=(0.4, 0.1)
                )
                
                self.plotter.add_slider_widget(
                    update_offset_y, 
                    [offset_range[0], offset_range[1]], 
                    value=self.grid_offset_y,
                    title="Grid Offset Y",
                    pointa=(0.6, 0.1), 
                    pointb=(0.9, 0.1)
                )
                
                self.plotter.add_slider_widget(
                    update_spacing_x, 
                    spacing_range, 
                    value=self.grid_spacing_x or self.voxel_pitch,
                    title="Grid Spacing X",
                    pointa=(0.1, 0.05), 
                    pointb=(0.4, 0.05)
                )
                
                self.plotter.add_slider_widget(
                    update_spacing_y, 
                    spacing_range, 
                    value=self.grid_spacing_y or self.voxel_pitch,
                    title="Grid Spacing Y",
                    pointa=(0.6, 0.05), 
                    pointb=(0.9, 0.05)
                )
                
                # Add a button to apply grid changes
                self.plotter.add_text("Use sliders to adjust grid position and spacing, then press 'A' to apply grid changes",
                                    position=(0.5, 0.95), font_size=10, color='white', 
                                    shadow=True, name='instructions', font='arial')
                  # Instead of a button widget, use a keyboard event
                def apply_changes_callback():
                    self.regenerate_structure()
                
                self.plotter.add_key_event('a', apply_changes_callback)
                self.plotter.add_text("Press 'A' to Apply Grid Changes", 
                                    position=(0.5, 0.15), font_size=12, 
                                    color='yellow', shadow=True, name='apply_button', font='arial')
            
            # Show the plot
            self.plotter.show()
            return True
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_grid_position(self, offset_x=None, offset_y=None, spacing_x=None, spacing_y=None):
        """Update the preview of grid position without regenerating structure."""
        # Update offset and spacing values if provided
        if offset_x is not None:
            self.grid_offset_x = offset_x
        if offset_y is not None:
            self.grid_offset_y = offset_y
        if spacing_x is not None:
            self.grid_spacing_x = spacing_x
        if spacing_y is not None:
            self.grid_spacing_y = spacing_y
        
        # Mark grid as moved
        self.is_grid_moved = True
        
        # Update the display text to show current values
        if hasattr(self, 'plotter') and self.plotter is not None:
            info_text = f"Grid Offset: X={self.grid_offset_x:.2f}, Y={self.grid_offset_y:.2f} | "
            info_text += f"Grid Spacing: X={self.grid_spacing_x or self.voxel_pitch:.2f}, Y={self.grid_spacing_y or self.voxel_pitch:.2f}"
            
            if self.plotter.add_text(info_text, position=(0.5, 0.9), 
                               font_size=10, color='white', shadow=True, 
                               name='grid_info', font='arial'):
                self.plotter.render()
        
        return True
    def regenerate_structure(self):
        """Regenerate the structural grid with current offset and spacing values."""
        if not self.is_grid_moved:
            # No changes to apply
            return
            
        try:
            # Clear current visualization
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.close()
                
            # Regenerate grid with current offset and spacing
            self.generate_grid()
            
            # Regenerate filtered points and structures
            self.filter_points()
            self.generate_structures()
            
            # Show information about the regenerated structure
            print(f"Regenerated grid with offsets X={self.grid_offset_x:.2f}, Y={self.grid_offset_y:.2f}")
            print(f"Spacing: X={self.grid_spacing_x or self.voxel_pitch:.2f}, Y={self.grid_spacing_y or self.voxel_pitch:.2f}")
            print(f"New structure has {len(self.columns)} columns and {len(self.beams)} beams")
            
            # Show the updated visualization
            self.visualize()
            
            # Announce completion
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.add_text("Structure regenerated successfully", 
                                  position=(0.5, 0.85), font_size=12, 
                                  color='lightgreen', shadow=True, name='regen_msg', font='arial')
                self.plotter.render()
                  return True
            
        except Exception as e:
            print(f"Error regenerating structure: {e}")
            import traceback
            traceback.print_exc()
            return False
              def export_structures(self, filepath):
        """Export columns and beams to a JSON file with element types."""
        if (not self.columns and not self.beams) or self.filtered_points is None:
            print("No structural elements to export!")
            return False
            
        try:
            import json
            from datetime import datetime
            
            # Prepare data structure for JSON - simplified format to match example
            structure_data = {
                "elements": {
                    "beams": []
                }
            }
            
            # Add columns data as beams
            for start, end in self.columns:
                structure_data["elements"]["beams"].append({
                    "start": [float(start[0]), float(start[1]), float(start[2])],
                    "end": [float(end[0]), float(end[1]), float(end[2])]
                })
                
            # Add beams data
            for start, end in self.beams:
                structure_data["elements"]["beams"].append({
                    "start": [float(start[0]), float(start[1]), float(start[2])],
                    "end": [float(end[0]), float(end[1]), float(end[2])]
                })
            
            # Write to JSON file
            with open(filepath, 'w') as f:
                json.dump(structure_data, f, indent=4)
                
            print(f"Exported {len(self.columns)} columns and {len(self.beams)} beams to {filepath} in JSON format")
            return True
        except Exception as e:
            print(f"Error exporting structures: {e}")
            return False
        
    def load_multiple_meshes(self, filepaths):
        """Load multiple mesh files and combine them for structural analysis."""
        if not filepaths:
            print("No files provided to load!")
            return False
            
        try:
            # Clear previous meshes
            self.meshes = []
            self.mesh_names = []
            
            # Prioritize loading column meshes first
            # This will help ensure column positions are used as grid references
            sorted_filepaths = sorted(filepaths, key=lambda path: 0 if "column" in os.path.basename(path).lower() else 1)
            
            # Load each mesh
            for filepath in sorted_filepaths:
                try:
                    mesh = trimesh.load(filepath, force='mesh', process=True)
                    self.meshes.append(mesh)
                    self.mesh_names.append(os.path.basename(filepath))
                    print(f"Loaded {os.path.basename(filepath)} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
                    
                    # Log priority status for columns
                    if "column" in os.path.basename(filepath).lower():
                        print(f"Prioritizing {os.path.basename(filepath)} as reference for structural grid")
                except Exception as e:
                    print(f"Error loading {os.path.basename(filepath)}: {str(e)}")
            
            if not self.meshes:
                print("Failed to load any meshes!")
                return False
                
            # Combine meshes if more than one was loaded
            if len(self.meshes) > 1:
                vertices_list = []
                faces_list = []
                vertex_offset = 0
                
                for mesh in self.meshes:
                    vertices_list.append(mesh.vertices)
                    # Offset faces indices for the combined mesh
                    current_faces = mesh.faces.copy()
                    current_faces += vertex_offset
                    faces_list.append(current_faces)
                    vertex_offset += len(mesh.vertices)
                    
                all_vertices = np.vstack(vertices_list)
                all_faces = np.vstack(faces_list)
                
                # Create the combined mesh
                self.mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
                print(f"Combined mesh has {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces")
            else:
                # Just use the single mesh
                self.mesh = self.meshes[0]
                
            # Compute basic mesh statistics
            if not self.mesh.is_watertight:
                print("Warning: Mesh is not watertight. Point filtering may be less accurate.")
            
            print(f"Mesh volume: {self.mesh.volume:.2f} cubic units")
            print(f"Mesh bounding box: {self.mesh.bounds}")
            
            return True
        except Exception as e:
            print(f"Error loading meshes: {str(e)}")
            return False


def run_gui():
    """Run the application with a basic GUI."""
    root = tk.Tk()
    root.title("Structural Grid Generator")
    root.geometry("600x400")
    root.resizable(True, True)
    
    app = StructuralGridGenerator()
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Main content area
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Status bar
    status_frame = ttk.Frame(main_frame)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
    status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(fill=tk.X)
    
    # --- Workflow UI ---    def load_obj():
        filepath = filedialog.askopenfilename(
            title="Select Mesh File",
            filetypes=[
                ("Mesh Files", "*.obj *.stl *.ply *.off *.glb"),
                ("OBJ Files", "*.obj"),
                ("GLB Files", "*.glb"),
                ("STL Files", "*.stl"),
                ("PLY Files", "*.ply"),
                ("OFF Files", "*.off"),
                ("All Files", "*.*")
            ]
        )
        if filepath:
            if app.load_mesh(filepath):
                status_label.config(text=f"Loaded: {os.path.basename(filepath)}")
            else:
                status_label.config(text="Error loading mesh!")
    
    def set_spacing():
        spacing = simpledialog.askfloat(
            "Grid Spacing",
            "Enter grid spacing value:",
            initialvalue=app.voxel_pitch or (
                max(app.mesh.bounds[1] - app.mesh.bounds[0]) / 20 if app.mesh else 1.0
            ),
            minvalue=0.001
        )
        if spacing:
            app.voxel_pitch = spacing
            status_label.config(text=f"Grid spacing set to {spacing}")
            
    def set_floors():
        floors = simpledialog.askinteger(
            "Number of Floors",
            "Enter number of horizontal divisions (floors):",
            initialvalue=app.num_floors,
            minvalue=2,
            maxvalue=100
        )
        if floors:
            app.num_floors = floors
            status_label.config(text=f"Number of floors set to {floors}")
    
    def process_grid():
        if app.mesh is None:
            messagebox.showerror("Error", "Please load a mesh first!")
            return
            
        # Generate grid
        if not app.generate_grid():
            messagebox.showerror("Error", "Failed to generate grid!")
            return
            
        # Filter points
        if not app.filter_points():
            messagebox.showerror("Error", "Failed to filter grid points!")
            return
            
        # Generate structural elements
        if not app.generate_structures():
            messagebox.showerror("Error", "Failed to generate structural elements!")
            return
            
        status_label.config(text=f"Generated {len(app.columns)} columns and {len(app.beams)} beams")
    
    def visualize_structure():
        if app.mesh is None:
            messagebox.showerror("Error", "Please load a mesh first!")
            return
            
        if not app.columns and not app.beams:
            messagebox.showerror("Error", "No structural elements to visualize. Process the grid first!")
            return
            
        app.visualize()
    def export_structure():
        if not app.columns and not app.beams:
            messagebox.showerror("Error", "No structural elements to export. Process the grid first!")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Save Structural Elements As",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            if app.export_structures(filepath):
                status_label.config(text=f"Exported structural elements to {filepath}")
            else:
                status_label.config(text="Error exporting structural elements!")

    # UI Layout - using pack for simplicity
    ttk.Button(content_frame, text="Load Mesh File", command=load_obj).pack(anchor=tk.W, pady=5)
    ttk.Button(content_frame, text="Set Grid Spacing", command=set_spacing).pack(anchor=tk.W, pady=5)
    ttk.Button(content_frame, text="Set Number of Floors", command=set_floors).pack(anchor=tk.W, pady=5)
    ttk.Button(content_frame, text="Process Grid and Generate Structure", command=process_grid).pack(anchor=tk.W, pady=5)
    ttk.Button(content_frame, text="Visualize Structure", command=visualize_structure).pack(anchor=tk.W, pady=5)
    ttk.Button(content_frame, text="Export Structure Elements", command=export_structure).pack(anchor=tk.W, pady=5)

    root.mainloop()


def run_cmd():
    """Run the application from command line."""
    if len(sys.argv) < 2:
        print("Usage: python structural_grid.py <mesh_file> [spacing] [num_floors] [output_json]")
        return
        
    mesh_path = sys.argv[1]
    spacing = float(sys.argv[2]) if len(sys.argv) > 2 else None
    num_floors = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    output_json = sys.argv[4] if len(sys.argv) > 4 else None
    
    app = StructuralGridGenerator()
    
    # Load mesh
    if not app.load_mesh(mesh_path):
        return
        
    # Set parameters
    app.voxel_pitch = spacing
    app.num_floors = num_floors
    
    # Process the grid
    app.generate_grid()
    app.filter_points()
    app.generate_structures()
      # Export or visualize
    if output_json:
        app.export_structures(output_json)
    else:
        app.visualize()


if __name__ == "__main__":
    # Check if running in command line mode or GUI mode
    if len(sys.argv) > 1:
        run_cmd()
    else:
        run_gui()