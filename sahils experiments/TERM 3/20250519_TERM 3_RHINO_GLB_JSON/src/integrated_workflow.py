#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrated Workflow for 3D Structural Analysis
---------------------------------------------
This script combines the functionality of:
1. Rhino_to_mesh.py - Converting Rhino 3DM models to OBJ meshes
2. obj_combiner.py - Managing and combining OBJ mesh layers
3. structural_grid.py - Generating and visualizing structural grids

Author: GitHub Copilot
Date: April 27, 2025
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import numpy as np
import glob
from pathlib import Path

# Try to import dependencies
try:
    import rhino3dm
    HAS_RHINO3DM = True
except ImportError:
    HAS_RHINO3DM = False
    print("Warning: rhino3dm module not found. 3DM import functionality will be disabled.")

try:
    import trimesh
    import pyvista as pv
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Warning: trimesh or pyvista modules not found. Visualization functionality will be limited.")

# Import local modules
try:
    from obj_combiner import ObjCombiner
    HAS_OBJ_COMBINER = True
except ImportError:
    HAS_OBJ_COMBINER = False
    print("Warning: obj_combiner.py not found in the current directory.")

try:
    from structural_grid import StructuralGridGenerator
    HAS_STRUCTURAL_GRID = True
except ImportError:
    HAS_STRUCTURAL_GRID = False
    print("Warning: structural_grid.py not found in the current directory.")


class RhinoMeshConverter:
    """Handles the conversion of Rhino 3DM files to OBJ meshes."""
    
    def __init__(self):
        if not HAS_RHINO3DM:
            raise ImportError("rhino3dm module is required for 3DM conversion")
        
        self.model = None
        self.output_dir = None
        self.layer_objs = []  # Stores paths to generated OBJs
    
    def load_3dm(self, file_path):
        """Load a Rhino 3DM file."""
        try:
            self.model = rhino3dm.File3dm.Read(file_path)
            print(f"Loaded 3DM file with {len(self.model.Objects)} objects across {len(self.model.Layers)} layers")
            return True
        except Exception as e:
            print(f"Error loading 3DM file: {str(e)}")
            return False
    
    def mesh_brep(self, brep, mesh_type=rhino3dm.MeshType.Default):
        """Mesh a Brep object from Rhino."""
        meshes = []
        for face in brep.Faces:
            try:
                m = face.GetMesh(mesh_type)
            except Exception:
                continue
            if m:
                meshes.append(m)
        return meshes
    
    def export_layers_to_obj(self, output_dir=None):
        """Export each layer from the 3DM file to separate OBJ files."""
        if self.model is None:
            print("No 3DM model loaded!")
            return False
        
        # Set output directory
        self.output_dir = output_dir or './input model/meshed_layers_obj'
        os.makedirs(self.output_dir, exist_ok=True)
        self.layer_objs = []
        
        # Process each layer
        for lyr in self.model.Layers:
            meshes = []
            for obj in self.model.Objects:
                if obj.Attributes.LayerIndex != lyr.Index:
                    continue
                geom = obj.Geometry
                if isinstance(geom, rhino3dm.Mesh):
                    meshes.append(geom)
                elif isinstance(geom, rhino3dm.Brep):
                    meshes.extend(self.mesh_brep(geom))
            
            print(f"Layer [{lyr.Index}] '{lyr.Name}' -> {len(meshes)} mesh(es)")
            if not meshes:
                continue
            
            # Write OBJ file
            safe_name = lyr.Name.replace(' ', '_')
            obj_path = os.path.join(self.output_dir, f'meshed_layer_{lyr.Index}_{safe_name}.obj')
            with open(obj_path, 'w') as f:
                # Write a comment header
                f.write(f"# Layer {lyr.Index}: {lyr.Name}\n")
                vertex_offset = 0
                
                # Write vertices and faces
                for mesh in meshes:
                    # Vertices
                    for v in mesh.Vertices:
                        f.write(f"v {v.X} {v.Y} {v.Z}\n")
                    # Faces
                    for face in mesh.Faces:
                        if isinstance(face, tuple):
                            idxs = face
                        else:
                            if getattr(face, "IsQuad", False):
                                idxs = (face.A, face.B, face.C, face.D)
                            else:
                                idxs = (face.A, face.B, face.C)
                        # OBJ indexing is 1-based
                        idxs = [i + 1 + vertex_offset for i in idxs]
                        f.write(f"f {' '.join(map(str, idxs))}\n")
                    vertex_offset += len(mesh.Vertices)
            
            self.layer_objs.append(obj_path)
            print(f"  Saved OBJ: {obj_path}")
        
        return len(self.layer_objs) > 0


class IntegratedWorkflow:
    """Main class that integrates the Rhino converter, OBJ combiner, and structural grid functionality."""
    
    def __init__(self):
        self.rhino_converter = None if not HAS_RHINO3DM else RhinoMeshConverter()
        self.obj_combiner = None if not HAS_OBJ_COMBINER else ObjCombiner()
        self.struct_generator = None if not HAS_STRUCTURAL_GRID else StructuralGridGenerator()
        
        self.current_3dm_file = None
        self.current_obj_files = []
        self.current_combined_obj = None
        self.current_struct_grid = None
    
    def import_3dm_model(self, file_path=None, output_dir=None):
        """Import a 3DM model and convert to OBJ meshes."""
        if not HAS_RHINO3DM:
            print("rhino3dm module is required for this functionality")
            return False
        
        if file_path is None:
            # Ask user to select file if not provided
            file_path = filedialog.askopenfilename(
                title="Select Rhino 3DM File",
                filetypes=[("Rhino 3DM", "*.3dm"), ("All Files", "*.*")]
            )
            if not file_path:
                return False
        
        # Set current 3DM file
        self.current_3dm_file = file_path
        
        # Load and convert the model
        if self.rhino_converter.load_3dm(file_path):
            if self.rhino_converter.export_layers_to_obj(output_dir):
                # Store generated OBJs
                self.current_obj_files = self.rhino_converter.layer_objs
                return True
        
        return False
    
    def select_obj_files(self, directory=None):
        """Select OBJ files to work with."""
        if directory is None:
            # Ask user to select directory if not provided
            directory = filedialog.askdirectory(
                title="Select Directory with OBJ Files"
            )
            if not directory:
                return False
        
        # Scan directory for OBJ files
        obj_files = glob.glob(os.path.join(directory, '*.obj'))
        if obj_files:
            self.current_obj_files = obj_files
            return True
        
        return False
    
    def combine_objs(self, export_path=None):
        """Combine selected OBJ files."""
        if not HAS_OBJ_COMBINER:
            print("obj_combiner.py is required for this functionality")
            return False
        
        if not self.current_obj_files:
            print("No OBJ files selected!")
            return False
        
        # Load and combine meshes
        if self.obj_combiner.load_obj_files(self.current_obj_files):
            if self.obj_combiner.combine_meshes():
                if export_path:
                    if self.obj_combiner.export_combined_obj(export_path):
                        self.current_combined_obj = export_path
                        return True
                else:
                    # Just combine without exporting
                    return True
        
        return False
    
    def visualize_objs(self, combine=False):
        """Visualize selected OBJ files."""
        if not HAS_OBJ_COMBINER or not HAS_VISUALIZATION:
            print("obj_combiner.py and visualization libraries are required for this functionality")
            return False
        
        if not self.current_obj_files:
            print("No OBJ files selected!")
            return False
        
        # Load and visualize meshes
        if self.obj_combiner.load_obj_files(self.current_obj_files):
            self.obj_combiner.visualize(show_combined=combine)
            return True
        
        return False
    
    def generate_structural_grid(self, mesh_file=None, mesh_files=None):
        """Generate structural grid from a mesh file or multiple mesh files."""
        if not HAS_STRUCTURAL_GRID:
            print("structural_grid.py is required for this functionality")
            return False
        
        # Handle multiple mesh files
        if mesh_files and len(mesh_files) > 0:
            if self.struct_generator.load_multiple_meshes(mesh_files):
                # Process the grid with combined meshes
                if (self.struct_generator.generate_grid() and 
                    self.struct_generator.filter_points() and 
                    self.struct_generator.generate_structures()):
                    return True
            return False
        
        # Handle single mesh file if no multiple files provided
        if mesh_file is None and not mesh_files:
            # If no specific mesh file is provided, use the combined OBJ if available
            if self.current_combined_obj:
                mesh_file = self.current_combined_obj
            # Otherwise use the first OBJ in the list
            elif self.current_obj_files:
                mesh_file = self.current_obj_files[0]
            else:
                print("No mesh file available!")
                return False
        
        # Load mesh
        if self.struct_generator.load_mesh(mesh_file):
            # Set parameters
            spacing = self.struct_generator.voxel_pitch  # Use default or let the generator calculate it
            num_floors = self.struct_generator.num_floors  # Use default
            
            # Process the grid
            if (self.struct_generator.generate_grid() and 
                self.struct_generator.filter_points() and 
                self.struct_generator.generate_structures()):
                return True
        
        return False
    
    def visualize_structural_grid(self, allow_grid_movement=True):
        """Visualize the generated structural grid with optional interactive grid movement controls."""
        if not HAS_STRUCTURAL_GRID or not HAS_VISUALIZATION:
            print("structural_grid.py and visualization libraries are required for this functionality")
            return False
        
        if self.struct_generator is None:
            print("No structural grid generator available!")
            return False
        
        # Visualize the grid with interactive controls if requested
        return self.struct_generator.visualize(interactive=allow_grid_movement)
    def export_structural_grid(self, filepath=None):
        """Export the structural grid elements to a JSON file."""
        if not HAS_STRUCTURAL_GRID:
            print("structural_grid.py is required for this functionality")
            return False
        
        if self.struct_generator is None:
            print("No structural grid generator available!")
            return False
        
        if filepath is None:            # Ask user to select file if not provided
            filepath = filedialog.asksaveasfilename(
                title="Save Structural Elements As",
                initialdir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "json"),
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
            )
            if not filepath:
                return False
        
        # Export the structural elements
        return self.struct_generator.export_structures(filepath)


def run_gui():
    """Run the integrated workflow application with a GUI."""
    root = tk.Tk()
    root.title("Integrated Structural Workflow")
    root.geometry("800x650")
    root.resizable(True, True)
    
    app = IntegratedWorkflow()
    
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
    
    # Create a notebook for workflow steps
    notebook = ttk.Notebook(content_frame)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # --- Step 1: Rhino Import Tab ---
    rhino_frame = ttk.Frame(notebook)
    notebook.add(rhino_frame, text="1. Import Rhino")
    
    rhino_content = ttk.Frame(rhino_frame, padding="10 10 10 10")
    rhino_content.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(rhino_content, text="Import Rhino 3DM models and convert to OBJ meshes.").pack(anchor=tk.W, pady=5)
    
    def import_3dm():
        file_path = filedialog.askopenfilename(
            title="Select Rhino 3DM File",
            filetypes=[("Rhino 3DM", "*.3dm"), ("All Files", "*.*")]
        )
        if file_path:
            output_dir = filedialog.askdirectory(
                title="Select Output Directory for OBJ Files",
                initialdir=os.path.dirname(file_path)
            )
            if output_dir:
                if app.import_3dm_model(file_path, output_dir):
                    status_label.config(text=f"Imported {file_path} and generated {len(app.current_obj_files)} OBJ files")
                    messagebox.showinfo("Import Complete", f"Successfully generated {len(app.current_obj_files)} OBJ files in {output_dir}")
                    # Switch to the next tab
                    notebook.select(1)
                else:
                    status_label.config(text="Error importing 3DM model")
    
    ttk.Button(rhino_content, text="Import 3DM File", command=import_3dm).pack(anchor=tk.W, pady=5)
    
    # Disable the tab if rhino3dm is not available
    if not HAS_RHINO3DM:
        rhino_disabled_label = ttk.Label(rhino_content, text="rhino3dm module not available.\nPlease install it with: pip install rhino3dm", foreground="red")
        rhino_disabled_label.pack(anchor=tk.CENTER, pady=20)
    
    # --- Step 2: OBJ Selection Tab ---
    obj_frame = ttk.Frame(notebook)
    notebook.add(obj_frame, text="2. OBJ Selection")
    
    obj_content = ttk.Frame(obj_frame, padding="10 10 10 10")
    obj_content.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(obj_content, text="Select OBJ files to work with or use the ones generated from Rhino.").pack(anchor=tk.W, pady=5)
    
    # List to display loaded files
    file_frame = ttk.LabelFrame(obj_content, text="Selected OBJ Files")
    file_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    file_listbox = tk.Listbox(file_frame)
    file_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def update_obj_list():
        file_listbox.delete(0, tk.END)
        for file_path in app.current_obj_files:
            file_listbox.insert(tk.END, os.path.basename(file_path))
    
    def select_obj_directory():
        directory = filedialog.askdirectory(
            title="Select Directory with OBJ Files"
        )
        if directory:
            if app.select_obj_files(directory):
                update_obj_list()
                status_label.config(text=f"Selected {len(app.current_obj_files)} OBJ files from {directory}")
            else:
                status_label.config(text="No OBJ files found in the selected directory")
    
    def select_obj_files():
        files = filedialog.askopenfilenames(
            title="Select OBJ Files",
            filetypes=[("OBJ Files", "*.obj"), ("All Files", "*.*")]
        )
        if files:
            app.current_obj_files = list(files)
            update_obj_list()
            status_label.config(text=f"Selected {len(app.current_obj_files)} OBJ files")
    
    def visualize_separate():
        if app.visualize_objs(combine=False):
            status_label.config(text="Visualizing OBJ files separately")
        else:
            status_label.config(text="Error visualizing OBJ files")
    
    button_frame = ttk.Frame(obj_content)
    button_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Button(button_frame, text="Select Directory", command=select_obj_directory).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Select OBJ Files", command=select_obj_files).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Visualize Separate", command=visualize_separate).pack(side=tk.LEFT, padx=5)
    
    # --- Step 3: OBJ Combining Tab ---
    combine_frame = ttk.Frame(notebook)
    notebook.add(combine_frame, text="3. Combine OBJs")
    
    combine_content = ttk.Frame(combine_frame, padding="10 10 10 10")
    combine_content.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(combine_content, text="Combine multiple OBJ files into a single mesh for structural analysis.").pack(anchor=tk.W, pady=5)
    
    def combine_and_visualize():
        if app.visualize_objs(combine=True):
            status_label.config(text="Visualizing combined OBJ")
        else:
            status_label.config(text="Error visualizing combined OBJ")
    
    def combine_and_export():
        filepath = filedialog.asksaveasfilename(
            title="Save Combined OBJ As",
            defaultextension=".obj",
            filetypes=[("OBJ Files", "*.obj"), ("All Files", "*.*")]
        )
        if filepath:
            if app.combine_objs(filepath):
                status_label.config(text=f"Combined and exported to {filepath}")
                # Offer to proceed to the next step
                if messagebox.askyesno("Continue", "Proceed to structural grid generation?"):
                    notebook.select(3)
            else:
                status_label.config(text="Error combining OBJ files")
    
    ttk.Button(combine_content, text="Combine & Visualize", command=combine_and_visualize).pack(anchor=tk.W, pady=5)
    ttk.Button(combine_content, text="Combine & Export", command=combine_and_export).pack(anchor=tk.W, pady=5)
    
    # --- Step 4: Structural Grid Tab ---
    struct_frame = ttk.Frame(notebook)
    notebook.add(struct_frame, text="4. Structural Grid")
    
    struct_content = ttk.Frame(struct_frame, padding="10 10 10 10")
    struct_content.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(struct_content, text="Generate and visualize a structural grid from the combined mesh.").pack(anchor=tk.W, pady=5)
    
    # Grid settings frame
    settings_frame = ttk.LabelFrame(struct_content, text="Grid Settings")
    settings_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Create a reference to the struct_generator for settings
    struct_generator = app.struct_generator
    
    # Grid spacing setting
    ttk.Label(settings_frame, text="Grid Spacing:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    spacing_var = tk.StringVar(value="Auto")
    spacing_entry = ttk.Entry(settings_frame, textvariable=spacing_var, width=10)
    spacing_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
    
    # Number of floors setting
    ttk.Label(settings_frame, text="Number of Floors:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    floors_var = tk.StringVar(value="3")
    floors_entry = ttk.Entry(settings_frame, textvariable=floors_var, width=10)
    floors_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
    
    def set_grid_settings():
        if struct_generator:
            try:
                # Set spacing if not "Auto"
                spacing_value = spacing_var.get()
                if spacing_value != "Auto":
                    struct_generator.voxel_pitch = float(spacing_value)
                else:
                    struct_generator.voxel_pitch = None  # Let the generator calculate it
                
                # Set number of floors
                struct_generator.num_floors = int(floors_var.get())
                
                status_label.config(text=f"Grid settings updated: Spacing={spacing_value}, Floors={floors_var.get()}")
                return True
            except ValueError as e:
                messagebox.showerror("Invalid Setting", f"Invalid grid setting: {str(e)}")
                return False
        return False
    
    ttk.Button(settings_frame, text="Apply Settings", command=set_grid_settings).grid(row=0, column=2, rowspan=2, padx=5, pady=5)
    
    def select_and_generate():        # Let user select a mesh file
        mesh_file = filedialog.askopenfilename(
            title="Select Mesh File",
            filetypes=[("OBJ Files", "*.obj"), ("GLB Files", "*.glb"), ("All Mesh Files", "*.obj;*.stl;*.ply;*.glb"), ("All Files", "*.*")]
        )
        if mesh_file:
            # Apply grid settings
            set_grid_settings()
            
            # Generate grid
            if app.generate_structural_grid(mesh_file):
                status_label.config(text=f"Generated structural grid for {os.path.basename(mesh_file)}")
                if messagebox.askyesno("Visualization", "Visualize the generated structural grid?"):
                    app.visualize_structural_grid()
            else:
                status_label.config(text="Error generating structural grid")
    
    def select_multiple_and_generate():        # Let user select multiple mesh files
        mesh_files = filedialog.askopenfilenames(
            title="Select Multiple Mesh Files",
            filetypes=[("OBJ Files", "*.obj"), ("GLB Files", "*.glb"), ("All Mesh Files", "*.obj;*.stl;*.ply;*.glb"), ("All Files", "*.*")]
        )
        if mesh_files and len(mesh_files) > 0:
            # Sort the files to prioritize column meshes
            sorted_mesh_files = sorted(mesh_files, 
                                      key=lambda path: 0 if "column" in os.path.basename(path).lower() else 1)
            
            # Apply grid settings
            set_grid_settings()
            
            # Generate grid with multiple meshes
            if app.generate_structural_grid(mesh_files=list(sorted_mesh_files)):
                # Find if there were any column meshes to highlight in status
                column_files = [f for f in sorted_mesh_files if "column" in os.path.basename(f).lower()]
                if column_files:
                    status_label.config(text=f"Generated grid using column mesh as reference")
                else:
                    status_label.config(text=f"Generated structural grid from {len(mesh_files)} mesh files")
                
                if messagebox.askyesno("Visualization", "Visualize the generated structural grid?"):
                    app.visualize_structural_grid()
            else:
                status_label.config(text="Error generating structural grid")
    
    def generate_from_combined():
        if not app.current_combined_obj and not app.current_obj_files:
            messagebox.showerror("No Mesh File", "No combined or individual mesh files available.")
            return
        
        # Apply grid settings
        set_grid_settings()
        
        # Generate grid using default file selection logic
        if app.generate_structural_grid():
            status_label.config(text="Generated structural grid")
            if messagebox.askyesno("Visualization", "Visualize the generated structural grid?"):
                app.visualize_structural_grid()
        else:
            status_label.config(text="Error generating structural grid")
    
    button_frame = ttk.Frame(struct_content)
    button_frame.pack(fill=tk.X, padx=5, pady=10)
    
    ttk.Button(button_frame, text="Select Mesh & Generate", command=select_and_generate).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Select Multiple Meshes", command=select_multiple_and_generate).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Generate from Combined", command=generate_from_combined).pack(side=tk.LEFT, padx=5)
    
    # Add separate buttons for visualization with and without grid movement
    def visualize_grid_with_movement():
        if app.visualize_structural_grid(allow_grid_movement=True):
            status_label.config(text="Visualizing structural grid with interactive grid movement controls")
        else:
            status_label.config(text="Error visualizing structural grid")
    
    def visualize_grid_without_movement():
        if app.visualize_structural_grid(allow_grid_movement=False):
            status_label.config(text="Visualizing fixed structural grid")
        else:
            status_label.config(text="Error visualizing structural grid")
    
    # Define the export_grid function
    def export_grid():
        filepath = filedialog.asksaveasfilename(
            title="Export Structural Elements",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            if app.export_structural_grid(filepath):
                status_label.config(text=f"Exported structural elements to {filepath}")
            else:
                status_label.config(text="Error exporting structural elements")
            
    visualization_frame = ttk.Frame(struct_content)
    visualization_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(visualization_frame, text="Visualization Options:").pack(anchor=tk.W, padx=5)
    
    visualization_buttons = ttk.Frame(visualization_frame)
    visualization_buttons.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Button(visualization_buttons, text="Interactive Grid Movement", 
              command=visualize_grid_with_movement).pack(side=tk.LEFT, padx=5)
    ttk.Button(visualization_buttons, text="Fixed Grid Visualization", 
              command=visualize_grid_without_movement).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(button_frame, text="Export Grid (JSON)", command=export_grid).pack(side=tk.LEFT, padx=5)
    
    # Help text with updated information about grid movement
    help_frame = ttk.Frame(struct_content)
    help_frame.pack(fill=tk.X, padx=5, pady=10)
    
    help_text = tk.Text(help_frame, height=8, width=50, wrap=tk.WORD)
    help_text.pack(fill=tk.X, padx=5, pady=5)
    help_text.insert(tk.END, "This tab allows you to generate a structural grid from a mesh file.\n\n"
                    "1. Adjust grid spacing (or leave as 'Auto') and number of floors\n"
                    "2. Select a mesh file or use the previously combined one\n"
                    "3. Generate the structural grid\n"
                    "4. Visualize and export the grid as needed\n\n"
                    "NEW: Use 'Interactive Grid Movement' to adjust grid positions during visualization. "
                    "Sliders will appear at the bottom of the 3D view to control X/Y offsets and spacing.")
    help_text.config(state=tk.DISABLED)
    
    # Initialize by updating the OBJ list if files are already loaded
    if app.current_obj_files:
        update_obj_list()

    root.mainloop()


def run_cmd():
    """Run the integrated workflow from command line."""
    if len(sys.argv) < 2:
        print("""
Usage: python integrated_workflow.py [options]

Options:
  --import-3dm <file.3dm> [output_dir]   Import a Rhino 3DM file and convert to OBJs
  --select-objs <dir or file1,file2,...>  Select OBJ files to work with
  --combine-objs [output.obj]             Combine selected OBJs
  --visualize-objs [--combined]           Visualize selected OBJs (separate or combined)
  --generate-grid [mesh.obj] [spacing] [floors]  Generate structural grid
  --visualize-grid                        Visualize structural grid
  --export-grid <output.json>              Export structural grid to JSON
        """)
        return
    
    app = IntegratedWorkflow()
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == "--import-3dm":
            if i + 1 < len(sys.argv):
                file_path = sys.argv[i + 1]
                output_dir = sys.argv[i + 2] if i + 2 < len(sys.argv) and not sys.argv[i + 2].startswith("--") else None
                app.import_3dm_model(file_path, output_dir)
                i += 2 if output_dir is None else 3
            else:
                print("Error: Missing file path after --import-3dm")
                return
        
        elif arg == "--select-objs":
            if i + 1 < len(sys.argv):
                source = sys.argv[i + 1]
                if os.path.isdir(source):
                    app.select_obj_files(source)
                else:
                    # Comma-separated list of files
                    app.current_obj_files = source.split(",")
                i += 2
            else:
                print("Error: Missing source after --select-objs")
                return
        
        elif arg == "--combine-objs":
            output_file = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else None
            app.combine_objs(output_file)
            i += 1 if output_file is None else 2
        
        elif arg == "--visualize-objs":
            combined = sys.argv[i + 1] == "--combined" if i + 1 < len(sys.argv) else False
            app.visualize_objs(combined)
            i += 1 if not combined else 2
        
        elif arg == "--generate-grid":
            mesh_file = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else None
            spacing = float(sys.argv[i + 2]) if i + 2 < len(sys.argv) and not sys.argv[i + 2].startswith("--") else None
            floors = int(sys.argv[i + 3]) if i + 3 < len(sys.argv) and not sys.argv[i + 3].startswith("--") else None
            
            if spacing is not None and app.struct_generator is not None:
                app.struct_generator.voxel_pitch = spacing
            if floors is not None and app.struct_generator is not None:
                app.struct_generator.num_floors = floors
                
            app.generate_structural_grid(mesh_file)
            i += 1
            if mesh_file is not None: i += 1
            if spacing is not None: i += 1
            if floors is not None: i += 1
        
        elif arg == "--visualize-grid":
            app.visualize_structural_grid()
            i += 1
        
        elif arg == "--export-grid":
            if i + 1 < len(sys.argv):
                app.export_structural_grid(sys.argv[i + 1])
                i += 2
            else:
                print("Error: Missing output file after --export-grid")
                return
        
        else:
            print(f"Unknown option: {arg}")
            i += 1


if __name__ == "__main__":
    # Check if running in command line mode or GUI mode
    if len(sys.argv) > 1:
        run_cmd()
    else:
        run_gui()