#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OBJ Combiner and Visualizer
---------------------------
This script loads multiple OBJ files, combines them while preserving their
original positions, and provides visualization of the combined structure.
"""

import os
import sys
import numpy as np
import trimesh
import pyvista as pv
import glob
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class ObjCombiner:
    def __init__(self):
        self.meshes = []  # List of loaded individual meshes
        self.combined_mesh = None  # Combined mesh (if multiple files loaded)
        self.mesh_colors = []  # Colors for different meshes
        self.plotter = None
        self.selected_index = None  # Index of the currently selected mesh
        self.pv_meshes = []  # List of PyVista mesh objects for selection
        self.mesh_names = []  # Names of the loaded mesh files

    def load_obj_files(self, file_paths):
        """Load multiple OBJ files and store them as separate meshes."""
        self.meshes = []
        self.mesh_colors = []
        self.mesh_names = []
        self.selected_index = None
        
        # Define a color palette for meshes
        color_palette = [
            [1.0, 0.6, 0.6],  # Light red
            [0.6, 1.0, 0.6],  # Light green
            [0.6, 0.6, 1.0],  # Light blue
            [1.0, 1.0, 0.6],  # Light yellow
            [1.0, 0.6, 1.0],  # Light magenta
            [0.6, 1.0, 1.0],  # Light cyan
            [0.9, 0.7, 0.5],  # Light brown
            [0.7, 0.7, 0.7],  # Light gray
        ]
        
        # Load each mesh and assign a color
        for i, file_path in enumerate(file_paths):
            try:
                # Load mesh using trimesh
                mesh = trimesh.load(file_path)
                print(f"Loaded {os.path.basename(file_path)} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
                
                # Add to our collection
                self.meshes.append(mesh)
                self.mesh_colors.append(color_palette[i % len(color_palette)])
                self.mesh_names.append(os.path.basename(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        print(f"Loaded {len(self.meshes)} mesh files")
        return len(self.meshes) > 0

    def combine_meshes(self):
        """Combine all loaded meshes into a single mesh while preserving positions."""
        if not self.meshes:
            print("No meshes to combine!")
            return False
            
        if len(self.meshes) == 1:
            # If only one mesh, no need to combine
            self.combined_mesh = self.meshes[0]
            return True
            
        # Combine all meshes
        # We use trimesh's boolean union, but we could also just concatenate vertices and faces
        try:
            # For simple concatenation without merging vertices
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
            self.combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
            print(f"Combined mesh has {len(self.combined_mesh.vertices)} vertices and {len(self.combined_mesh.faces)} faces")
            return True
        except Exception as e:
            print(f"Error combining meshes: {str(e)}")
            return False

    def select_mesh(self, mesh_index):
        """Select a mesh by index and highlight it."""
        if mesh_index is not None and (mesh_index < 0 or mesh_index >= len(self.meshes)):
            print(f"Invalid mesh index: {mesh_index}")
            return False
            
        self.selected_index = mesh_index
        return True

    def get_selected_mesh_name(self):
        """Get the name of the currently selected mesh."""
        if self.selected_index is not None and 0 <= self.selected_index < len(self.mesh_names):
            return self.mesh_names[self.selected_index]
        return None

    def handle_pick_event(self, mesh):
        """Handle picking event from PyVista."""
        if mesh is None:
            return
            
        try:
            # Find which mesh was picked
            for i, pv_mesh in enumerate(self.pv_meshes):
                if pv_mesh == mesh:
                    print(f"Selected mesh: {self.mesh_names[i]}")
                    self.select_mesh(i)
                    
                    # Update visualization with new selection
                    self.update_mesh_appearances()
                    break
        except Exception as e:
            print(f"Error in pick event handler: {str(e)}")

    def update_mesh_appearances(self):
        """Update the appearance of all meshes based on selection state."""
        if not self.plotter or not self.pv_meshes:
            return
            
        for i, pv_mesh in enumerate(self.pv_meshes):
            if i == self.selected_index:
                # Highlight the selected mesh
                self.plotter.add_mesh(pv_mesh, color='yellow', opacity=1.0, 
                                      show_edges=True, reset_camera=False, name=f"mesh_{i}")
            else:
                # Regular appearance for non-selected meshes
                self.plotter.add_mesh(pv_mesh, color=self.mesh_colors[i], opacity=0.7, 
                                      show_edges=True, reset_camera=False, name=f"mesh_{i}")

    def visualize(self, show_combined=False):
        """Visualize the loaded meshes or the combined mesh."""
        if not self.meshes:
            print("No meshes to visualize!")
            return False
            
        # Create plotter with enable_picking set to True
        self.plotter = pv.Plotter()
        self.pv_meshes = []
        
        if show_combined and self.combined_mesh is not None:
            # Visualize the combined mesh
            vertices = self.combined_mesh.vertices
            faces = np.hstack([[3, f[0], f[1], f[2]] for f in self.combined_mesh.faces])
            pv_mesh = pv.PolyData(vertices, faces)
            self.plotter.add_mesh(pv_mesh, color='lightblue', opacity=0.7, show_edges=True)
        else:
            # Visualize individual meshes with different colors
            for i, mesh in enumerate(self.meshes):
                vertices = mesh.vertices
                faces = np.hstack([[3, f[0], f[1], f[2]] for f in mesh.faces])
                pv_mesh = pv.PolyData(vertices, faces)
                self.pv_meshes.append(pv_mesh)
                
                # Set initial appearance
                opacity = 1.0 if i == self.selected_index else 0.7
                color = 'yellow' if i == self.selected_index else self.mesh_colors[i]
                self.plotter.add_mesh(pv_mesh, color=color, opacity=opacity, 
                                      show_edges=True, name=f"mesh_{i}")

            # Set up picking callback
            if not show_combined:
                self.plotter.enable_mesh_picking(callback=self.handle_pick_event, 
                                                left_clicking=True, show=False)
        
        # Add bounding box and axes
        self.plotter.add_bounding_box()
        self.plotter.add_axes()
        
        # Show the plot
        self.plotter.show()
        return True

    def export_combined_obj(self, filepath):
        """Export the combined mesh as an OBJ file."""
        if self.combined_mesh is None:
            print("No combined mesh to export!")
            return False
            
        try:
            self.combined_mesh.export(filepath)
            print(f"Exported combined mesh to {filepath}")
            return True
        except Exception as e:
            print(f"Error exporting combined mesh: {str(e)}")
            return False


def scan_obj_directory(directory_path):
    """Scan a directory for OBJ files."""
    obj_files = glob.glob(os.path.join(directory_path, '*.obj'))
    return obj_files


def run_gui():
    """Run the application with a GUI."""
    root = tk.Tk()
    root.title("OBJ Combiner and Visualizer")
    root.geometry("600x450")  # Slightly taller to accommodate selection info
    root.resizable(True, True)
    
    app = ObjCombiner()
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Main content area
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # List to display loaded files
    file_frame = ttk.LabelFrame(content_frame, text="Loaded OBJ Files")
    file_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    file_listbox = tk.Listbox(file_frame)
    file_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    file_paths = []
    
    # Selection info frame
    selection_frame = ttk.LabelFrame(content_frame, text="Selection")
    selection_frame.pack(fill=tk.X, padx=5, pady=5)
    
    selection_label = ttk.Label(selection_frame, text="No mesh selected")
    selection_label.pack(side=tk.LEFT, padx=5, pady=5)
    
    def update_selection_info():
        selected_name = app.get_selected_mesh_name()
        if selected_name:
            selection_label.config(text=f"Selected: {selected_name}")
        else:
            selection_label.config(text="No mesh selected")
    
    def reset_selection():
        app.select_mesh(None)
        update_selection_info()
        # Re-visualize to reset the view
        if app.meshes:
            app.visualize(show_combined=False)
    
    ttk.Button(selection_frame, text="Reset Selection", command=reset_selection).pack(side=tk.RIGHT, padx=5, pady=5)
    
    # Status bar
    status_frame = ttk.Frame(main_frame)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
    status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(fill=tk.X)
    
    # --- Workflow UI ---
    def select_directory():
        # Select directory containing OBJ files
        directory = filedialog.askdirectory(
            title="Select Directory with OBJ Files"
        )
        if directory:
            obj_files = scan_obj_directory(directory)
            if obj_files:
                file_paths.clear()
                file_listbox.delete(0, tk.END)
                for file_path in obj_files:
                    file_paths.append(file_path)
                    file_listbox.insert(tk.END, os.path.basename(file_path))
                status_label.config(text=f"Loaded {len(obj_files)} OBJ files from directory")
            else:
                messagebox.showinfo("No OBJ Files", "No OBJ files found in the selected directory.")
    
    def select_files():
        # Select multiple OBJ files
        files = filedialog.askopenfilenames(
            title="Select OBJ Files",
            filetypes=[("OBJ Files", "*.obj"), ("All Files", "*.*")]
        )
        if files:
            file_paths.clear()
            file_listbox.delete(0, tk.END)
            for file_path in files:
                file_paths.append(file_path)
                file_listbox.insert(tk.END, os.path.basename(file_path))
            status_label.config(text=f"Selected {len(files)} OBJ files")
    
    def load_and_visualize():
        if not file_paths:
            messagebox.showerror("Error", "No OBJ files selected!")
            return
            
        if app.load_obj_files(file_paths):
            status_label.config(text=f"Loaded {len(app.meshes)} OBJ files")
            update_selection_info()  # Reset selection info
            app.visualize(show_combined=False)
        else:
            messagebox.showerror("Error", "Failed to load OBJ files!")
    
    def combine_and_visualize():
        if not file_paths:
            messagebox.showerror("Error", "No OBJ files selected!")
            return
            
        if not app.meshes:
            if not app.load_obj_files(file_paths):
                messagebox.showerror("Error", "Failed to load OBJ files!")
                return
                
        if app.combine_meshes():
            status_label.config(text="Combined meshes successfully")
            update_selection_info()  # Reset selection info
            app.visualize(show_combined=True)
        else:
            messagebox.showerror("Error", "Failed to combine meshes!")
    
    def export_combined():
        if not app.meshes:
            messagebox.showerror("Error", "No meshes loaded!")
            return
            
        if app.combined_mesh is None:
            if not app.combine_meshes():
                messagebox.showerror("Error", "Failed to combine meshes!")
                return
                
        filepath = filedialog.asksaveasfilename(
            title="Save Combined OBJ As",
            defaultextension=".obj",
            filetypes=[("OBJ Files", "*.obj"), ("All Files", "*.*")]
        )
        if filepath:
            if app.export_combined_obj(filepath):
                status_label.config(text=f"Exported combined mesh to {os.path.basename(filepath)}")
            else:
                messagebox.showerror("Error", "Failed to export combined mesh!")
    
    # Button frame
    button_frame = ttk.Frame(content_frame)
    button_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # UI Layout - using pack for simplicity
    ttk.Button(button_frame, text="Select Directory", command=select_directory).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Select OBJ Files", command=select_files).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Visualize Separate", command=load_and_visualize).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Combine & Visualize", command=combine_and_visualize).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Export Combined", command=export_combined).pack(side=tk.LEFT, padx=5)

    root.mainloop()


def run_cmd():
    """Run the application from command line."""
    if len(sys.argv) < 2:
        print("Usage: python obj_combiner.py [-d directory | file1.obj file2.obj ...] [-o output.obj]")
        return
        
    # Parse arguments
    output_file = None
    obj_files = []
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-d":
            if i + 1 < len(sys.argv):
                directory = sys.argv[i + 1]
                obj_files.extend(scan_obj_directory(directory))
                i += 2
            else:
                print("Error: Missing directory after -d")
                return
        elif arg == "-o":
            if i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
            else:
                print("Error: Missing output file after -o")
                return
        else:
            # Assume it's an OBJ file
            if arg.endswith(".obj"):
                obj_files.append(arg)
            i += 1
    
    if not obj_files:
        print("Error: No OBJ files specified!")
        return
    
    # Process files
    app = ObjCombiner()
    
    if app.load_obj_files(obj_files):
        app.combine_meshes()
        
        # Export combined mesh if requested
        if output_file:
            app.export_combined_obj(output_file)
        else:
            # Visualize if no output file specified
            app.visualize(show_combined=True)
    else:
        print("Failed to load OBJ files!")


if __name__ == "__main__":
    # Check if running in command line mode or GUI mode
    if len(sys.argv) > 1:
        run_cmd()
    else:
        run_gui()