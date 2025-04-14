import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QPushButton, QComboBox, QLabel, QSlider, 
                           QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QColorDialog)
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph.opengl as gl
import trimesh
from scipy.spatial import Voronoi, Delaunay

class StructuralSystemGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Structural System Generator")
        self.resize(1200, 800)
        
        # Model data
        self.mesh = None
        self.selected_faces = []
        self.structural_elements = []
        
        # Initialize UI
        self.initUI()
        
    def initUI(self):
        # Main layout
        main_layout = QHBoxLayout()
        
        # 3D Viewport
        self.view3d = gl.GLViewWidget()
        main_layout.addWidget(self.view3d, 3)
        
        # Create grid for reference
        grid = gl.GLGridItem()
        grid.setSize(10, 10, 1)
        grid.setSpacing(1, 1, 1)
        self.view3d.addItem(grid)
        
        # Controls panel
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        
        # Load model button
        load_btn = QPushButton("Load 3D Model")
        load_btn.clicked.connect(self.load_model)
        controls_layout.addWidget(load_btn)
        
        # Selection mode
        selection_group = QGroupBox("Selection")
        selection_layout = QVBoxLayout()
        self.selection_mode = QComboBox()
        self.selection_mode.addItems(["Face", "Edge", "Vertex"])
        selection_layout.addWidget(self.selection_mode)
        
        select_btn = QPushButton("Enter Selection Mode")
        select_btn.clicked.connect(self.enter_selection_mode)
        selection_layout.addWidget(select_btn)
        
        clear_selection_btn = QPushButton("Clear Selection")
        clear_selection_btn.clicked.connect(self.clear_selection)
        selection_layout.addWidget(clear_selection_btn)
        
        selection_group.setLayout(selection_layout)
        controls_layout.addWidget(selection_group)
        
        # Structural systems
        systems_group = QGroupBox("Structural Systems")
        systems_layout = QVBoxLayout()
        
        self.system_type = QComboBox()
        self.system_type.addItems([
            "Grid", "Diagrid", "Space Frame", 
            "Voronoi", "Triangulated", "Waffle"
        ])
        systems_layout.addWidget(self.system_type)
        
        # System parameters
        params_form = QFormLayout()
        
        # Density parameter
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(1, 100)
        self.density_slider.setValue(20)
        params_form.addRow("Density:", self.density_slider)
        
        # Depth parameter
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.01, 5.0)
        self.depth_spin.setValue(0.5)
        self.depth_spin.setSingleStep(0.1)
        params_form.addRow("Depth:", self.depth_spin)
        
        # Color selection
        self.color_btn = QPushButton("Select Color")
        self.color_btn.clicked.connect(self.select_color)
        self.element_color = [255, 0, 0]  # Default red
        params_form.addRow("Color:", self.color_btn)
        
        systems_layout.addLayout(params_form)
        
        # Apply button
        apply_btn = QPushButton("Apply System")
        apply_btn.clicked.connect(self.apply_system)
        systems_layout.addWidget(apply_btn)
        
        systems_group.setLayout(systems_layout)
        controls_layout.addWidget(systems_group)
        
        # Export button
        export_btn = QPushButton("Export Structure")
        export_btn.clicked.connect(self.export_structure)
        controls_layout.addWidget(export_btn)
        
        main_layout.addWidget(controls_panel, 1)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def load_model(self):
        """Load a 3D model file"""
        # In a real app, use QFileDialog to get the path
        # For demonstration, we'll create a simple mesh
        
        # Here you would normally load an actual model:
        # self.mesh = trimesh.load('path/to/your/model.obj')
        
        # For demo, create a simple cube mesh
        self.mesh = trimesh.creation.box(extents=[2, 2, 2])
        
        # Clear previous mesh from view
        self.view3d.clear()
        
        # Add grid back
        grid = gl.GLGridItem()
        grid.setSize(10, 10, 1)
        grid.setSpacing(1, 1, 1)
        self.view3d.addItem(grid)
        
        # Display the mesh
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        mesh_item = gl.GLMeshItem(
            vertexes=vertices, 
            faces=faces, 
            faceColors=np.array([[0.7, 0.7, 1.0, 0.5] for _ in range(len(faces))]),
            smooth=False
        )
        self.view3d.addItem(mesh_item)
        self.mesh_item = mesh_item
    
    def enter_selection_mode(self):
        """Enter the mode for selecting faces/edges/vertices"""
        print(f"Entered selection mode: {self.selection_mode.currentText()}")
        # In a real implementation, you would set up event handlers for mouse picking
        
        # For this demo, let's simulate selecting all faces on the top of the cube
        if self.mesh is not None:
            # For demo, select the top face of the cube
            top_face_indices = []
            for i, face in enumerate(self.mesh.faces):
                # Get face center
                face_center = self.mesh.triangles_center[i]
                # If y coordinate is close to 1, it's on top (for our demo cube)
                if face_center[1] > 0.9:
                    top_face_indices.append(i)
            
            # Highlight selected faces
            face_colors = np.array([[0.7, 0.7, 1.0, 0.5] for _ in range(len(self.mesh.faces))])
            for idx in top_face_indices:
                face_colors[idx] = [1.0, 0.0, 0.0, 0.8]  # Highlight in red
            
            self.mesh_item.setMeshData(
                vertexes=self.mesh.vertices, 
                faces=self.mesh.faces, 
                faceColors=face_colors
            )
            
            self.selected_faces = top_face_indices
    
    def clear_selection(self):
        """Clear the current selection"""
        if self.mesh is not None:
            # Reset all face colors
            face_colors = np.array([[0.7, 0.7, 1.0, 0.5] for _ in range(len(self.mesh.faces))])
            self.mesh_item.setMeshData(
                vertexes=self.mesh.vertices, 
                faces=self.mesh.faces, 
                faceColors=face_colors
            )
            self.selected_faces = []
    
    def select_color(self):
        """Open color dialog to pick element color"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.element_color = [color.red(), color.green(), color.blue()]
            self.color_btn.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})")
    
    def apply_system(self):
        """Apply the selected structural system to the selection"""
        if not self.mesh or not self.selected_faces:
            print("Please load a model and select faces first")
            return
        
        system_type = self.system_type.currentText()
        density = self.density_slider.value() / 10.0  # Convert to a reasonable range
        depth = self.depth_spin.value()
        
        # Get the submesh of selected faces
        selected_vertices = set()
        for face_idx in self.selected_faces:
            face = self.mesh.faces[face_idx]
            selected_vertices.update(face)
        
        # Get unique vertices
        vertices = np.array([self.mesh.vertices[i] for i in selected_vertices])
        
        # Clear previous structural elements
        for item in self.structural_elements:
            self.view3d.removeItem(item)
        self.structural_elements = []
        
        # Apply the selected system
        if system_type == "Grid":
            self.apply_grid_system(vertices, density, depth)
        elif system_type == "Diagrid":
            self.apply_diagrid_system(vertices, density, depth)
        elif system_type == "Space Frame":
            self.apply_space_frame(vertices, density, depth)
        elif system_type == "Voronoi":
            self.apply_voronoi_system(vertices, density, depth)
        elif system_type == "Triangulated":
            self.apply_triangulated_system(vertices, density, depth)
        elif system_type == "Waffle":
            self.apply_waffle_system(vertices, density, depth)
    
    def apply_grid_system(self, vertices, density, depth):
        """Apply a grid system to the selected vertices"""
        if len(vertices) < 3:
            return
            
        # For simplicity, let's create a grid in the XZ plane, offset by depth
        # In a real implementation, you'd project onto the actual face
        
        # Find bounds
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Create a grid
        x_vals = np.linspace(min_bounds[0], max_bounds[0], int(density))
        z_vals = np.linspace(min_bounds[2], max_bounds[2], int(density))
        
        # Create horizontal elements
        for x in x_vals:
            pts = np.array([[x, min_bounds[1] + depth, z] for z in z_vals])
            line = gl.GLLinePlotItem(pos=pts, color=pg_color(self.element_color), width=2)
            self.view3d.addItem(line)
            self.structural_elements.append(line)
        
        for z in z_vals:
            pts = np.array([[x, min_bounds[1] + depth, z] for x in x_vals])
            line = gl.GLLinePlotItem(pos=pts, color=pg_color(self.element_color), width=2)
            self.view3d.addItem(line)
            self.structural_elements.append(line)
    
    def apply_diagrid_system(self, vertices, density, depth):
        """Apply a diagonal grid system"""
        if len(vertices) < 3:
            return
            
        # Similar to grid but with diagonal elements
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Create a grid
        x_vals = np.linspace(min_bounds[0], max_bounds[0], int(density))
        z_vals = np.linspace(min_bounds[2], max_bounds[2], int(density))
        
        # Create diagonal elements
        for i in range(len(x_vals) - 1):
            for j in range(len(z_vals) - 1):
                # Diagonal 1
                pts = np.array([
                    [x_vals[i], min_bounds[1] + depth, z_vals[j]],
                    [x_vals[i+1], min_bounds[1] + depth, z_vals[j+1]]
                ])
                line = gl.GLLinePlotItem(pos=pts, color=pg_color(self.element_color), width=2)
                self.view3d.addItem(line)
                self.structural_elements.append(line)
                
                # Diagonal 2
                pts = np.array([
                    [x_vals[i], min_bounds[1] + depth, z_vals[j+1]],
                    [x_vals[i+1], min_bounds[1] + depth, z_vals[j]]
                ])
                line = gl.GLLinePlotItem(pos=pts, color=pg_color(self.element_color), width=2)
                self.view3d.addItem(line)
                self.structural_elements.append(line)
    
    def apply_space_frame(self, vertices, density, depth):
        """Apply a space frame system with top and bottom layers"""
        if len(vertices) < 3:
            return
            
        # Create top layer (similar to grid)
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Create grid points
        x_vals = np.linspace(min_bounds[0], max_bounds[0], int(density))
        z_vals = np.linspace(min_bounds[2], max_bounds[2], int(density))
        
        top_points = []
        for x in x_vals:
            for z in z_vals:
                top_points.append([x, min_bounds[1], z])
        
        # Create bottom layer (offset by depth)
        bottom_points = [[p[0], p[1] + depth, p[2]] for p in top_points]
        
        # Create vertical connections
        for i in range(len(top_points)):
            pts = np.array([top_points[i], bottom_points[i]])
            line = gl.GLLinePlotItem(pos=pts, color=pg_color(self.element_color), width=2)
            self.view3d.addItem(line)
            self.structural_elements.append(line)
        
        # Create horizontal connections (top layer)
        # For simplicity, just connect each point to its neighbors
        for i in range(len(x_vals)):
            for j in range(len(z_vals) - 1):
                idx1 = i * len(z_vals) + j
                idx2 = i * len(z_vals) + (j + 1)
                pts = np.array([top_points[idx1], top_points[idx2]])
                line = gl.GLLinePlotItem(pos=pts, color=pg_color(self.element_color), width=2)
                self.view3d.addItem(line)
                self.structural_elements.append(line)
        
        for i in range(len(x_vals) - 1):
            for j in range(len(z_vals)):
                idx1 = i * len(z_vals) + j
                idx2 = (i + 1) * len(z_vals) + j
                pts = np.array([top_points[idx1], top_points[idx2]])
                line = gl.GLLinePlotItem(pos=pts, color=pg_color(self.element_color), width=2)
                self.view3d.addItem(line)
                self.structural_elements.append(line)
    
    def apply_voronoi_system(self, vertices, density, depth):
        """Apply a Voronoi-based structural system"""
        if len(vertices) < 3:
            return
            
        # Project vertices to 2D (XZ plane for demo)
        points_2d = np.array([[v[0], v[2]] for v in vertices])
        
        # Generate random points in the bounding area
        min_bounds = np.min(points_2d, axis=0)
        max_bounds = np.max(points_2d, axis=0)
        
        # Generate random seed points
        num_points = int(density * 2)  # More points for more detail
        points = np.random.uniform(min_bounds, max_bounds, size=(num_points, 2))
        
        # Compute Voronoi diagram
        vor = Voronoi(points)
        
        # Draw Voronoi edges
        for simplex in vor.ridge_vertices:
            # Ridge from infinity not included
            if -1 not in simplex:
                # Get 2D points
                p1 = vor.vertices[simplex[0]]
                p2 = vor.vertices[simplex[1]]
                
                # Convert to 3D
                p1_3d = [p1[0], min_bounds[1] + depth, p1[1]]
                p2_3d = [p2[0], min_bounds[1] + depth, p2[1]]
                
                # Add line
                pts = np.array([p1_3d, p2_3d])
                line = gl.GLLinePlotItem(pos=pts, color=pg_color(self.element_color), width=2)
                self.view3d.addItem(line)
                self.structural_elements.append(line)
    
    def apply_triangulated_system(self, vertices, density, depth):
        """Apply a triangulated structural system"""
        if len(vertices) < 3:
            return
        
        # Project vertices to 2D (XZ plane for demo)
        points_2d = np.array([[v[0], v[2]] for v in vertices])
        
        # Generate random points in the bounding area
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Generate random seed points
        num_points = int(density * 2)
        random_points_2d = np.random.uniform(
            [min_bounds[0], min_bounds[2]], 
            [max_bounds[0], max_bounds[2]], 
            size=(num_points, 2)
        )
        
        # Compute Delaunay triangulation
        tri = Delaunay(random_points_2d)
        
        # Draw triangulation edges
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                if edge not in edges:
                    edges.add(edge)
                    
                    # Get 3D points
                    p1 = [random_points_2d[edge[0]][0], min_bounds[1] + depth, random_points_2d[edge[0]][1]]
                    p2 = [random_points_2d[edge[1]][0], min_bounds[1] + depth, random_points_2d[edge[1]][1]]
                    
                    # Add line
                    pts = np.array([p1, p2])
                    line = gl.GLLinePlotItem(pos=pts, color=pg_color(self.element_color), width=2)
                    self.view3d.addItem(line)
                    self.structural_elements.append(line)
    
    def apply_waffle_system(self, vertices, density, depth):
        """Apply a waffle structural system (interlocking perpendicular planes)"""
        if len(vertices) < 3:
            return
            
        # Find bounds
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Create X direction slices
        x_vals = np.linspace(min_bounds[0], max_bounds[0], int(density))
        for x in x_vals:
            # Create a vertical plane at this x
            verts = np.array([
                [x, min_bounds[1], min_bounds[2]],
                [x, min_bounds[1] + depth, min_bounds[2]],
                [x, min_bounds[1] + depth, max_bounds[2]],
                [x, min_bounds[1], max_bounds[2]]
            ])
            
            # Create a mesh plane
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            mesh = gl.GLMeshItem(
                vertexes=verts, 
                faces=faces, 
                faceColors=np.array([pg_color(self.element_color, alpha=0.7) for _ in range(len(faces))])
            )
            self.view3d.addItem(mesh)
            self.structural_elements.append(mesh)
        
        # Create Z direction slices
        z_vals = np.linspace(min_bounds[2], max_bounds[2], int(density))
        for z in z_vals:
            # Create a vertical plane at this z
            verts = np.array([
                [min_bounds[0], min_bounds[1], z],
                [min_bounds[0], min_bounds[1] + depth, z],
                [max_bounds[0], min_bounds[1] + depth, z],
                [max_bounds[0], min_bounds[1], z]
            ])
            
            # Create a mesh plane
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            mesh = gl.GLMeshItem(
                vertexes=verts, 
                faces=faces, 
                faceColors=np.array([pg_color(self.element_color, alpha=0.7) for _ in range(len(faces))])
            )
            self.view3d.addItem(mesh)
            self.structural_elements.append(mesh)
    
    def export_structure(self):
        """Export the structural system to a file"""
        # In a real app, use QFileDialog to get the output path
        print("Structure would be exported to file")
        # This would save the structural system to OBJ, STL, etc.

def pg_color(color, alpha=1.0):
    """Convert a color array to the format expected by pyqtgraph"""
    return (color[0]/255, color[1]/255, color[2]/255, alpha)

def main():
    app = QApplication(sys.argv)
    window = StructuralSystemGenerator()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()