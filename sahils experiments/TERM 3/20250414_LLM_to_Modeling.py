import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QPushButton, QComboBox, QLabel, QSlider, 
                           QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QColorDialog,
                           QFileDialog, QListWidget, QMessageBox, QTabWidget, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QDir
from PyQt5.QtGui import QMouseEvent
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import trimesh
from scipy.spatial import Voronoi, Delaunay
import OpenGL.GL as ogl
from OpenGL.GLU import gluUnProject

class GLViewWidgetWithPicking(gl.GLViewWidget):
    """Extended GLViewWidget with picking support"""
    pickFaceSignal = pyqtSignal(int)  # Signal to emit when a face is picked
    hoverFaceSignal = pyqtSignal(int)  # Signal to emit when hovering over a face
    dragSelectionSignal = pyqtSignal(list)  # Signal to emit faces selected by dragging
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_mode = False
        self.mesh_data = None
        self.hover_face = None
        
        # For drag selection
        self.is_dragging = False
        self.drag_start_pos = None
        self.faces_under_drag = []
        
    def setMeshData(self, mesh_data):
        """Store mesh data for picking"""
        self.mesh_data = mesh_data
        
    def enableSelectionMode(self, enable=True):
        """Enable or disable selection mode"""
        self.selection_mode = enable
        
    def mousePressEvent(self, ev):
        if self.selection_mode and self.mesh_data is not None and ev.button() == Qt.LeftButton:
            # Get mouse position
            pos = ev.pos()
            
            # Start drag selection
            self.is_dragging = True
            self.drag_start_pos = pos
            self.faces_under_drag = []
            
            # Also capture the face under cursor when starting the drag
            face_idx = self.pick_face(pos)
            if face_idx is not None and face_idx not in self.faces_under_drag:
                self.faces_under_drag.append(face_idx)
                
            # Don't call parent handler yet to prevent camera movement during selection
            ev.accept()
            return
            
        # Default behavior for camera control
        super().mousePressEvent(ev)
    
    def mouseMoveEvent(self, ev):
        """Handle mouse movement for hover highlighting and drag selection"""
        if self.selection_mode and self.mesh_data is not None:
            # Get mouse position
            pos = ev.pos()
            
            if self.is_dragging:
                # Add any face under the drag to the selection list
                face_idx = self.pick_face(pos)
                if face_idx is not None and face_idx not in self.faces_under_drag:
                    self.faces_under_drag.append(face_idx)
                    # Emit continuous updates for visual feedback during drag
                    self.dragSelectionSignal.emit(self.faces_under_drag)
                ev.accept()
                return
            else:
                # Just hover behavior when not dragging
                face_idx = self.pick_face(pos)
                
                # Only emit if the hover face has changed
                if face_idx != self.hover_face:
                    self.hover_face = face_idx
                    self.hoverFaceSignal.emit(face_idx if face_idx is not None else -1)
        
        # Default behavior for camera control (only when not dragging)
        if not self.is_dragging:
            super().mouseMoveEvent(ev)
    
    def mouseReleaseEvent(self, ev):
        """Handle mouse release to complete drag selection"""
        if self.selection_mode and self.is_dragging and ev.button() == Qt.LeftButton:
            self.is_dragging = False
            # Emit the final list of faces selected in this drag operation
            if self.faces_under_drag:
                self.dragSelectionSignal.emit(self.faces_under_drag)
            ev.accept()
            return
            
        # Default behavior
        super().mouseReleaseEvent(ev)
        
    def pick_face(self, mouse_pos):
        """
        Pick a face from the mesh using ray casting
        Returns the face index that was hit, or None if no hit
        """
        # Get viewport dimensions
        viewport = ogl.glGetIntegerv(ogl.GL_VIEWPORT)
        
        # Get the projection and modelview matrices
        proj_matrix = self.projectionMatrix().data()
        mv_matrix = self.viewMatrix().data()
        
        # Mouse coordinates with origin at bottom left (OpenGL convention)
        x = mouse_pos.x()
        y = viewport[3] - mouse_pos.y()
        
        # Get ray origin and direction in world space
        try:
            # Get near and far points on the ray
            wx1, wy1, wz1 = gluUnProject(x, y, 0.0, mv_matrix, proj_matrix, viewport)
            wx2, wy2, wz2 = gluUnProject(x, y, 1.0, mv_matrix, proj_matrix, viewport)
            
            # Ray origin and direction
            ray_origin = np.array([wx1, wy1, wz1])
            ray_direction = np.array([wx2-wx1, wy2-wy1, wz2-wz1])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            
            # Improved face picking with proper ray-triangle intersection
            if hasattr(self.mesh_data, 'faces') and hasattr(self.mesh_data, 'vertices'):
                faces = self.mesh_data.faces
                vertices = self.mesh_data.vertices
                
                closest_face = None
                min_distance = float('inf')
                
                # Calculate face centers for efficient preliminary filtering
                if not hasattr(self.mesh_data, 'triangles_center'):
                    self.mesh_data.triangles_center = np.array([
                        np.mean([vertices[f[0]], vertices[f[1]], vertices[f[2]]], axis=0)
                        for f in faces
                    ])
                centers = self.mesh_data.triangles_center
                
                # First, filter faces based on dot product with ray direction
                # This eliminates faces that are facing away from the camera
                for i, face in enumerate(faces):
                    # Calculate face normal
                    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                    face_normal = np.cross(v1 - v0, v2 - v0)
                    face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-10)  # Avoid division by zero
                    
                    # Skip faces that face away from the camera (backface culling)
                    if np.dot(face_normal, ray_direction) >= 0:
                        continue
                    
                    # Perform ray-triangle intersection (Möller–Trumbore algorithm)
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    h = np.cross(ray_direction, edge2)
                    a = np.dot(edge1, h)
                    
                    # If ray is parallel to triangle
                    if abs(a) < 1e-6:
                        continue
                    
                    f = 1.0 / a
                    s = ray_origin - v0
                    u = f * np.dot(s, h)
                    
                    # Ray misses triangle
                    if u < 0.0 or u > 1.0:
                        continue
                    
                    q = np.cross(s, edge1)
                    v = f * np.dot(ray_direction, q)
                    
                    # Ray misses triangle
                    if v < 0.0 or u + v > 1.0:
                        continue
                    
                    # Distance to intersection
                    t = f * np.dot(edge2, q)
                    
                    # Intersection is behind ray origin
                    if t < 0:
                        continue
                    
                    # This is the closest intersection so far
                    if t < min_distance:
                        min_distance = t
                        closest_face = i
                
                return closest_face
            
            # Fallback to center-based picking if the above fails
            elif hasattr(self.mesh_data, 'triangles_center'):
                centers = self.mesh_data.triangles_center
                
                closest_face = None
                min_distance = float('inf')
                
                for i, center in enumerate(centers):
                    # Vector from ray origin to face center
                    to_center = center - ray_origin
                    
                    # Project onto ray direction
                    projection = np.dot(to_center, ray_direction)
                    
                    # Skip faces behind the ray
                    if projection < 0:
                        continue
                    
                    # Distance from ray to center
                    closest_point = ray_origin + projection * ray_direction
                    distance = np.linalg.norm(center - closest_point)
                    
                    # Check if this is closer than previous closest
                    if distance < min_distance and distance < 0.2:  # Adding threshold for hit detection
                        min_distance = distance
                        closest_face = i
                
                return closest_face
            
            return None
        except Exception as e:
            print(f"Picking error: {e}")
            return None

class StructuralSystemGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Structural System Generator")
        self.resize(1200, 800)
        
        # Model data
        self.mesh = None
        self.selected_faces = []
        self.structural_elements = []
        self.selection_active = False
        
        # Create models directory if it doesn't exist
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize UI
        self.initUI()
        
    def initUI(self):
        # Main layout
        main_layout = QHBoxLayout()
        
        # 3D Viewport with picking support
        self.view3d = GLViewWidgetWithPicking()
        self.view3d.pickFaceSignal.connect(self.on_face_picked)
        self.view3d.hoverFaceSignal.connect(self.on_face_hovered)
        self.view3d.dragSelectionSignal.connect(self.on_faces_drag_selected)
        main_layout.addWidget(self.view3d, 3)
        
        # Create grid for reference
        grid = gl.GLGridItem()
        grid.setSize(10, 10, 1)
        grid.setSpacing(1, 1, 1)
        self.view3d.addItem(grid)
        
        # Controls panel
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        
        # Tabs for organization
        tabs = QTabWidget()
        
        # Tab 1: Model Management
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        
        # Models group
        models_group = QGroupBox("3D Models")
        models_layout = QVBoxLayout()
        
        # Local models list
        models_layout.addWidget(QLabel("Local Models:"))
        self.models_list = QListWidget()
        self.models_list.itemClicked.connect(self.on_model_selected)
        models_layout.addWidget(self.models_list)
        
        # Model management buttons
        model_btn_layout = QHBoxLayout()
        
        upload_btn = QPushButton("Upload Model")
        upload_btn.clicked.connect(self.upload_model)
        model_btn_layout.addWidget(upload_btn)
        
        delete_btn = QPushButton("Delete Model")
        delete_btn.clicked.connect(self.delete_model)
        model_btn_layout.addWidget(delete_btn)
        
        models_layout.addLayout(model_btn_layout)
        models_group.setLayout(models_layout)
        model_layout.addWidget(models_group)
        
        # Selection group
        selection_group = QGroupBox("Selection")
        selection_layout = QVBoxLayout()
        
        self.selection_mode = QComboBox()
        self.selection_mode.addItems(["Face", "Edge", "Vertex"])
        selection_layout.addWidget(self.selection_mode)
        
        select_btn = QPushButton("Enter Selection Mode")
        select_btn.clicked.connect(self.enter_selection_mode)
        selection_layout.addWidget(select_btn)
        
        # Hover hint
        hover_hint = QLabel("Hover over faces to highlight them for selection")
        hover_hint.setStyleSheet("color: blue;")
        selection_layout.addWidget(hover_hint)
        
        # Selection status label
        self.selection_status = QLabel("Selection Mode: Inactive")
        selection_layout.addWidget(self.selection_status)
        
        clear_selection_btn = QPushButton("Clear Selection")
        clear_selection_btn.clicked.connect(self.clear_selection)
        selection_layout.addWidget(clear_selection_btn)
        
        selection_group.setLayout(selection_layout)
        model_layout.addWidget(selection_group)
        
        # Add model tab
        tabs.addTab(model_tab, "Model & Selection")
        
        # Tab 2: Structural Systems
        systems_tab = QWidget()
        systems_layout = QVBoxLayout(systems_tab)
        
        # Structural systems
        systems_group = QGroupBox("Structural System Type")
        system_type_layout = QVBoxLayout()
        
        self.system_type = QComboBox()
        self.system_type.addItems([
            "Grid", "Diagrid", "Space Frame", 
            "Voronoi", "Triangulated", "Waffle"
        ])
        self.system_type.currentIndexChanged.connect(self.update_system_parameters)
        system_type_layout.addWidget(self.system_type)
        
        systems_group.setLayout(system_type_layout)
        systems_layout.addWidget(systems_group)
        
        # Dynamic parameters group
        self.params_group = QGroupBox("System Parameters")
        self.params_layout = QFormLayout()
        
        # Common parameters
        # Density parameter
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(1, 100)
        self.density_slider.setValue(20)
        self.params_layout.addRow("Density:", self.density_slider)
        
        # Depth parameter
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.01, 5.0)
        self.depth_spin.setValue(0.5)
        self.depth_spin.setSingleStep(0.1)
        self.params_layout.addRow("Depth:", self.depth_spin)
        
        # System-specific parameters (initially empty)
        self.dynamic_params_widgets = {}
        
        # Color selection
        self.color_btn = QPushButton("Select Color")
        self.color_btn.clicked.connect(self.select_color)
        self.element_color = [255, 0, 0]  # Default red
        self.color_btn.setStyleSheet("background-color: rgb(255, 0, 0)")
        self.params_layout.addRow("Color:", self.color_btn)
        
        self.params_group.setLayout(self.params_layout)
        systems_layout.addWidget(self.params_group)
        
        # Apply button
        apply_btn = QPushButton("Apply System")
        apply_btn.clicked.connect(self.apply_system)
        systems_layout.addWidget(apply_btn)
        
        # Export button
        export_btn = QPushButton("Export Structure")
        export_btn.clicked.connect(self.export_structure)
        systems_layout.addWidget(export_btn)
        
        # Add systems tab
        tabs.addTab(systems_tab, "Structural Systems")
        
        # Add tabs to main controls
        controls_layout.addWidget(tabs)
        
        main_layout.addWidget(controls_panel, 1)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Initialize dynamic parameters for the default system
        self.update_system_parameters()
        
        # Refresh models list
        self.refresh_models_list()
    
    def refresh_models_list(self):
        """Refresh the list of local models"""
        self.models_list.clear()
        if os.path.exists(self.models_dir):
            for filename in os.listdir(self.models_dir):
                if filename.lower().endswith(('.obj', '.stl', '.ply', '.glb', '.3ds')):
                    self.models_list.addItem(filename)
    
    def upload_model(self):
        """Upload a 3D model to the local directory"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select 3D Model", "", 
            "3D Models (*.obj *.stl *.ply *.glb *.3ds);;All Files (*)"
        )
        
        if file_path:
            # Copy the file to our local models directory
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.models_dir, filename)
            
            try:
                # Copy the file
                with open(file_path, 'rb') as src_file:
                    with open(dest_path, 'wb') as dest_file:
                        dest_file.write(src_file.read())
                
                # Refresh the list and load the model
                self.refresh_models_list()
                self.load_model(dest_path)
                
                # Select the item in the list
                items = self.models_list.findItems(filename, Qt.MatchExactly)
                if items:
                    self.models_list.setCurrentItem(items[0])
                    
                QMessageBox.information(self, "Upload Successful", 
                                        f"Model '{filename}' uploaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Upload Failed", 
                                     f"Failed to upload model: {str(e)}")
    
    def delete_model(self):
        """Delete the selected model from the local directory"""
        current_item = self.models_list.currentItem()
        if current_item:
            filename = current_item.text()
            file_path = os.path.join(self.models_dir, filename)
            
            reply = QMessageBox.question(
                self, "Confirm Deletion",
                f"Are you sure you want to delete '{filename}'?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    os.remove(file_path)
                    self.refresh_models_list()
                    
                    # Clear the view if we deleted the currently loaded model
                    self.clear_view()
                    
                    QMessageBox.information(self, "Deletion Successful", 
                                           f"Model '{filename}' deleted successfully.")
                except Exception as e:
                    QMessageBox.critical(self, "Deletion Failed", 
                                        f"Failed to delete model: {str(e)}")
    
    def on_model_selected(self, item):
        """Load the selected model when clicked in the list"""
        if item:
            filename = item.text()
            file_path = os.path.join(self.models_dir, filename)
            self.load_model(file_path)
    
    def clear_view(self):
        """Clear the 3D view and reset state"""
        self.view3d.clear()
        
        # Add grid back
        grid = gl.GLGridItem()
        grid.setSize(10, 10, 1)
        grid.setSpacing(1, 1, 1)
        self.view3d.addItem(grid)
        
        # Reset state
        self.mesh = None
        self.selected_faces = []
        self.structural_elements = []
        self.selection_active = False
        self.selection_status.setText("Selection Mode: Inactive")
    
    def load_model(self, file_path=None):
        """Load a 3D model file"""
        # Clear previous mesh from view
        self.clear_view()
        
        if file_path and os.path.exists(file_path):
            try:
                # Load the mesh
                self.mesh = trimesh.load(file_path)
                
                # Position the mesh on top of the grid
                # First, get the bounding box min z value
                min_z = np.min(self.mesh.vertices[:, 2])
                
                # Translate mesh vertices to place bottom at z=0 (on top of the grid)
                translation = np.array([0, 0, 0])
                if min_z != 0:
                    translation[2] = -min_z
                
                # Center the model on the grid horizontally (x,y)
                bbox_center = np.mean(self.mesh.bounds, axis=0)
                translation[0] = -bbox_center[0]
                translation[1] = -bbox_center[1]
                
                # Apply the translation
                self.mesh.vertices += translation
                
                # Display the mesh with reduced opacity
                vertices = self.mesh.vertices
                faces = self.mesh.faces
                
                # Create face colors with reduced opacity (0.3)
                face_colors = np.array([[0.7, 0.7, 1.0, 0.3] for _ in range(len(faces))])
                
                # Create the mesh item
                mesh_item = gl.GLMeshItem(
                    vertexes=vertices, 
                    faces=faces, 
                    faceColors=face_colors,
                    smooth=False,
                    drawEdges=True,
                    edgeColor=(0.9, 0.9, 1.0, 1.0)  # Light blue-white color for edges
                )
                self.view3d.addItem(mesh_item)
                self.mesh_item = mesh_item
                self.view3d.setMeshData(self.mesh)
                
            except Exception as e:
                QMessageBox.critical(self, "Loading Failed", 
                                    f"Failed to load model: {str(e)}")
                # Fallback to a simple cube
                self.load_default_cube()
        else:
            # For demo, create a simple cube mesh if no file is provided
            self.load_default_cube()
    
    def load_default_cube(self):
        """Load a default cube mesh for demonstration"""
        self.mesh = trimesh.creation.box(extents=[2, 2, 2])
        
        # Position the mesh on top of the grid
        # Center the cube horizontally and place bottom at z=0
        # First, get the current vertices and find the lowest z value
        min_z = np.min(self.mesh.vertices[:, 2])
        
        # Translate mesh to place it on the grid
        if min_z < 0:
            translation = np.array([0, 0, -min_z])
            self.mesh.vertices += translation
        
        # Display the mesh with reduced opacity
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        # Create face colors with reduced opacity (0.3)
        face_colors = np.array([[0.7, 0.7, 1.0, 0.3] for _ in range(len(faces))])
        
        # Create the mesh item
        mesh_item = gl.GLMeshItem(
            vertexes=vertices, 
            faces=faces, 
            faceColors=face_colors,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.9, 0.9, 1.0, 1.0)  # Light blue-white color for edges
        )
        self.view3d.addItem(mesh_item)
        self.mesh_item = mesh_item
        self.view3d.setMeshData(self.mesh)
        
        QMessageBox.information(self, "Default Model", 
                               "Loaded a default cube mesh for demonstration.")
    
    def update_system_parameters(self):
        """Update the parameters form based on the selected system type"""
        # Clear any existing dynamic parameters
        for widget in self.dynamic_params_widgets.values():
            widget_item = self.params_layout.labelForField(widget)
            if widget_item:
                self.params_layout.removeRow(widget_item)
        self.dynamic_params_widgets = {}
        
        system_type = self.system_type.currentText()
        
        # Add system-specific parameters
        if system_type == "Grid":
            # Column spacing
            self.dynamic_params_widgets["column_spacing"] = QDoubleSpinBox()
            self.dynamic_params_widgets["column_spacing"].setRange(0.1, 10.0)
            self.dynamic_params_widgets["column_spacing"].setValue(1.0)
            self.dynamic_params_widgets["column_spacing"].setSingleStep(0.1)
            self.params_layout.addRow("Column Spacing:", self.dynamic_params_widgets["column_spacing"])
            
            # Column thickness
            self.dynamic_params_widgets["column_thickness"] = QDoubleSpinBox()
            self.dynamic_params_widgets["column_thickness"].setRange(0.01, 1.0)
            self.dynamic_params_widgets["column_thickness"].setValue(0.1)
            self.dynamic_params_widgets["column_thickness"].setSingleStep(0.01)
            self.params_layout.addRow("Column Thickness:", self.dynamic_params_widgets["column_thickness"])
            
        elif system_type == "Diagrid":
            # Angle
            self.dynamic_params_widgets["diagrid_angle"] = QSlider(Qt.Horizontal)
            self.dynamic_params_widgets["diagrid_angle"].setRange(30, 80)
            self.dynamic_params_widgets["diagrid_angle"].setValue(45)
            self.params_layout.addRow("Diagrid Angle:", self.dynamic_params_widgets["diagrid_angle"])
            
            # Module height
            self.dynamic_params_widgets["module_height"] = QDoubleSpinBox()
            self.dynamic_params_widgets["module_height"].setRange(0.5, 5.0)
            self.dynamic_params_widgets["module_height"].setValue(1.0)
            self.dynamic_params_widgets["module_height"].setSingleStep(0.1)
            self.params_layout.addRow("Module Height:", self.dynamic_params_widgets["module_height"])
            
        elif system_type == "Space Frame":
            # Node connection type
            self.dynamic_params_widgets["node_type"] = QComboBox()
            self.dynamic_params_widgets["node_type"].addItems(["Spherical", "Rigid", "Flexible"])
            self.params_layout.addRow("Node Type:", self.dynamic_params_widgets["node_type"])
            
            # Strut diameter
            self.dynamic_params_widgets["strut_diameter"] = QDoubleSpinBox()
            self.dynamic_params_widgets["strut_diameter"].setRange(0.01, 0.5)
            self.dynamic_params_widgets["strut_diameter"].setValue(0.05)
            self.dynamic_params_widgets["strut_diameter"].setSingleStep(0.01)
            self.params_layout.addRow("Strut Diameter:", self.dynamic_params_widgets["strut_diameter"])
            
        elif system_type == "Voronoi":
            # Seed count
            self.dynamic_params_widgets["seed_count"] = QSpinBox()
            self.dynamic_params_widgets["seed_count"].setRange(5, 100)
            self.dynamic_params_widgets["seed_count"].setValue(20)
            self.params_layout.addRow("Seed Count:", self.dynamic_params_widgets["seed_count"])
            
            # Cell regularity
            self.dynamic_params_widgets["cell_regularity"] = QSlider(Qt.Horizontal)
            self.dynamic_params_widgets["cell_regularity"].setRange(0, 100)
            self.dynamic_params_widgets["cell_regularity"].setValue(50)
            self.params_layout.addRow("Cell Regularity:", self.dynamic_params_widgets["cell_regularity"])
            
        elif system_type == "Triangulated":
            # Subdivision level
            self.dynamic_params_widgets["subdivision"] = QSpinBox()
            self.dynamic_params_widgets["subdivision"].setRange(1, 5)
            self.dynamic_params_widgets["subdivision"].setValue(2)
            self.params_layout.addRow("Subdivision Level:", self.dynamic_params_widgets["subdivision"])
            
            # Edge thickness
            self.dynamic_params_widgets["edge_thickness"] = QDoubleSpinBox()
            self.dynamic_params_widgets["edge_thickness"].setRange(0.01, 0.5)
            self.dynamic_params_widgets["edge_thickness"].setValue(0.05)
            self.dynamic_params_widgets["edge_thickness"].setSingleStep(0.01)
            self.params_layout.addRow("Edge Thickness:", self.dynamic_params_widgets["edge_thickness"])
            
        elif system_type == "Waffle":
            # Slot width
            self.dynamic_params_widgets["slot_width"] = QDoubleSpinBox()
            self.dynamic_params_widgets["slot_width"].setRange(0.01, 0.5)
            self.dynamic_params_widgets["slot_width"].setValue(0.1)
            self.dynamic_params_widgets["slot_width"].setSingleStep(0.01)
            self.params_layout.addRow("Slot Width:", self.dynamic_params_widgets["slot_width"])
            
            # Material thickness
            self.dynamic_params_widgets["material_thickness"] = QDoubleSpinBox()
            self.dynamic_params_widgets["material_thickness"].setRange(0.01, 0.3)
            self.dynamic_params_widgets["material_thickness"].setValue(0.05)
            self.dynamic_params_widgets["material_thickness"].setSingleStep(0.01)
            self.params_layout.addRow("Material Thickness:", self.dynamic_params_widgets["material_thickness"])
            
            # Direction
            self.dynamic_params_widgets["direction"] = QComboBox()
            self.dynamic_params_widgets["direction"].addItems(["Orthogonal", "Radial", "Contour"])
            self.params_layout.addRow("Direction:", self.dynamic_params_widgets["direction"])

    def on_face_hovered(self, face_idx):
        """Handle face hover for highlighting during selection mode"""
        if not self.selection_active or self.mesh is None:
            return
        
        if face_idx == -1:  # No face being hovered
            # Reset colors to normal or selected state
            face_colors = np.array([[0.7, 0.7, 1.0, 0.5] for _ in range(len(self.mesh.faces))])
            
            # Highlight selected faces
            for idx in self.selected_faces:
                if idx < len(face_colors):
                    face_colors[idx] = [0.2, 0.8, 0.2, 0.8]  # Green for selected
            
            self.mesh_item.setMeshData(
                vertexes=self.mesh.vertices,
                faces=self.mesh.faces,
                faceColors=face_colors
            )
        else:
            # Highlight the hovered face
            face_colors = np.array([[0.7, 0.7, 1.0, 0.5] for _ in range(len(self.mesh.faces))])
            
            # Highlight selected faces
            for idx in self.selected_faces:
                if idx < len(face_colors):
                    face_colors[idx] = [0.2, 0.8, 0.2, 0.8]  # Green for selected
            
            # Highlight hovered face (unless already selected)
            if face_idx not in self.selected_faces:
                face_colors[face_idx] = [0.8, 0.8, 0.0, 0.8]  # Yellow for hover
            
            self.mesh_item.setMeshData(
                vertexes=self.mesh.vertices,
                faces=self.mesh.faces,
                faceColors=face_colors
            )
    
    def on_face_picked(self, face_idx):
        """Handle face picking for selection"""
        if not self.selection_active or self.mesh is None:
            return
        
        # Toggle selection of the face
        if face_idx in self.selected_faces:
            self.selected_faces.remove(face_idx)
        else:
            self.selected_faces.append(face_idx)
        
        # Update face colors
        face_colors = np.array([[0.7, 0.7, 1.0, 0.5] for _ in range(len(self.mesh.faces))])
        
        # Highlight selected faces
        for idx in self.selected_faces:
            if idx < len(face_colors):
                face_colors[idx] = [0.2, 0.8, 0.2, 0.8]  # Green for selected
                
        self.mesh_item.setMeshData(
            vertexes=self.mesh.vertices,
            faces=self.mesh.faces,
            faceColors=face_colors
        )
        
        # Update selection status
        self.selection_status.setText(f"Selection Mode: Active - {len(self.selected_faces)} faces selected")
    
    def on_faces_drag_selected(self, face_indices):
        """Handle faces selected via drag operation"""
        if not self.selection_active or self.mesh is None:
            return
        
        # Update the selected faces list
        for idx in face_indices:
            if idx not in self.selected_faces:
                self.selected_faces.append(idx)
        
        # Update face colors to show selection
        face_colors = np.array([[0.7, 0.7, 1.0, 0.5] for _ in range(len(self.mesh.faces))])
        
        # Highlight all selected faces
        for idx in self.selected_faces:
            if idx < len(face_colors):
                face_colors[idx] = [0.2, 0.8, 0.2, 0.8]  # Green for selected
                
        self.mesh_item.setMeshData(
            vertexes=self.mesh.vertices,
            faces=self.mesh.faces,
            faceColors=face_colors
        )
        
        # Update selection status
        self.selection_status.setText(f"Selection Mode: Active - {len(self.selected_faces)} faces selected")
    
    def enter_selection_mode(self):
        """Enter or exit face selection mode"""
        self.selection_active = not self.selection_active
        self.view3d.enableSelectionMode(self.selection_active)
        
        if self.selection_active:
            self.selection_status.setText("Selection Mode: Active - Click on faces to select")
        else:
            self.selection_status.setText("Selection Mode: Inactive")
    
    def clear_selection(self):
        """Clear all selected faces"""
        self.selected_faces = []
        
        if self.mesh is not None:
            # Reset face colors
            face_colors = np.array([[0.7, 0.7, 1.0, 0.5] for _ in range(len(self.mesh.faces))])
            self.mesh_item.setMeshData(
                vertexes=self.mesh.vertices,
                faces=self.mesh.faces,
                faceColors=face_colors
            )
        
        self.selection_status.setText("Selection Mode: Active - 0 faces selected" if self.selection_active else "Selection Mode: Inactive")
    
    def select_color(self):
        """Open color dialog to select element color"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.element_color = [color.red(), color.green(), color.blue()]
            self.color_btn.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})")
    
    def apply_system(self):
        """Apply the selected structural system to the selected faces"""
        if not self.mesh or not self.selected_faces:
            QMessageBox.warning(self, "Selection Required", 
                               "Please select faces to apply the structural system.")
            return
            
        system_type = self.system_type.currentText()
        
        # Get common parameters
        density = self.density_slider.value()
        depth = self.depth_spin.value()
        color = self.element_color
        
        # Collection for any new elements
        new_elements = []
        
        # Apply system based on type
        if system_type == "Grid":
            # Specific parameters for Grid system
            column_spacing = self.dynamic_params_widgets["column_spacing"].value()
            column_thickness = self.dynamic_params_widgets["column_thickness"].value()
            
            # Generate grid elements for each selected face
            for face_idx in self.selected_faces:
                # Get face data
                face = self.mesh.faces[face_idx]
                vertices = self.mesh.vertices[face]
                
                # Create grid on face
                elements = self.generate_grid_system(vertices, density, depth, 
                                                   column_spacing, column_thickness)
                new_elements.extend(elements)
        
        elif system_type == "Diagrid":
            # Specific parameters for Diagrid system
            diagrid_angle = self.dynamic_params_widgets["diagrid_angle"].value()
            module_height = self.dynamic_params_widgets["module_height"].value()
            
            # Generate diagrid elements for each selected face
            for face_idx in self.selected_faces:
                # Get face data
                face = self.mesh.faces[face_idx]
                vertices = self.mesh.vertices[face]
                
                # Create diagrid on face
                elements = self.generate_diagrid_system(vertices, density, depth,
                                                      diagrid_angle, module_height)
                new_elements.extend(elements)
        
        elif system_type == "Space Frame":
            # Specific parameters for Space Frame
            node_type = self.dynamic_params_widgets["node_type"].currentText()
            strut_diameter = self.dynamic_params_widgets["strut_diameter"].value()
            
            # Generate space frame elements for each selected face
            for face_idx in self.selected_faces:
                # Get face data
                face = self.mesh.faces[face_idx]
                vertices = self.mesh.vertices[face]
                
                # Create space frame on face
                elements = self.generate_space_frame(vertices, density, depth,
                                                   node_type, strut_diameter)
                new_elements.extend(elements)
        
        elif system_type == "Voronoi":
            # Specific parameters for Voronoi
            seed_count = self.dynamic_params_widgets["seed_count"].value()
            cell_regularity = self.dynamic_params_widgets["cell_regularity"].value() / 100.0
            
            # Generate voronoi elements for each selected face
            for face_idx in self.selected_faces:
                # Get face data
                face = self.mesh.faces[face_idx]
                vertices = self.mesh.vertices[face]
                
                # Create voronoi on face
                elements = self.generate_voronoi_system(vertices, depth,
                                                      seed_count, cell_regularity)
                new_elements.extend(elements)
        
        elif system_type == "Triangulated":
            # Specific parameters for Triangulated
            subdivision = self.dynamic_params_widgets["subdivision"].value()
            edge_thickness = self.dynamic_params_widgets["edge_thickness"].value()
            
            # Generate triangulated elements for each selected face
            for face_idx in self.selected_faces:
                # Get face data
                face = self.mesh.faces[face_idx]
                vertices = self.mesh.vertices[face]
                
                # Create triangulated system on face
                elements = self.generate_triangulated_system(vertices, depth,
                                                          subdivision, edge_thickness)
                new_elements.extend(elements)
        
        elif system_type == "Waffle":
            # Specific parameters for Waffle
            slot_width = self.dynamic_params_widgets["slot_width"].value()
            material_thickness = self.dynamic_params_widgets["material_thickness"].value()
            direction = self.dynamic_params_widgets["direction"].currentText()
            
            # Generate waffle elements for each selected face
            for face_idx in self.selected_faces:
                # Get face data
                face = self.mesh.faces[face_idx]
                vertices = self.mesh.vertices[face]
                
                # Create waffle system on face
                elements = self.generate_waffle_system(vertices, depth,
                                                     slot_width, material_thickness,
                                                     direction)
                new_elements.extend(elements)
                
        # Add new elements to the scene
        for element in new_elements:
            self.view3d.addItem(element)
            
        # Store elements for tracking
        self.structural_elements.extend(new_elements)
        
        QMessageBox.information(self, "System Applied", 
                               f"{system_type} system applied to {len(self.selected_faces)} faces.")
    
    def generate_grid_system(self, face_vertices, density, depth, column_spacing, column_thickness):
        """Generate a grid structural system for a face"""
        elements = []
        
        # Compute face normal and center
        v0, v1, v2 = face_vertices
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)
        center = np.mean(face_vertices, axis=0)
        
        # Create an orthogonal basis in the face plane
        if np.abs(normal[2]) < 0.9:
            # If normal is not too close to z-axis, use z-axis for reference
            ref = np.array([0, 0, 1])
        else:
            # Otherwise use x-axis
            ref = np.array([1, 0, 0])
            
        u = np.cross(normal, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # Find bounding box in the face plane
        min_u = min_v = float('inf')
        max_u = max_v = float('-inf')
        
        for vertex in face_vertices:
            rel_pos = vertex - center
            u_coord = np.dot(rel_pos, u)
            v_coord = np.dot(rel_pos, v)
            
            min_u = min(min_u, u_coord)
            max_u = max(max_u, u_coord)
            min_v = min(min_v, v_coord)
            max_v = max(max_v, v_coord)
        
        # Expand slightly to ensure coverage
        margin = 0.1
        min_u -= margin
        max_u += margin
        min_v -= margin
        max_v += margin
        
        # Create grid lines
        cols = max(2, int((max_u - min_u) / column_spacing))
        rows = max(2, int((max_v - min_v) / column_spacing))
        
        # Adjust spacing to evenly distribute
        u_spacing = (max_u - min_u) / (cols - 1)
        v_spacing = (max_v - min_v) / (rows - 1)
        
        # Create columns (vertical lines)
        for i in range(cols):
            u_coord = min_u + i * u_spacing
            
            # Line endpoints in 3D
            start = center + u * u_coord + v * min_v
            end = center + u * u_coord + v * max_v
            
            # Create line with the specified thickness
            line = self.create_cylinder(start, end, column_thickness, pg_color(self.element_color))
            elements.append(line)
        
        # Create beams (horizontal lines)
        for j in range(rows):
            v_coord = min_v + j * v_spacing
            
            # Line endpoints in 3D
            start = center + u * min_u + v * v_coord
            end = center + u * max_u + v * v_coord
            
            # Create line with the specified thickness
            line = self.create_cylinder(start, end, column_thickness, pg_color(self.element_color))
            elements.append(line)
            
        return elements
    
    def generate_diagrid_system(self, face_vertices, density, depth, diagrid_angle, module_height):
        """Generate a diagrid structural system for a face"""
        elements = []
        
        # Similar approach to grid, but with diagonal elements
        # Convert the angle from degrees to radians
        angle_rad = np.radians(diagrid_angle)
        
        # Compute face normal and center
        v0, v1, v2 = face_vertices
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)
        center = np.mean(face_vertices, axis=0)
        
        # Create an orthogonal basis in the face plane
        if np.abs(normal[2]) < 0.9:
            # If normal is not too close to z-axis, use z-axis for reference
            ref = np.array([0, 0, 1])
        else:
            # Otherwise use x-axis
            ref = np.array([1, 0, 0])
            
        u = np.cross(normal, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # Find bounding box in the face plane
        min_u = min_v = float('inf')
        max_u = max_v = float('-inf')
        
        for vertex in face_vertices:
            rel_pos = vertex - center
            u_coord = np.dot(rel_pos, u)
            v_coord = np.dot(rel_pos, v)
            
            min_u = min(min_u, u_coord)
            max_u = max(max_u, u_coord)
            min_v = min(min_v, v_coord)
            max_v = max(max_v, v_coord)
        
        # Expand slightly to ensure coverage
        margin = 0.1
        min_u -= margin
        max_u += margin
        min_v -= margin
        max_v += margin
        
        # Calculate module width based on angle and height
        module_width = module_height / np.tan(angle_rad)
        
        # Number of modules horizontally and vertically
        n_modules_h = max(1, int((max_u - min_u) / module_width))
        n_modules_v = max(1, int((max_v - min_v) / module_height))
        
        # Adjust module dimensions to fit evenly
        module_width = (max_u - min_u) / n_modules_h
        module_height = (max_v - min_v) / n_modules_v
        
        # Thickness of the diagonals
        diagonal_thickness = 0.05
        
        # Create diagonal elements
        for i in range(n_modules_h):
            for j in range(n_modules_v):
                # Module corners
                bottom_left = center + u * (min_u + i * module_width) + v * (min_v + j * module_height)
                bottom_right = center + u * (min_u + (i+1) * module_width) + v * (min_v + j * module_height)
                top_left = center + u * (min_u + i * module_width) + v * (min_v + (j+1) * module_height)
                top_right = center + u * (min_u + (i+1) * module_width) + v * (min_v + (j+1) * module_height)
                
                # Create diagonals (X-pattern)
                diagonal1 = self.create_cylinder(bottom_left, top_right, diagonal_thickness, pg_color(self.element_color))
                diagonal2 = self.create_cylinder(bottom_right, top_left, diagonal_thickness, pg_color(self.element_color))
                
                elements.append(diagonal1)
                elements.append(diagonal2)
        
        return elements
    
    def create_cylinder(self, start, end, radius, color):
        """Create a cylinder between two points with given radius and color"""
        # Calculate cylinder length
        length = np.linalg.norm(end - start)
        
        # Create the cylinder mesh
        cylinder = gl.MeshData.cylinder(rows=10, cols=10, radius=[radius, radius], length=length)
        
        # Calculate the orientation
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        
        # Default cylinder direction in the mesh is along the y-axis
        default_dir = np.array([0, 1, 0])
        
        # Calculate the rotation axis and angle
        rotation_axis = np.cross(default_dir, direction)
        
        # If rotation axis has zero length (vectors are parallel or anti-parallel)
        if np.linalg.norm(rotation_axis) < 1e-6:
            if direction[1] < 0:  # Anti-parallel case
                # Rotate 180 degrees around x-axis
                rotation_axis = np.array([1, 0, 0])
                rotation_angle = np.pi
            else:  # Parallel case
                # No rotation needed
                rotation_axis = np.array([1, 0, 0])
                rotation_angle = 0
        else:
            # Normalize rotation axis
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            # Calculate rotation angle
            rotation_angle = np.arccos(np.dot(default_dir, direction))
        
        # Create a GLMeshItem
        mesh = gl.GLMeshItem(meshdata=cylinder, color=color, glOptions='translucent')
        
        # Create a rotation matrix
        tr = pg.Transform3D()
        tr.translate(start[0], start[1], start[2])
        tr.rotate(np.degrees(rotation_angle), rotation_axis[0], rotation_axis[1], rotation_axis[2])
        
        # Apply the transformation
        mesh.setTransform(tr)
        
        return mesh
    
    def export_structure(self):
        """Export the generated structural system"""
        # Simple implementation: export as STL
        if not self.structural_elements:
            QMessageBox.warning(self, "Nothing to Export", 
                               "Please apply a structural system before exporting.")
            return
        
        # In a real app, you'd combine all meshes and export
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Export Structure", "", 
            "STL Files (*.stl);;OBJ Files (*.obj)"
        )
        
        if file_path:
            # For demo, show a success message
            QMessageBox.information(self, "Export", "Structure exported successfully!")
    
    # Placeholders for other structural system generators
    def generate_space_frame(self, face_vertices, density, depth, node_type, strut_diameter):
        """Generate a space frame structural system"""
        # Placeholder implementation
        return []
    
    def generate_voronoi_system(self, face_vertices, depth, seed_count, cell_regularity):
        """Generate a Voronoi structural system"""
        # Placeholder implementation
        return []
    
    def generate_triangulated_system(self, face_vertices, depth, subdivision, edge_thickness):
        """Generate a triangulated structural system"""
        # Placeholder implementation
        return []
    
    def generate_waffle_system(self, face_vertices, depth, slot_width, material_thickness, direction):
        """Generate a waffle structural system"""
        # Placeholder implementation
        return []

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
`