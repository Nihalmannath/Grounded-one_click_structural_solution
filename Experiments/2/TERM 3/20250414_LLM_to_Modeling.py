import sys
import os
import glob  # Add missing glob import
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
    pickGeometrySignal = pyqtSignal(object)  # Signal to emit when a geometry is picked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_mode = False
        self.mesh_data = None
        self.hover_face = None
        self.selection_type = "Face"  # Can be "Face" or "Geometry"
        
        # For drag selection
        self.is_dragging = False
        self.drag_start_pos = None
        self.current_drag_pos = None
        self.faces_under_drag = []
        
        # For geometry selection
        self.geometries = []  # List of geometry objects in the scene
        self.selected_geometry = None
        self.hover_geometry = None
        
        # Override the GLViewWidget's default mouse behavior
        self.setMouseTracking(True)
        
    def setMeshData(self, mesh_data):
        """Store mesh data for picking"""
        self.mesh_data = mesh_data
        
    def addGeometry(self, geometry_obj, mesh_item):
        """Add a geometry object to the selection list"""
        self.geometries.append({
            'data': geometry_obj,
            'mesh_item': mesh_item,
            'bounds': geometry_obj.bounds if hasattr(geometry_obj, 'bounds') else None
        })
        
    def clearGeometries(self):
        """Clear the list of selectable geometries"""
        self.geometries = []
        self.selected_geometry = None
        self.hover_geometry = None
        
    def setSelectionType(self, selection_type):
        """Set selection type (Face or Geometry)"""
        self.selection_type = selection_type
        
    def enableSelectionMode(self, enable=True):
        """Enable or disable selection mode"""
        self.selection_mode = enable

    def mousePressEvent(self, ev):
        # Middle mouse button for camera rotation/movement
        if ev.button() == Qt.MiddleButton:
            super().mousePressEvent(ev)
            return
            
        # Left button for selection
        if self.selection_mode and ev.button() == Qt.LeftButton:
            # Get mouse position
            pos = ev.pos()
            
            if self.selection_type == "Geometry" and self.geometries:
                # Pick geometry directly without dragging
                geometry_idx = self.pick_geometry(pos)
                if geometry_idx is not None:
                    self.selected_geometry = geometry_idx
                    self.pickGeometrySignal.emit(self.geometries[geometry_idx])
                ev.accept()
                return
            
        # For any other buttons, use default behavior
        super().mousePressEvent(ev)
    
    def pick_geometry(self, mouse_pos):
        """
        Pick a geometry from the scene
        Returns the index of the geometry that was hit, or None if no hit
        """
        if not self.geometries:
            return None
            
        # Get viewport dimensions
        viewport = self.getViewport()
        
        # Get the projection and modelview matrices
        proj_matrix = self.projectionMatrix().data()
        mv_matrix = self.viewMatrix().data()
        
        # Create a ray from the mouse position
        x, y = mouse_pos.x(), mouse_pos.y()
        
        # Convert to normalized device coordinates
        # The viewport is likely a tuple of (x, y, width, height)
        width, height = viewport[2], viewport[3]
        
        # Create a ray in world space
        ndc_x = (2.0 * x / width) - 1.0
        ndc_y = 1.0 - (2.0 * y / height)
        
        ray_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
        ray_eye = np.linalg.inv(np.array(proj_matrix).reshape(4, 4)) @ ray_clip
        ray_eye[2] = -1.0
        ray_eye[3] = 0.0
        ray_world = np.linalg.inv(np.array(mv_matrix).reshape(4, 4)) @ ray_eye
        ray_direction = np.array([ray_world[0], ray_world[1], ray_world[2]])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        # Get camera position (ray origin)
        camera_pos = self.cameraPosition()
        ray_origin = np.array([camera_pos.x(), camera_pos.y(), camera_pos.z()])
        
        # Check intersection with all geometries
        closest_geometry = None
        min_distance = float('inf')
        
        for i, geom in enumerate(self.geometries):
            if 'data' not in geom or not hasattr(geom['data'], 'faces') or not hasattr(geom['data'], 'vertices'):
                continue
                
            mesh = geom['data']
            
            # Check if ray intersects bounding box first
            if hasattr(mesh, 'bounds'):
                bbox_min, bbox_max = mesh.bounds
                if not self._ray_bbox_intersect(ray_origin, ray_direction, bbox_min, bbox_max):
                    continue
            
            # Check intersection with faces
            for face_idx, face in enumerate(mesh.faces):
                vertices = mesh.vertices[face]
                
                # Ray-triangle intersection (Möller–Trumbore algorithm)
                v0, v1, v2 = vertices
                
                # Calculate edges
                edge1 = v1 - v0
                edge2 = v2 - v0
                
                # Calculate determinant
                pvec = np.cross(ray_direction, edge2)
                det = np.dot(edge1, pvec)
                
                # If backface culling is off, we need to check both sides
                # If det is close to 0, ray lies in plane of triangle
                if abs(det) < 1e-6:
                    continue
                
                inv_det = 1.0 / det
                
                # Calculate distance from v0 to ray origin
                tvec = ray_origin - v0
                
                # Calculate u parameter and test bounds
                u = np.dot(tvec, pvec) * inv_det
                if u < 0.0 or u > 1.0:
                    continue
                
                # Calculate v parameter and test bounds
                qvec = np.cross(tvec, edge1)
                v = np.dot(ray_direction, qvec) * inv_det
                if v < 0.0 or u + v > 1.0:
                    continue
                
                # Calculate t, ray intersects triangle
                t = np.dot(edge2, qvec) * inv_det
                
                # Only consider positive t (in front of camera)
                if t <= 0:
                    continue
                
                # If this is closer than previous hits, update
                if t < min_distance:
                    min_distance = t
                    closest_geometry = i
        
        return closest_geometry
    
    def _ray_bbox_intersect(self, ray_origin, ray_direction, bbox_min, bbox_max):
        """Check if ray intersects axis-aligned bounding box"""
        t_min = -float('inf')
        t_max = float('inf')
        
        # Check intersection with each pair of planes
        for i in range(3):
            if abs(ray_direction[i]) < 1e-6:
                # Ray is parallel to slab in this dimension
                if ray_origin[i] < bbox_min[i] or ray_origin[i] > bbox_max[i]:
                    return False
            else:
                # Compute intersection t values
                t1 = (bbox_min[i] - ray_origin[i]) / ray_direction[i]
                t2 = (bbox_max[i] - ray_origin[i]) / ray_direction[i]
                
                # Make sure t1 is the smaller value
                if t1 > t2:
                    t1, t2 = t2, t1
                
                # Update t_min and t_max
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                
                # No intersection if t_max < t_min
                if t_max < t_min:
                    return False
        
        # Ray intersects all 3 slabs
        return True

class StructuralSystemGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Structural System Generator")
        self.resize(1200, 800)
        
        # Model data
        self.mesh = None
        self.structural_elements = []
        self.selection_active = False
        self.model_visible = True  # Track model visibility
        self.structural_systems = []  # Track structural systems separately
        
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
        self.view3d.pickGeometrySignal.connect(self.on_geometry_selected)
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

        # Model visibility toggle
        self.visibility_btn = QPushButton("Hide Model")
        self.visibility_btn.clicked.connect(self.toggle_model_visibility)
        models_layout.addWidget(self.visibility_btn)
        
        models_group.setLayout(models_layout)
        model_layout.addWidget(models_group)
        
        # Geometries group - replaces the selection group
        geometries_group = QGroupBox("Geometries")
        geometries_layout = QVBoxLayout()
        
        geometries_layout.addWidget(QLabel("Available Geometries:"))
        self.geometries_list = QListWidget()
        self.geometries_list.itemClicked.connect(self.on_geometry_list_selected)
        geometries_layout.addWidget(self.geometries_list)
        
        # Selection status label
        self.selection_status = QLabel("No geometry selected")
        geometries_layout.addWidget(self.selection_status)
        
        # Refresh geometries button
        refresh_geoms_btn = QPushButton("Refresh Geometries")
        refresh_geoms_btn.clicked.connect(self.refresh_geometries_list)
        geometries_layout.addWidget(refresh_geoms_btn)
        
        geometries_group.setLayout(geometries_layout)
        model_layout.addWidget(geometries_group)
        
        # Add model tab
        tabs.addTab(model_tab, "Model & Geometries")
        
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

        # Systems management group
        systems_manage_group = QGroupBox("Systems Management")
        systems_manage_layout = QVBoxLayout()
        
        # Systems list
        systems_manage_layout.addWidget(QLabel("Applied Systems:"))
        self.systems_list = QListWidget()
        systems_manage_layout.addWidget(self.systems_list)
        
        # Systems management buttons
        systems_btn_layout = QHBoxLayout()
        
        delete_system_btn = QPushButton("Delete Selected System")
        delete_system_btn.clicked.connect(self.delete_system)
        systems_btn_layout.addWidget(delete_system_btn)
        
        clear_all_systems_btn = QPushButton("Clear All Systems")
        clear_all_systems_btn.clicked.connect(self.clear_all_systems)
        systems_btn_layout.addWidget(clear_all_systems_btn)
        
        systems_manage_layout.addLayout(systems_btn_layout)
        systems_manage_group.setLayout(systems_manage_layout)
        systems_layout.addWidget(systems_manage_group)
        
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
    
    def upload_model(self):
        """Open a file dialog to upload a 3D model file"""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("3D Models (*.obj *.stl *.ply *.glb *.gltf)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setDirectory(QDir.homePath())
        
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                source_path = file_paths[0]
                file_name = os.path.basename(source_path)
                destination_path = os.path.join(self.models_dir, file_name)
                
                try:
                    # Copy the file to our models directory
                    import shutil
                    shutil.copy2(source_path, destination_path)
                    
                    # Refresh the models list
                    self.refresh_models_list()
                    
                    # Load the model
                    self.load_model(destination_path)
                    
                    # Update selection status
                    self.selection_status.setText(f"Uploaded and loaded: {file_name}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to upload model: {str(e)}")
    
    def load_model(self, file_path):
        """Load a 3D model from file and display it in the 3D view"""
        try:
            # Load the mesh using trimesh
            mesh = trimesh.load(file_path)
            filename = os.path.basename(file_path)

            # Check if this is a multi-geometry file
            if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
                # This is a scene with multiple geometries
                print(f"Detected multi-geometry file with {len(mesh.geometry)} geometries")
                
                # Process each geometry separately
                for i, (geom_name, geometry) in enumerate(mesh.geometry.items()):
                    # Set metadata
                    geometry.metadata = {
                        'name': f"{filename} - {geom_name}",
                        'path': file_path,
                        'index': i
                    }
                    
                    # Center and place on grid
                    bounds = geometry.bounds
                    min_bounds, max_bounds = bounds
                    center = (min_bounds + max_bounds) / 2.0
                    
                    translation = np.eye(4)
                    translation[0, 3] = -center[0]
                    translation[1, 3] = -center[1]
                    translation[2, 3] = -min_bounds[2]
                    
                    geometry.apply_transform(translation)
                    
                    # Create GL mesh
                    vertices = geometry.vertices
                    faces = geometry.faces
                    face_colors = np.array([[0.7, 0.7, 1.0, 0.2] for _ in range(len(faces))])
                    
                    mesh_item = gl.GLMeshItem(
                        vertexes=vertices,
                        faces=faces,
                        faceColors=face_colors,
                        smooth=True,
                        drawEdges=False
                    )
                    mesh_item.user_data = {'type': 'model', 'path': file_path, 'name': geometry.metadata['name']}
                    
                    # Add to view
                    self.view3d.addItem(mesh_item)
                    self.view3d.addGeometry(geometry, mesh_item)
                    self.structural_elements.append(geometry)
                
                # Reset camera
                self.view3d.setCameraPosition(distance=10)
                
                # Refresh the geometries list
                self.refresh_geometries_list()
                
                return True
            else:
                # Single geometry file
                # Set metadata
                mesh.metadata = {
                    'name': filename,
                    'path': file_path,
                    'index': 0
                }
                
                # Get the bounds of the mesh to calculate both height and center
                bounds = mesh.bounds
                min_bounds, max_bounds = bounds

                # Calculate center of the mesh (in all dimensions)
                center = (min_bounds + max_bounds) / 2.0

                # Create a transformation matrix to center and place on grid
                translation = np.eye(4)
                translation[0, 3] = -center[0]  # Center the model in x
                translation[1, 3] = -center[1]  # Center the model in y
                translation[2, 3] = -min_bounds[2]  # Move the bottom of the model to z=0 (grid level)

                # Apply the transformation
                mesh.apply_transform(translation)

                # Create GL mesh for display
                vertices = mesh.vertices
                faces = mesh.faces
                face_colors = np.array([[0.7, 0.7, 1.0, 0.2] for _ in range(len(faces))])

                # Create the mesh item
                mesh_item = gl.GLMeshItem(
                    vertexes=vertices,
                    faces=faces,
                    faceColors=face_colors,
                    smooth=True,
                    drawEdges=False
                )
                mesh_item.user_data = {'type': 'model', 'path': file_path, 'name': mesh.metadata['name']}

                # Add the mesh to the view
                self.view3d.addItem(mesh_item)
                self.view3d.addGeometry(mesh, mesh_item)
                self.structural_elements.append(mesh)

                # Reset the camera to frame the model
                self.view3d.setCameraPosition(distance=max(mesh.extents)*2)

                # Refresh the geometries list
                self.refresh_geometries_list()

                # Print some useful debug info
                print(f"Model loaded and centered: {os.path.basename(file_path)}")
                print(f"Original bounds: {bounds}")
                print(f"Original center: {center}")
                print(f"New bounds: {mesh.bounds}")

                return True
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            return False

    def delete_model(self):
        """Delete the selected model file"""
        selected_items = self.models_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a model to delete")
            return
            
        filename = selected_items[0].text()
        file_path = os.path.join(self.models_dir, filename)
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete {filename}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Remove from view if it's the current model
                if self.mesh and hasattr(self.mesh, 'metadata') and self.mesh.metadata.get('file_path') == file_path:
                    for item in self.view3d.items:
                        if isinstance(item, gl.GLMeshItem) and hasattr(item, 'user_data') and item.user_data.get('path') == file_path:
                            self.view3d.removeItem(item)
                    self.mesh = None
                    self.view3d.clearGeometries()
                    self.selected_geometry = None
                
                # Delete the file
                os.remove(file_path)
                
                # Refresh the models list
                self.refresh_models_list()
                
                # Update selection status
                self.selection_status.setText(f"Deleted: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete model: {str(e)}")
    
    def refresh_models_list(self):
        """Refresh the list of available models"""
        self.models_list.clear()
        
        # Get all model files
        model_files = []
        for ext in ['.obj', '.stl', '.ply', '.glb', '.gltf']:
            model_files.extend(glob.glob(os.path.join(self.models_dir, f"*{ext}")))
        
        # Add to list
        if model_files:
            for file_path in model_files:
                self.models_list.addItem(os.path.basename(file_path))
        else:
            self.models_list.addItem("No models found (upload one)")
    
    def toggle_model_visibility(self):
        """Toggle the visibility of the current model"""
        # Find all model mesh items
        for item in self.view3d.items:
            if isinstance(item, gl.GLMeshItem) and hasattr(item, 'user_data') and item.user_data.get('type') == 'model':
                if self.model_visible:
                    # Hide the model
                    item.hide()
                    self.visibility_btn.setText("Show Model")
                else:
                    # Show the model
                    item.show()
                    self.visibility_btn.setText("Hide Model")
        
        # Update tracking variable
        self.model_visible = not self.model_visible
    
    def on_geometry_selected(self, geometry):
        """Handle selection of a geometry from the 3D view"""
        # Store the selected geometry
        self.selected_geometry = geometry

        # Update the selection status
        self.selection_status.setText("Geometry selected - Ready to apply structural system")

        # Reset all geometries to default appearance
        for geom in self.view3d.geometries:
            if geom['mesh_item'] is not None:
                mesh_item = geom['mesh_item']
                mesh_item.setColor((0.7, 0.7, 1.0, 0.2))  # Default semi-transparent blue

        # Highlight the selected geometry
        if geometry is not None and geometry['mesh_item'] is not None:
            mesh_item = geometry['mesh_item']
            mesh_item.setColor((1.0, 0.5, 0.0, 0.7))  # Bright orange for selection

        # Ensure the selected geometry is visually distinct
        self.view3d.update()

    def on_geometry_list_selected(self, item):
        """Handle selection of a geometry from the list"""
        geometry_name = item.text()
        for geom in self.view3d.geometries:
            if 'data' in geom and hasattr(geom['data'], 'metadata') and geom['data'].metadata.get('name') == geometry_name:
                self.on_geometry_selected(geom)
                self.selection_status.setText(f"Selected: {geometry_name}")
                break

    def refresh_geometries_list(self):
        """Refresh the list of available geometries"""
        self.geometries_list.clear()
        for geom in self.view3d.geometries:
            if 'data' in geom and hasattr(geom['data'], 'metadata'):
                self.geometries_list.addItem(geom['data'].metadata.get('name', 'Unnamed Geometry'))

    def on_model_selected(self, item):
        """Handle selection of a model from the list"""
        filename = item.text()
        
        # Skip if the item is just an informational message
        if filename.startswith("No models found"):
            return
            
        file_path = os.path.join(self.models_dir, filename)
        
        if os.path.exists(file_path):
            # Update selection status in UI
            self.selection_status.setText(f"Loading model: {filename}")
            
            # Clear previous selection state
            self.view3d.clearGeometries()
            self.selected_geometry = None
            
            # Load the selected model
            self.load_model(file_path)
            
            # Update the button to indicate that this model can be made visible/invisible
            self.visibility_btn.setEnabled(True)
            self.visibility_btn.setText("Hide Model")
            self.model_visible = True
            
            # Update selection status with model info
            self.selection_status.setText(f"Loaded: {filename}")
            
            # Highlight the selected item in the list
            for i in range(self.models_list.count()):
                if self.models_list.item(i).text() == filename:
                    self.models_list.item(i).setSelected(True)
                else:
                    self.models_list.item(i).setSelected(False)

    def update_system_parameters(self):
        """Update dynamic parameters based on selected system type"""
        system_type = self.system_type.currentText()
        
        # Clear previous dynamic parameters
        for widget in self.dynamic_params_widgets.values():
            if widget.parent():
                self.params_layout.removeRow(widget)
        
        self.dynamic_params_widgets = {}
        
        # Add parameters specific to the selected system type
        if system_type == "Grid":
            # Grid spacing parameter
            grid_spacing = QSpinBox()
            grid_spacing.setRange(1, 50)
            grid_spacing.setValue(10)
            self.params_layout.addRow("Grid Spacing:", grid_spacing)
            self.dynamic_params_widgets["grid_spacing"] = grid_spacing
            
            # Grid orientation parameter
            orientation = QComboBox()
            orientation.addItems(["Horizontal", "Vertical", "Both"])
            orientation.setCurrentText("Both")
            self.params_layout.addRow("Orientation:", orientation)
            self.dynamic_params_widgets["orientation"] = orientation
            
            # Floor type parameter
            floor_type = QComboBox()
            floor_type.addItems(["Concrete Slab", "Composite Deck", "Timber", "Steel Deck"])
            floor_type.setCurrentText("Concrete Slab")
            self.params_layout.addRow("Floor Type:", floor_type)
            self.dynamic_params_widgets["floor_type"] = floor_type
            
        elif system_type == "Diagrid":
            # Angle parameter
            angle = QSpinBox()
            angle.setRange(30, 80)
            angle.setValue(45)
            self.params_layout.addRow("Angle (degrees):", angle)
            self.dynamic_params_widgets["angle"] = angle
            
            # Cell size parameter
            cell_size = QDoubleSpinBox()
            cell_size.setRange(0.1, 5.0)
            cell_size.setValue(1.0)
            cell_size.setSingleStep(0.1)
            self.params_layout.addRow("Cell Size:", cell_size)
            self.dynamic_params_widgets["cell_size"] = cell_size
            
            # Floor type parameter
            floor_type = QComboBox()
            floor_type.addItems(["Concrete Slab", "Composite Deck", "Timber", "Steel Deck"])
            floor_type.setCurrentText("Concrete Slab")
            self.params_layout.addRow("Floor Type:", floor_type)
            self.dynamic_params_widgets["floor_type"] = floor_type
            
        elif system_type == "Space Frame":
            # Layer count parameter
            layers = QSpinBox()
            layers.setRange(1, 5)
            layers.setValue(2)
            self.params_layout.addRow("Layers:", layers)
            self.dynamic_params_widgets["layers"] = layers
            
            # Node connection type
            node_type = QComboBox()
            node_type.addItems(["Ball Joint", "Rigid Connection"])
            self.params_layout.addRow("Node Type:", node_type)
            self.dynamic_params_widgets["node_type"] = node_type
            
            # Floor type parameter
            floor_type = QComboBox()
            floor_type.addItems(["Concrete Slab", "Composite Deck", "Timber", "Steel Deck"])
            floor_type.setCurrentText("Concrete Slab")
            self.params_layout.addRow("Floor Type:", floor_type)
            self.dynamic_params_widgets["floor_type"] = floor_type
            
        elif system_type == "Voronoi":
            # Seed count parameter
            seeds = QSpinBox()
            seeds.setRange(5, 100)
            seeds.setValue(20)
            self.params_layout.addRow("Seeds:", seeds)
            self.dynamic_params_widgets["seeds"] = seeds
            
            # Relaxation parameter
            relaxation = QSpinBox()
            relaxation.setRange(0, 10)
            relaxation.setValue(3)
            self.params_layout.addRow("Relaxation:", relaxation)
            self.dynamic_params_widgets["relaxation"] = relaxation
            
            # Floor type parameter
            floor_type = QComboBox()
            floor_type.addItems(["Concrete Slab", "Composite Deck", "Timber", "Steel Deck"])
            floor_type.setCurrentText("Concrete Slab")
            self.params_layout.addRow("Floor Type:", floor_type)
            self.dynamic_params_widgets["floor_type"] = floor_type
            
        elif system_type == "Triangulated":
            # Minimum angle parameter
            min_angle = QSpinBox()
            min_angle.setRange(15, 45)
            min_angle.setValue(30)
            self.params_layout.addRow("Min Angle:", min_angle)
            self.dynamic_params_widgets["min_angle"] = min_angle
            
            # Refinement level
            refinement = QSpinBox()
            refinement.setRange(0, 3)
            refinement.setValue(1)
            self.params_layout.addRow("Refinement:", refinement)
            self.dynamic_params_widgets["refinement"] = refinement
            
            # Floor type parameter
            floor_type = QComboBox()
            floor_type.addItems(["Concrete Slab", "Composite Deck", "Timber", "Steel Deck"])
            floor_type.setCurrentText("Concrete Slab")
            self.params_layout.addRow("Floor Type:", floor_type)
            self.dynamic_params_widgets["floor_type"] = floor_type
            
        elif system_type == "Waffle":
            # Slot depth parameter
            slot_depth = QDoubleSpinBox()
            slot_depth.setRange(0.1, 1.0)
            slot_depth.setValue(0.5)
            slot_depth.setSingleStep(0.05)
            self.params_layout.addRow("Slot Depth:", slot_depth)
            self.dynamic_params_widgets["slot_depth"] = slot_depth
            
            # Slot width parameter
            slot_width = QDoubleSpinBox()
            slot_width.setRange(0.05, 0.5)
            slot_width.setValue(0.1)
            slot_width.setSingleStep(0.01)
            self.params_layout.addRow("Slot Width:", slot_width)
            self.dynamic_params_widgets["slot_width"] = slot_width
            
            # Floor type parameter
            floor_type = QComboBox()
            floor_type.addItems(["Concrete Slab", "Composite Deck", "Timber", "Steel Deck"])
            floor_type.setCurrentText("Concrete Slab")
            self.params_layout.addRow("Floor Type:", floor_type)
            self.dynamic_params_widgets["floor_type"] = floor_type
    
    def select_color(self):
        """Open color picker dialog for structural elements"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.element_color = [color.red(), color.green(), color.blue()]
            # Update button color
            self.color_btn.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})")
    
    def enter_selection_mode(self):
        """Toggle selection mode on/off"""
        self.selection_active = not self.selection_active
        
        # Enable selection in the 3D viewer
        self.view3d.enableSelectionMode(self.selection_active)
        self.view3d.setSelectionType("Geometry")  # Always use Geometry selection
        
        # Update button text and status
        btn_text = "Exit Selection Mode" if self.selection_active else "Enter Selection Mode"
        status_text = "Selection Mode: Active" if self.selection_active else "Selection Mode: Inactive"
        
        # Find the button and update its text
        for child in self.findChildren(QPushButton):
            if child.text() in ["Enter Selection Mode", "Exit Selection Mode"]:
                child.setText(btn_text)
                break
                
        # Update status label
        self.selection_status.setText(status_text)
    
    def clear_selection(self):
        """Clear the current selection"""
        # Reset selection tracking
        self.selected_geometry = None
        
        # Reset appearance for all geometries to default
        for geom in self.view3d.geometries:
            if geom['mesh_item'] is not None:
                geom['mesh_item'].setColor((0.7, 0.7, 1.0, 0.2))  # Default semi-transparent blue
        
        # Update status
        self.selection_status.setText("Selection cleared")
        
        # Deselect any selected items in the geometries list
        self.geometries_list.clearSelection()
        
    def apply_system(self):
        """Apply the selected structural system to the current geometry"""
        if not self.selected_geometry:
            # If no selection, check if there's an item selected in the list
            selected_items = self.geometries_list.selectedItems()
            if selected_items:
                geometry_name = selected_items[0].text()
                # Find the corresponding geometry
                for geom in self.view3d.geometries:
                    if 'data' in geom and hasattr(geom['data'], 'metadata') and geom['data'].metadata.get('name') == geometry_name:
                        self.selected_geometry = geom
                        break
            else:
                QMessageBox.warning(self, "Warning", "Please select a geometry first")
                return

        system_type = self.system_type.currentText()

        # Get common parameters
        density = self.density_slider.value() / 100.0  # Normalize to 0-1
        depth = self.depth_spin.value()
        color = self.element_color
        
        # Get floor type 
        floor_type = "Concrete Slab"  # Default
        if "floor_type" in self.dynamic_params_widgets:
            floor_type = self.dynamic_params_widgets["floor_type"].currentText()

        # Create a structural system
        system_info = {
            'type': system_type,
            'density': density,
            'depth': depth,
            'color': color,
            'floor_type': floor_type,
            'parameters': {},
            'mesh_items': []
        }

        # Add system-specific parameters
        for param_name, widget in self.dynamic_params_widgets.items():
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                system_info['parameters'][param_name] = widget.value()
            elif isinstance(widget, QComboBox):
                system_info['parameters'][param_name] = widget.currentText()

        # Display a message about the system application
        geometry_name = "Unknown"
        if self.selected_geometry and 'data' in self.selected_geometry and hasattr(self.selected_geometry['data'], 'metadata'):
            geometry_name = self.selected_geometry['data'].metadata.get('name', "Unknown")
            
        QMessageBox.information(self, "System Applied", 
                              f"Applied {system_type} system with density {density:.2f}, depth {depth}, and floor type {floor_type} to {geometry_name}.\n"
                              f"Note: This is a placeholder. Actual system generation to be implemented.")

        # Add to systems list
        system_name = f"{system_type} System on {geometry_name} #{len(self.structural_systems) + 1}"
        self.systems_list.addItem(system_name)
        system_info['name'] = system_name
        self.structural_systems.append(system_info)
    
    def delete_system(self):
        """Delete the selected structural system"""
        selected_items = self.systems_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a system to delete")
            return
            
        system_name = selected_items[0].text()
        
        # Find and remove the system
        for i, system in enumerate(self.structural_systems):
            if system['name'] == system_name:
                # Remove mesh items from view
                for mesh_item in system['mesh_items']:
                    self.view3d.removeItem(mesh_item)
                
                # Remove from list
                self.structural_systems.pop(i)
                break
        
        # Update the UI
        self.systems_list.takeItem(self.systems_list.row(selected_items[0]))
    
    def clear_all_systems(self):
        """Clear all structural systems"""
        confirm = QMessageBox.question(
            self, "Confirm Clear", 
            "Are you sure you want to clear all structural systems?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Remove all mesh items
            for system in self.structural_systems:
                for mesh_item in system['mesh_items']:
                    self.view3d.removeItem(mesh_item)
            
            # Clear the list
            self.structural_systems = []
            self.systems_list.clear()
    
    def export_structure(self):
        """Export the current structural systems to a file"""
        if not self.structural_systems:
            QMessageBox.warning(self, "Warning", "No structural systems to export")
            return
            
        # Create a file dialog
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("OBJ Files (*.obj);;STL Files (*.stl)")
        file_dialog.setDefaultSuffix("obj")
        
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                # Display "Not Implemented" message for now
                QMessageBox.information(self, "Export", 
                                      f"Export functionality not yet implemented.\n"
                                      f"Would save to: {file_paths[0]}")

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