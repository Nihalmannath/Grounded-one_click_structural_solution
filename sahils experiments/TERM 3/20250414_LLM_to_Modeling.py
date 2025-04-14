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
        
    def paintGL(self):
        """Override paintGL to draw selection box"""
        super().paintGL()
        
        # Draw selection box if dragging
        if self.selection_mode and self.is_dragging and self.drag_start_pos and self.current_drag_pos:
            ogl.glMatrixMode(ogl.GL_PROJECTION)
            ogl.glPushMatrix()
            ogl.glLoadIdentity()
            viewport = ogl.glGetIntegerv(ogl.GL_VIEWPORT)
            ogl.glOrtho(0, viewport[2], 0, viewport[3], -1, 1)
            
            ogl.glMatrixMode(ogl.GL_MODELVIEW)
            ogl.glPushMatrix()
            ogl.glLoadIdentity()
            
            # Draw selection rectangle
            ogl.glColor4f(0.2, 0.7, 1.0, 0.3)  # Semi-transparent blue
            ogl.glPolygonMode(ogl.GL_FRONT_AND_BACK, ogl.GL_FILL)
            ogl.glBegin(ogl.GL_QUADS)
            # Convert to OpenGL coordinates (origin at bottom-left)
            x1, y1 = self.drag_start_pos.x(), viewport[3] - self.drag_start_pos.y()
            x2, y2 = self.current_drag_pos.x(), viewport[3] - self.current_drag_pos.y()
            ogl.glVertex2f(x1, y1)
            ogl.glVertex2f(x2, y1)
            ogl.glVertex2f(x2, y2)
            ogl.glVertex2f(x1, y2)
            ogl.glEnd()
            
            # Draw selection rectangle border
            ogl.glColor4f(0.2, 0.7, 1.0, 0.7)  # Less transparent blue
            ogl.glLineWidth(2.0)
            ogl.glPolygonMode(ogl.GL_FRONT_AND_BACK, ogl.GL_LINE)
            ogl.glBegin(ogl.GL_LINE_LOOP)
            ogl.glVertex2f(x1, y1)
            ogl.glVertex2f(x2, y1)
            ogl.glVertex2f(x2, y2)
            ogl.glVertex2f(x1, y2)
            ogl.glEnd()
            
            ogl.glPopMatrix()
            ogl.glMatrixMode(ogl.GL_PROJECTION)
            ogl.glPopMatrix()
        
    def mousePressEvent(self, ev):
        # Middle mouse button for camera rotation/movement
        if ev.button() == Qt.MiddleButton:
            super().mousePressEvent(ev)
            return
            
        # Left button for selection
        if self.selection_mode and ev.button() == Qt.LeftButton:
            # Get mouse position
            pos = ev.pos()
            
            if self.selection_type == "Face" and self.mesh_data is not None:
                # Start drag selection box for faces
                self.is_dragging = True
                self.drag_start_pos = pos
                self.current_drag_pos = pos
                self.faces_under_drag = []
                
                # Prevent default handling
                ev.accept()
                self.update()  # Trigger repaint for selection box
                return
            elif self.selection_type == "Geometry" and self.geometries:
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
            
            # Find closest geometry
            closest_geometry = None
            min_distance = float('inf')
            
            for i, geom in enumerate(self.geometries):
                # Check if the geometry has a mesh data to test intersection with
                if hasattr(geom['data'], 'faces') and hasattr(geom['data'], 'vertices'):
                    mesh = geom['data']
                    # Use ray-triangle intersection to check if the ray hits any face in the geometry
                    
                    # For efficiency, first check if ray intersects the bounding box
                    if hasattr(mesh, 'bounds'):
                        bbox_min, bbox_max = mesh.bounds
                        # Simple ray-box intersection test
                        if not self.ray_bbox_intersect(ray_origin, ray_direction, bbox_min, bbox_max):
                            continue
                    
                    faces = mesh.faces
                    vertices = mesh.vertices
                    
                    # Check for intersection with any face in the geometry
                    for face in faces:
                        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                        
                        # Möller–Trumbore ray-triangle intersection
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
                            closest_geometry = i
                            break  # Found an intersection with this geometry, no need to check more faces
                
            return closest_geometry
            
        except Exception as e:
            print(f"Geometry picking error: {e}")
            return None
    
    def ray_bbox_intersect(self, ray_origin, ray_direction, bbox_min, bbox_max):
        """Simple ray-box intersection test"""
        # Calculate inverse ray direction to avoid division
        inv_dir = 1.0 / (ray_direction + 1e-10)  # Avoid division by zero
        
        # Calculate intersection distances
        t1 = (bbox_min[0] - ray_origin[0]) * inv_dir[0]
        t2 = (bbox_max[0] - ray_origin[0]) * inv_dir[0]
        t3 = (bbox_min[1] - ray_origin[1]) * inv_dir[1]
        t4 = (bbox_max[1] - ray_origin[1]) * inv_dir[1]
        t5 = (bbox_min[2] - ray_origin[2]) * inv_dir[2]
        t6 = (bbox_max[2] - ray_origin[2]) * inv_dir[2]
        
        # Find the maximum entry point
        tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
        # Find the minimum exit point
        tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))
        
        # If tmax < 0, ray is intersecting AABB, but the whole AABB is behind us
        if tmax < 0:
            return False
            
        # If tmin > tmax, ray doesn't intersect AABB
        if tmin > tmax:
            return False
            
        return True
    
    def mouseMoveEvent(self, ev):
        # Update current position for drag box
        if self.selection_mode and self.is_dragging:
            self.current_drag_pos = ev.pos()
            self.update()  # Trigger repaint for selection box
            ev.accept()
            return
            
        # Hover behavior when not dragging
        if self.selection_mode:
            # Get mouse position
            pos = ev.pos()
            
            if self.selection_type == "Face" and self.mesh_data is not None:
                face_idx = self.pick_face(pos)
                
                # Only emit if the hover face has changed
                if face_idx != self.hover_face:
                    self.hover_face = face_idx
                    self.hoverFaceSignal.emit(face_idx if face_idx is not None else -1)
            elif self.selection_type == "Geometry" and self.geometries:
                geom_idx = self.pick_geometry(pos)
                
                # Only update if the hover geometry has changed
                if geom_idx != self.hover_geometry:
                    # Reset old hover highlight
                    if self.hover_geometry is not None and self.hover_geometry < len(self.geometries):
                        old_geom = self.geometries[self.hover_geometry]
                        if 'mesh_item' in old_geom and old_geom['mesh_item'] is not None:
                            # Reset color if it's not selected
                            if self.hover_geometry != self.selected_geometry:
                                old_geom['mesh_item'].setColor(pg_color((180, 180, 255), 0.5))
                    
                    # Set new hover highlight
                    self.hover_geometry = geom_idx
                    if geom_idx is not None and geom_idx < len(self.geometries):
                        geom = self.geometries[geom_idx]
                        if 'mesh_item' in geom and geom['mesh_item'] is not None:
                            # Set hover color (yellow) if it's not selected
                            if geom_idx != self.selected_geometry:
                                geom['mesh_item'].setColor(pg_color((255, 255, 0), 0.8))
        
        # Default behavior for camera control
        super().mouseMoveEvent(ev)
    
    def mouseReleaseEvent(self, ev):
        # Handle selection box completion
        if self.selection_mode and self.is_dragging and ev.button() == Qt.LeftButton:
            # Only use the selection box for face selection
            if self.selection_type == "Face":
                # Calculate which faces are in the selection box
                self.faces_under_drag = self.get_faces_in_selection_box(
                    self.drag_start_pos, self.current_drag_pos)
                
                # Emit the selected faces
                if self.faces_under_drag:
                    self.dragSelectionSignal.emit(self.faces_under_drag)
            
            # Reset drag state
            self.is_dragging = False
            self.update()  # Trigger repaint to remove selection box
            ev.accept()
            return
            
        # Default behavior
        super().mouseReleaseEvent(ev)

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
        
        # Selection group - modified to only have geometry selection
        selection_group = QGroupBox("Selection")
        selection_layout = QVBoxLayout()
        
        # Remove selection type dropdown, always use Geometry
        select_btn = QPushButton("Enter Selection Mode")
        select_btn.clicked.connect(self.enter_selection_mode)
        selection_layout.addWidget(select_btn)
        
        # Update hover hint for geometry selection
        self.hover_hint = QLabel("Hover over geometry to highlight it for selection")
        self.hover_hint.setStyleSheet("color: blue;")
        selection_layout.addWidget(self.hover_hint)
        
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
                
                # Display the mesh with improved visibility
                vertices = self.mesh.vertices
                faces = self.mesh.faces
                
                # Make the face colors more transparent with lower opacity (0.1)
                face_colors = np.array([[0.7, 0.7, 1.0, 0.1] for _ in range(len(faces))])
                
                # Create the mesh item with smooth shading and no edges for cleaner visualization
                mesh_item = gl.GLMeshItem(
                    vertexes=vertices, 
                    faces=faces, 
                    faceColors=face_colors,
                    smooth=True,
                    drawEdges=False,
                    shader='shaded'  # Use shaded shader for better lighting
                )
                self.view3d.addItem(mesh_item)
                self.mesh_item = mesh_item
                
                # Set mesh data for picking
                self.view3d.setMeshData(self.mesh)
                
                # Add geometry for geometry picking
                self.view3d.addGeometry(self.mesh, mesh_item)
                
                # Find coplanar surface groups
                self.coplanar_groups = self.find_coplanar_surfaces(self.mesh)
                print(f"Found {len(self.coplanar_groups)} coplanar surface groups")
                
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
        
        # Display the mesh with higher opacity for better visibility
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        # Make the face colors more transparent and don't show edges
        face_colors = np.array([[0.7, 0.7, 1.0, 0.2] for _ in range(len(faces))])
        
        # Create the mesh item - with no edges
        mesh_item = gl.GLMeshItem(
            vertexes=vertices, 
            faces=faces, 
            faceColors=face_colors,
            smooth=True,
            drawEdges=False
        )
        self.view3d.addItem(mesh_item)
        self.mesh_item = mesh_item
        self.view3d.setMeshData(self.mesh)
        
        # Add geometry for geometry picking
        self.view3d.addGeometry(self.mesh, mesh_item)
        
        QMessageBox.information(self, "Default Model", 
                               "Loaded a default cube mesh for demonstration.")

    def toggle_model_visibility(self):
        """Toggle the visibility of the model"""
        if not hasattr(self, 'mesh_item') or self.mesh_item is None:
            return
            
        self.model_visible = not self.model_visible
        
        if self.model_visible:
            self.mesh_item.setVisible(True)
            self.visibility_btn.setText("Hide Model")
        else:
            self.mesh_item.setVisible(False)
            self.visibility_btn.setText("Show Model")

    def apply_system(self):
        """Apply the selected structural system to the selected faces or geometry"""
        # Check if we have a selection based on the current selection type
        if not self.selected_geometry:
            QMessageBox.warning(self, "Selection Required", 
                               "Please select a geometry to apply the structural system.")
            return
            
        system_type = self.system_type.currentText()
        
        # Get common parameters
        density = self.density_slider.value()
        depth = self.depth_spin.value()
        color = self.element_color
        
        # Collection for new elements
        new_elements = []
        
        # Apply system based on geometry selection
        if self.selected_geometry and hasattr(self.selected_geometry['data'], 'faces'):
            mesh = self.selected_geometry['data']
            
            # Use coplanar surfaces rather than individual faces
            if hasattr(self, 'coplanar_groups') and self.coplanar_groups:
                # Process each coplanar surface group
                for group in self.coplanar_groups:
                    # Get all face vertices for this coplanar group
                    coplanar_vertices = []
                    for face_idx in group:
                        if face_idx < len(mesh.faces):
                            face = mesh.faces[face_idx]
                            for vertex_idx in face:
                                coplanar_vertices.append(mesh.vertices[vertex_idx])
                    
                    # Remove duplicates while preserving order
                    unique_vertices = []
                    seen = set()
                    for vertex in coplanar_vertices:
                        vertex_tuple = tuple(vertex)
                        if vertex_tuple not in seen:
                            seen.add(vertex_tuple)
                            unique_vertices.append(vertex)
                    
                    # Skip if too few vertices
                    if len(unique_vertices) < 3:
                        continue
                    
                    # Convert back to numpy array
                    vertices = np.array(unique_vertices)
                    
                    # Generate elements based on the selected system type for this coplanar group
                    elements = self.generate_system_for_face(system_type, vertices, density, depth)
                    if elements:
                        new_elements.extend(elements)
            else:
                # Fallback to processing individual faces
                for face_idx, face in enumerate(mesh.faces):
                    vertices = mesh.vertices[face]
                    elements = self.generate_system_for_face(system_type, vertices, density, depth)
                    if elements:
                        new_elements.extend(elements)
        
        # Add new elements to the scene
        for element in new_elements:
            self.view3d.addItem(element)
            
        # Create a system object to track this specific structural system
        system_info = {
            'type': system_type,
            'elements': new_elements,
            'selection_type': "Geometry",
            'selection': "Geometry",
            'parameters': {
                'density': density,
                'depth': depth,
                'color': color.copy(),
                # Add any system-specific parameters
            }
        }
        
        # Store the system
        self.structural_systems.append(system_info)
        self.structural_elements.extend(new_elements)
        
        # Update the systems list
        self._update_systems_list()
        
        QMessageBox.information(self, "System Applied", 
                               f"{system_type} system applied to selected geometry.")

    def delete_system(self):
        """Delete the selected structural system from the view"""
        selected_items = self.systems_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No System Selected", 
                               "Please select a system to delete.")
            return
            
        selected_idx = self.systems_list.row(selected_items[0])
        
        if selected_idx >= 0 and selected_idx < len(self.structural_systems):
            # Get the system to delete
            system = self.structural_systems[selected_idx]
            
            # Remove elements from view
            for element in system['elements']:
                self.view3d.removeItem(element)
                
                # Also remove from structural_elements list
                if element in self.structural_elements:
                    self.structural_elements.remove(element)
            
            # Remove the system from the list
            self.structural_systems.pop(selected_idx)
            
            # Update the systems list
            self._update_systems_list()
            
            QMessageBox.information(self, "System Deleted", 
                                   f"{system['type']} system has been deleted.")

    def clear_all_systems(self):
        """Clear all structural systems from the view"""
        if not self.structural_systems:
            QMessageBox.information(self, "No Systems", 
                                  "There are no structural systems to clear.")
            return
            
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            "Are you sure you want to delete all structural systems?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove all elements from view
            for system in self.structural_systems:
                for element in system['elements']:
                    self.view3d.removeItem(element)
            
            # Clear the lists
            self.structural_systems = []
            self.structural_elements = []
            
            # Update the systems list
            self._update_systems_list()
            
            QMessageBox.information(self, "Systems Cleared", 
                                   "All structural systems have been cleared.")

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
        self.structural_elements = []
        self.structural_systems = []  # Clear systems as well
        self.selection_active = False
        self.selection_status.setText("Selection Mode: Inactive")
        self.model_visible = True
        
        # Clear geometry selection data
        self.view3d.clearGeometries()
        self.selected_geometry = None
        
        # Update the systems list
        self._update_systems_list()

    def _update_systems_list(self):
        """Update the list of structural systems in the UI"""
        self.systems_list.clear()
        
        for i, system in enumerate(self.structural_systems):
            # Create descriptive text for the system
            item_text = f"{system['type']} (on geometry)"
            self.systems_list.addItem(item_text)

    def refresh_models_list(self):
        """Refresh the list of available models"""
        self.models_list.clear()
        
        # List all model files in the models directory
        if os.path.exists(self.models_dir):
            # Get list of files with the right extensions
            model_files = []
            for file in os.listdir(self.models_dir):
                if file.endswith(('.obj', '.stl', '.ply')):
                    model_files.append(file)
            
            # Sort models alphabetically for better organization
            model_files.sort()
            
            # Add models to the list with improved formatting
            for file in model_files:
                self.models_list.addItem(file)
            
            # Add style to make the list more visible
            self.models_list.setStyleSheet("""
                QListWidget {
                    background-color: #f0f0f0;
                    border: 1px solid #999;
                    border-radius: 3px;
                }
                QListWidget::item {
                    padding: 5px;
                    border-bottom: 1px solid #ddd;
                }
                QListWidget::item:selected {
                    background-color: #3498db;
                    color: white;
                }
                QListWidget::item:hover {
                    background-color: #e0e0e0;
                }
            """)
            
            # If there are no models, add an informative message
            if len(model_files) == 0:
                self.models_list.addItem("No models found. Upload a model to begin.")
    
    def upload_model(self):
        """Upload a 3D model file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open 3D Model", "", "3D Models (*.obj *.stl *.ply)")
        
        if file_path:
            # Copy the file to the models directory
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.models_dir, filename)
            
            try:
                # Check if file already exists
                if os.path.exists(dest_path):
                    reply = QMessageBox.question(
                        self, "File Exists",
                        f"The file {filename} already exists. Overwrite?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                    )
                    
                    if reply == QMessageBox.No:
                        return
                
                # Copy the file
                import shutil
                shutil.copy2(file_path, dest_path)
                
                # Refresh the models list
                self.refresh_models_list()
                
                # Load the new model
                self.load_model(dest_path)
                
                QMessageBox.information(self, "Upload Successful", 
                                      f"Model {filename} has been uploaded.")
            except Exception as e:
                QMessageBox.critical(self, "Upload Failed", 
                                    f"Failed to upload model: {str(e)}")
    
    def delete_model(self):
        """Delete the selected model file"""
        selected_items = self.models_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Model Selected", 
                               "Please select a model to delete.")
            return
            
        filename = selected_items[0].text()
        file_path = os.path.join(self.models_dir, filename)
        
        if os.path.exists(file_path):
            reply = QMessageBox.question(
                self, "Confirm Deletion",
                f"Are you sure you want to delete {filename}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    os.remove(file_path)
                    
                    # Refresh the models list
                    self.refresh_models_list()
                    
                    # Clear view if the deleted model was loaded
                    if self.mesh is not None:
                        self.clear_view()
                    
                    QMessageBox.information(self, "Deletion Successful", 
                                          f"Model {filename} has been deleted.")
                except Exception as e:
                    QMessageBox.critical(self, "Deletion Failed", 
                                        f"Failed to delete model: {str(e)}")
    
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
    
    def enter_selection_mode(self):
        """Enter or exit selection mode for picking geometries"""
        # Toggle selection mode
        self.selection_active = not self.selection_active
        
        # Update selection status
        if self.selection_active:
            self.selection_status.setText("Selection Mode: Active")
            self.view3d.enableSelectionMode(True)
            self.view3d.setSelectionType("Geometry")  # Always use Geometry selection type
            
            # Make sure model is visible with proper opacity
            if hasattr(self, 'mesh_item') and self.mesh_item is not None and self.mesh is not None:
                vertices = self.mesh.vertices
                faces = self.mesh.faces
                
                # Create face colors with proper visibility
                face_colors = np.array([[0.7, 0.7, 1.0, 0.4] for _ in range(len(faces))])
                
                self.mesh_item.setMeshData(
                    vertexes=vertices,
                    faces=faces,
                    faceColors=face_colors,
                    smooth=True,
                    drawEdges=False
                )
                
                # Re-add the geometry for selection
                self.view3d.clearGeometries()
                self.view3d.addGeometry(self.mesh, self.mesh_item)
        else:
            self.selection_status.setText("Selection Mode: Inactive")
            self.view3d.enableSelectionMode(False)
            
            # Reset model to normal visibility
            if hasattr(self, 'mesh_item') and self.mesh_item is not None and self.mesh is not None:
                vertices = self.mesh.vertices
                faces = self.mesh.faces
                
                # Reset to default colors
                face_colors = np.array([[0.7, 0.7, 1.0, 0.2] for _ in range(len(faces))])
                
                self.mesh_item.setMeshData(
                    vertexes=vertices,
                    faces=faces,
                    faceColors=face_colors,
                    smooth=True,
                    drawEdges=False
                )
    
    def clear_selection(self):
        """Clear the current selection"""
        # Clear geometry selection data
        self.view3d.clearGeometries()
        self.selected_geometry = None
        
        # Update selection status
        self.selection_status.setText("Selection Mode: Active - No geometry selected")
    
    def update_system_parameters(self):
        """Update UI parameters based on selected system type"""
        # Clear current system-specific parameters
        for widget in self.dynamic_params_widgets.values():
            self.params_layout.removeRow(widget)
        self.dynamic_params_widgets = {}
        
        # Get the selected system type
        system_type = self.system_type.currentText()
        
        # Add system-specific parameters
        if system_type == "Grid":
            # Add grid spacing parameter
            grid_spacing = QDoubleSpinBox()
            grid_spacing.setRange(0.1, 2.0)
            grid_spacing.setValue(0.5)
            grid_spacing.setSingleStep(0.1)
            self.params_layout.addRow("Grid Spacing:", grid_spacing)
            self.dynamic_params_widgets["grid_spacing"] = grid_spacing
            
            # Add grid pattern parameter
            grid_pattern = QComboBox()
            grid_pattern.addItems(["Square", "Rectangular", "Triangular"])
            self.params_layout.addRow("Grid Pattern:", grid_pattern)
            self.dynamic_params_widgets["grid_pattern"] = grid_pattern
            
            # Add number of floors parameter
            num_floors = QSpinBox()
            num_floors.setRange(1, 20)
            num_floors.setValue(1)
            num_floors.setSingleStep(1)
            self.params_layout.addRow("Number of Floors:", num_floors)
            self.dynamic_params_widgets["num_floors"] = num_floors
            
            # Add floor height parameter
            floor_height = QDoubleSpinBox()
            floor_height.setRange(0.5, 5.0)
            floor_height.setValue(3.0)
            floor_height.setSingleStep(0.5)
            self.params_layout.addRow("Floor Height (m):", floor_height)
            self.dynamic_params_widgets["floor_height"] = floor_height
            
            # Add maximum span parameter
            max_span = QDoubleSpinBox()
            max_span.setRange(1.0, 10.0)
            max_span.setValue(6.0)
            max_span.setSingleStep(0.5)
            self.params_layout.addRow("Maximum Span (m):", max_span)
            self.dynamic_params_widgets["max_span"] = max_span
            
        elif system_type == "Diagrid":
            # Add angle parameter
            angle_spin = QDoubleSpinBox()
            angle_spin.setRange(30, 80)
            angle_spin.setValue(60)
            angle_spin.setSingleStep(5)
            self.params_layout.addRow("Diagrid Angle (°):", angle_spin)
            self.dynamic_params_widgets["angle"] = angle_spin
            
        elif system_type == "Space Frame":
            # Add layer parameter
            layer_spin = QSpinBox()
            layer_spin.setRange(1, 3)
            layer_spin.setValue(1)
            self.params_layout.addRow("Layers:", layer_spin)
            self.dynamic_params_widgets["layers"] = layer_spin
            
        elif system_type == "Voronoi":
            # Add point count parameter
            point_spin = QSpinBox()
            point_spin.setRange(5, 50)
            point_spin.setValue(10)
            self.params_layout.addRow("Point Count:", point_spin)
            self.dynamic_params_widgets["points"] = point_spin
            
        elif system_type == "Triangulated":
            # Add subdivision parameter
            subdiv_spin = QSpinBox()
            subdiv_spin.setRange(1, 5)
            subdiv_spin.setValue(2)
            self.params_layout.addRow("Subdivisions:", subdiv_spin)
            self.dynamic_params_widgets["subdivisions"] = subdiv_spin
            
        elif system_type == "Waffle":
            # Add spacing parameter
            spacing_spin = QDoubleSpinBox()
            spacing_spin.setRange(0.1, 1.0)
            spacing_spin.setValue(0.3)
            spacing_spin.setSingleStep(0.1)
            self.params_layout.addRow("Spacing:", spacing_spin)
            self.dynamic_params_widgets["spacing"] = spacing_spin
            
            # Add direction parameter
            direction = QComboBox()
            direction.addItems(["X", "Y", "Both"])
            self.params_layout.addRow("Direction:", direction)
            self.dynamic_params_widgets["direction"] = direction
    
    def select_color(self):
        """Open color picker dialog for structural elements"""
        color = QColorDialog.getColor()
        
        if color.isValid():
            # Store the selected color
            self.element_color = [color.red(), color.green(), color.blue()]
            
            # Update the button background color
            self.color_btn.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})")
    
    def generate_system_for_face(self, system_type, vertices, density, depth, **kwargs):
        """Generate structural elements for a face based on the selected system type
        
        Args:
            system_type: The type of structural system to generate
            vertices: Face vertices as a numpy array
            density: Density parameter value
            depth: Depth parameter value
            **kwargs: System-specific parameters
            
        Returns:
            List of GL mesh items representing structural elements
        """
        if len(vertices) < 3:
            return []
            
        # Calculate face normal and center
        v0, v1, v2 = vertices[0], vertices[1], vertices[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0, 0, 1])
        center = np.mean(vertices, axis=0)
        
        # Get parameters
        color = self.element_color
        normalized_color = [c/255 for c in color] + [1.0]  # Add alpha=1.0
        
        # Create elements based on system type
        elements = []
        
        if system_type == "Grid":
            # Get system-specific parameters
            grid_spacing = 0.5
            max_span = 6.0
            num_floors = 1
            floor_height = 3.0
            
            if "grid_spacing" in self.dynamic_params_widgets:
                grid_spacing = self.dynamic_params_widgets["grid_spacing"].value()
            
            if "max_span" in self.dynamic_params_widgets:
                max_span = self.dynamic_params_widgets["max_span"].value()
                
            if "num_floors" in self.dynamic_params_widgets:
                num_floors = self.dynamic_params_widgets["num_floors"].value()
                
            if "floor_height" in self.dynamic_params_widgets:
                floor_height = self.dynamic_params_widgets["floor_height"].value()
            
            # Project vertices to 2D (horizontal plane)
            vertices_2d = vertices.copy()
            vertices_2d[:, 2] = 0  # Zero out z-component to get horizontal projection
            
            # Calculate the 2D bounding box of the projected vertices
            min_x = np.min(vertices_2d[:, 0])
            max_x = np.max(vertices_2d[:, 0])
            min_y = np.min(vertices_2d[:, 1])
            max_y = np.max(vertices_2d[:, 1])
            
            # Calculate grid dimensions based on max_span
            width = max_x - min_x
            height = max_y - min_y
            
            divisions_x = max(2, int(np.ceil(width / max_span)))
            divisions_y = max(2, int(np.ceil(height / max_span)))
            
            # Create grid points
            x_points = np.linspace(min_x, max_x, divisions_x + 1)
            y_points = np.linspace(min_y, max_y, divisions_y + 1)
            
            # Generate columns and beams for all floors
            for floor in range(num_floors):
                # Calculate floor elevation
                z_base = np.min(vertices[:, 2])  # Bottom of geometry
                floor_base = z_base + floor * floor_height
                floor_top = floor_base + floor_height
                
                # Create columns (vertical beams)
                for x in x_points:
                    for y in y_points:
                        # Check if point is within the horizontal projection of the face
                        if self._is_point_in_face_2d(np.array([x, y]), vertices_2d):
                            # Create a column from floor_base to floor_top
                            start = np.array([x, y, floor_base])
                            end = np.array([x, y, floor_top])
                            column = self._create_beam(start, end, depth/2, normalized_color, is_column=True)
                            elements.append(column)
                
                # Create beams at the top of this floor
                for i in range(len(x_points)):
                    for j in range(len(y_points) - 1):
                        x = x_points[i]
                        y1 = y_points[j]
                        y2 = y_points[j+1]
                        
                        # Create horizontal beam along y-axis if both ends are in the face
                        if (self._is_point_in_face_2d(np.array([x, y1]), vertices_2d) and 
                            self._is_point_in_face_2d(np.array([x, y2]), vertices_2d)):
                            start = np.array([x, y1, floor_top])
                            end = np.array([x, y2, floor_top])
                            beam = self._create_beam(start, end, depth/2, normalized_color)
                            elements.append(beam)
                
                for i in range(len(x_points) - 1):
                    for j in range(len(y_points)):
                        x1 = x_points[i]
                        x2 = x_points[i+1]
                        y = y_points[j]
                        
                        # Create horizontal beam along x-axis if both ends are in the face
                        if (self._is_point_in_face_2d(np.array([x1, y]), vertices_2d) and 
                            self._is_point_in_face_2d(np.array([x2, y]), vertices_2d)):
                            start = np.array([x1, y, floor_top])
                            end = np.array([x2, y, floor_top])
                            beam = self._create_beam(start, end, depth/2, normalized_color)
                            elements.append(beam)
            
            # If using the old implementation, return the elements here
            if elements:
                return elements
                
        # ... [rest of the function with other system types remains unchanged]

    def _create_beam(self, start, end, radius, color, is_column=False):
        """Create a cylindrical beam between two points
        
        If is_column is True, creates a hollow/wireframe column instead of a solid beam
        """
        # Calculate beam direction and length
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 1e-6:  # Avoid zero-length beams
            return None
            
        direction = direction / length
        
        if is_column:
            # For columns, create a wireframe tube with lines instead of a solid cylinder
            # Use a smaller radius for the frame elements
            wireframe_radius = radius * 0.3
            
            # Create vertical line elements for the corners of a square
            corner_points = [
                (radius, radius), (radius, -radius),
                (-radius, -radius), (-radius, radius)
            ]
            
            # Create a compound item to hold all elements of the wireframe column
            column = gl.GLViewWidget.GLViewWidget()
            
            # Create vertical lines at corners
            for x, y in corner_points:
                # Create a thin cylinder for each corner
                corner_mesh = gl.MeshData.cylinder(rows=6, cols=6, radius=[wireframe_radius, wireframe_radius], length=length)
                corner_line = gl.GLMeshItem(
                    meshdata=corner_mesh,
                    smooth=True,
                    color=color,
                    shader='shaded',
                    glOptions='translucent'
                )
                
                # Position at corner
                corner_line.translate(x, y, -length/2)
                column.addItem(corner_line)
            
            # Create horizontal connectors at top and bottom
            for z_pos in [-length/2, length/2]:
                for i in range(4):
                    x1, y1 = corner_points[i]
                    x2, y2 = corner_points[(i+1) % 4]
                    
                    # Calculate connector length and direction
                    conn_start = np.array([x1, y1, z_pos])
                    conn_end = np.array([x2, y2, z_pos])
                    conn_dir = conn_end - conn_start
                    conn_length = np.linalg.norm(conn_dir)
                    
                    if conn_length < 1e-6:
                        continue
                        
                    # Create connector mesh
                    conn_mesh = gl.MeshData.cylinder(rows=6, cols=6, 
                                                   radius=[wireframe_radius, wireframe_radius], 
                                                   length=conn_length)
                    connector = gl.GLMeshItem(
                        meshdata=conn_mesh,
                        smooth=True,
                        color=color,
                        shader='shaded',
                        glOptions='translucent'
                    )
                    
                    # Position and orient connector
                    conn_dir = conn_dir / conn_length
                    connector.translate(0, 0, -conn_length/2)
                    
                    # Rotate to align with connector direction
                    z_axis = np.array([0, 0, 1])
                    rot_axis = np.cross(z_axis, np.array([conn_dir[0], conn_dir[1], 0]))
                    rot_angle = np.arccos(np.dot(z_axis, np.array([0, 0, 1])))
                    if np.linalg.norm(rot_axis) > 1e-6:
                        connector.rotate(rot_angle * 180 / np.pi, rot_axis[0], rot_axis[1], rot_axis[2])
                    
                    # Rotate to horizontal
                    connector.rotate(90, 1, 0, 0)
                    
                    # Move to position
                    connector.translate(x1, y1, z_pos)
                    column.addItem(connector)
            
            # Now apply the overall transformation to the compound item
            # Calculate rotation to align with direction
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, direction)
            
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(z_axis, direction))
                
                # Apply rotation using direct rotate method
                column.rotate(angle * 180 / np.pi, rotation_axis[0], rotation_axis[1], rotation_axis[2])
            
            # Translate to the final position
            column.translate(start[0], start[1], start[2])
            
            return column
        else:
            # For beams, create a regular solid cylinder
            mesh = gl.MeshData.cylinder(rows=10, cols=10, radius=[radius, radius], length=length)
            
            # Create mesh item
            beam = gl.GLMeshItem(
                meshdata=mesh,
                smooth=True,
                color=color,
                shader='shaded',
                glOptions='opaque'
            )
            
            # Position and orient the beam:
            # First, place the beam with one end at the origin
            beam.translate(0, 0, -length/2)
            
            # Calculate rotation to align with direction
            # Default cylinder is along z-axis
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, direction)
            
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(z_axis, direction))
                
                # Apply rotation using separate rotate methods
                beam.rotate(angle * 180 / np.pi, rotation_axis[0], rotation_axis[1], rotation_axis[2])
            
            # Translate to the final position
            beam.translate(start[0], start[1], start[2])
            
            return beam

    def _create_waffle_beam(self, start, end, normal, depth, color):
        """Create a waffle-style beam (rectangular profile)"""
        # Calculate beam direction and length
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 1e-6:  # Avoid zero-length beams
            return None
            
        direction = direction / length
        
        # Define beam dimensions
        width = depth / 4
        height = depth
        
        # Create vertices for a box mesh
        verts = np.array([
            [-width/2, -height/2, 0],
            [width/2, -height/2, 0],
            [width/2, height/2, 0],
            [-width/2, height/2, 0],
            [-width/2, -height/2, length],
            [width/2, -height/2, length],
            [width/2, height/2, length],
            [-width/2, height/2, length]
        ])
        
        # Define faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Front face
            [4, 6, 5], [4, 7, 6],  # Back face
            [0, 4, 5], [0, 5, 1],  # Bottom face
            [1, 5, 6], [1, 6, 2],  # Right face
            [2, 6, 7], [2, 7, 3],  # Top face
            [3, 7, 4], [3, 4, 0]   # Left face
        ])
        
        # Create mesh data
        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        
        # Create mesh item
        beam = gl.GLMeshItem(
            meshdata=meshdata,
            smooth=False,
            color=color,
            shader='shaded',
            glOptions='opaque'
        )
        
        # Position and orient the beam
        # First align with direction axis
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.dot(z_axis, direction))
            
            # Apply rotation using direct rotate method instead of matrix
            beam.rotate(angle * 180 / np.pi, rotation_axis[0], rotation_axis[1], rotation_axis[2])
            
        # Additional rotation to align height with the normal vector
        # This makes the beam stand perpendicular to the face
        right_vec = np.cross(direction, normal)
        if np.linalg.norm(right_vec) > 1e-6:
            right_vec = right_vec / np.linalg.norm(right_vec)
            up_vec = np.cross(right_vec, direction)
            if np.linalg.norm(up_vec) > 1e-6:
                up_vec = up_vec / np.linalg.norm(up_vec)
                
                # Calculate rotation to align y-axis with up_vec
                y_axis = np.array([0, 1, 0])
                rot_axis = np.cross(y_axis, up_vec)
                
                if np.linalg.norm(rot_axis) > 1e-6:
                    rot_axis = rot_axis / np.linalg.norm(rot_axis)
                    angle = np.arccos(np.dot(y_axis, up_vec))
                    
                    # Apply rotation using direct rotate method
                    beam.rotate(angle * 180 / np.pi, rot_axis[0], rot_axis[1], rot_axis[2])
        
        # Translate to the final position
        beam.translate(start[0], start[1], start[2])
        
        return beam
    
    def _is_point_in_face(self, point, face_vertices):
        """Check if a point is inside a 3D face"""
        if len(face_vertices) < 3:
            return False
            
        # Calculate face normal
        v0, v1, v2 = face_vertices[0], face_vertices[1], face_vertices[2]
        normal = np.cross(v1 - v0, v2 - v0)
        
        if np.linalg.norm(normal) < 1e-6:
            return False
            
        normal = normal / np.linalg.norm(normal)
        
        # Project point and face to 2D
        # Choose a coordinate system in the face plane
        basis1 = (v1 - v0) / np.linalg.norm(v1 - v0)
        basis2 = np.cross(normal, basis1)
        
        # Project face vertices to 2D
        face_2d = []
        for v in face_vertices:
            v_rel = v - v0
            x = np.dot(v_rel, basis1)
            y = np.dot(v_rel, basis2)
            face_2d.append(np.array([x, y]))
            
        # Project point to 2D
        p_rel = point - v0
        p_2d = np.array([np.dot(p_rel, basis1), np.dot(p_rel, basis2)])
        
        # Check if point is in the 2D polygon
        return self._is_point_in_polygon_2d(p_2d, face_2d)
    
    def _is_point_in_polygon_2d(self, point, polygon):
        """Check if a 2D point is inside a 2D polygon using ray casting algorithm"""
        if len(polygon) < 3:
            return False
            
        # Ray casting algorithm
        inside = False
        x, y = point
        
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            # Check if ray intersects segment
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
                
        return inside
    
    def _is_line_intersecting_face(self, start, end, face_vertices):
        """Check if a line segment intersects with a face"""
        if len(face_vertices) < 3:
            return False
        
        # Calculate face normal
        v0, v1, v2 = face_vertices[0], face_vertices[1], face_vertices[2]
        normal = np.cross(v1 - v0, v2 - v0)
        
        if np.linalg.norm(normal) < 1e-6:
            return False
            
        normal = normal / np.linalg.norm(normal)
        
        # Calculate intersection point with the face plane
        direction = end - start
        t = np.dot(v0 - start, normal) / np.dot(direction, normal)
        
        # Check if intersection point is within the segment
        if t < 0 or t > 1:
            return False
            
        # Calculate intersection point
        intersection = start + t * direction
        
        # Check if intersection point is inside the face
        return self._is_point_in_face(intersection, face_vertices)
    
    def _line_polygon_intersections_3d(self, start, end, polygon_vertices):
        """Find all intersection points between a line and a 3D polygon"""
        if len(polygon_vertices) < 3:
            return []
            
        # Calculate polygon normal
        v0, v1, v2 = polygon_vertices[0], polygon_vertices[1], polygon_vertices[2]
        normal = np.cross(v1 - v0, v2 - v0)
        
        if np.linalg.norm(normal) < 1e-6:
            return []
            
        normal = normal / np.linalg.norm(normal)
        
        # Project to 2D coordinates in the polygon plane
        basis1 = (v1 - v0) / np.linalg.norm(v1 - v0)
        basis2 = np.cross(normal, basis1)
        
        # Project polygon vertices
        polygon_2d = []
        for v in polygon_vertices:
            v_rel = v - v0
            x = np.dot(v_rel, basis1)
            y = np.dot(v_rel, basis2)
            polygon_2d.append(np.array([x, y]))
        
        # Project line to plane
        direction = end - start
        
        # If line is parallel to plane, no intersections
        if abs(np.dot(direction, normal)) < 1e-6:
            return []
            
        # Calculate intersection with the plane
        t_plane = np.dot(v0 - start, normal) / np.dot(direction, normal)
        
        if t_plane < 0 or t_plane > 1:
            # Line segment doesn't intersect the plane
            return []
            
        plane_intersection = start + t_plane * direction
        
        # Check if intersection is inside polygon
        if self._is_point_in_face(plane_intersection, polygon_vertices):
            return [plane_intersection]
            
        # Generate additional points along the line to handle edge intersections
        intersections = []
        
        # Check intersections with polygon edges
        for i in range(len(polygon_vertices)):
            j = (i + 1) % len(polygon_vertices)
            edge_start = polygon_vertices[i]
            edge_end = polygon_vertices[j]
            
            # Check for intersection between line and edge
            intersection = self._line_line_intersection_3d(start, end, edge_start, edge_end)
            if intersection is not None:
                intersections.append(intersection)
        
        return intersections
    
    def _line_line_intersection_3d(self, line1_start, line1_end, line2_start, line2_end):
        """Find intersection between two 3D line segments if it exists"""
        # Convert to parametric form
        p1, p2 = line1_start, line1_end
        p3, p4 = line2_start, line2_end
        
        v1 = p2 - p1
        v2 = p4 - p3
        
        # Check if lines are parallel
        cross_v1v2 = np.cross(v1, v2)
        if np.linalg.norm(cross_v1v2) < 1e-6:
            return None
            
        # Calculate parameters for closest points
        p13 = p1 - p3
        a = np.dot(v1, v1)
        b = np.dot(v1, v2)
        c = np.dot(v2, v2)
        d = np.dot(v1, p13)
        e = np.dot(v2, p13)
        
        denominator = a*c - b*b
        if abs(denominator) < 1e-6:
            return None
            
        t1 = (b*e - c*d) / denominator
        t2 = (a*e - b*d) / denominator
        
        # Check if intersection is within both segments
        if t1 < 0 or t1 > 1 or t2 < 0 or t2 > 1:
            return None
            
        # Calculate closest points
        p5 = p1 + t1 * v1
        p6 = p3 + t2 * v2
        
        # Check if points are close enough to consider them intersecting
        if np.linalg.norm(p5 - p6) < 1e-6:
            return p5
            
        return None
    
    def export_structure(self):
        """Export the structural system to a 3D file"""
        if not self.structural_systems:
            QMessageBox.warning(self, "No Systems", 
                               "There are no structural systems to export.")
            return
            
        # Select save location
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Export Structure", "", "OBJ Files (*.obj);;STL Files (*.stl)")
            
        if not file_path:
            return
            
        try:
            # Create a new mesh combining all structural elements
            vertices = []
            faces = []
            vertex_offset = 0
            
            # Combine all elements
            for system in self.structural_systems:
                for element in system['elements']:
                    # Get mesh data
                    if hasattr(element, 'meshdata'):
                        md = element.meshdata
                        verts = md.vertexes()
                        element_faces = md.faces()
                        
                        if verts is not None and element_faces is not None:
                            # Apply element transformation
                            transform = element.transform()
                            transformed_verts = []
                            
                            for v in verts:
                                tv = transform.map(pg.Vector3D(v[0], v[1], v[2]))
                                transformed_verts.append([tv.x(), tv.y(), tv.z()])
                            
                            # Add to combined mesh
                            vertices.extend(transformed_verts)
                            
                            # Adjust face indices
                            adjusted_faces = element_faces + vertex_offset
                            faces.extend(adjusted_faces)
                            
                            # Update offset
                            vertex_offset += len(transformed_verts)
            
            # Create trimesh object
            if vertices and faces:
                export_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
                
                # Export to file
                export_mesh.export(file_path)
                
                QMessageBox.information(self, "Export Successful", 
                                       f"Structure exported to {file_path}")
            else:
                QMessageBox.warning(self, "Export Failed", 
                                   "Failed to create mesh from structural elements.")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", 
                               f"Failed to export structure: {str(e)}")

    def on_geometry_selected(self, geometry):
        """Handle selection of a geometry from the 3D view"""
        # Store the selected geometry
        self.selected_geometry = geometry
        
        # Update the selection status
        self.selection_status.setText("Geometry selected - Ready to apply structural system")
        
        # First reset all geometries to default appearance
        for i, geom in enumerate(self.view3d.geometries):
            if 'mesh_item' in geom and geom['mesh_item'] is not None and 'data' in geom:
                mesh = geom['data']
                if hasattr(mesh, 'faces'):
                    vertices = mesh.vertices
                    faces = mesh.faces
                    
                    # Default semi-transparent blue for non-selected geometries
                    face_colors = np.array([[0.7, 0.7, 1.0, 0.2] for _ in range(len(faces))])
                    
                    geom['mesh_item'].setMeshData(
                        vertexes=vertices,
                        faces=faces,
                        faceColors=face_colors,
                        smooth=True,
                        drawEdges=False
                    )
        
        # Now highlight the selected geometry with a vivid green color
        if 'mesh_item' in geometry and geometry['mesh_item'] is not None:
            # Set selection color (bright green) with higher opacity for better visibility
            if 'data' in geometry and hasattr(geometry['data'], 'faces'):
                mesh = geometry['data']
                vertices = mesh.vertices
                faces = mesh.faces
                
                # Create face colors that are highly visible - bright green with opacity
                face_colors = np.array([[0.2, 0.9, 0.3, 0.7] for _ in range(len(faces))])
                
                # Create edge colors for wireframe highlight (bright green)
                edge_colors = np.array([[0.4, 1.0, 0.4, 0.9] for _ in range(len(faces)*3)])
                
                geometry['mesh_item'].setMeshData(
                    vertexes=vertices,
                    faces=faces,
                    faceColors=face_colors,
                    edgeColors=edge_colors,  # Set edge colors directly here
                    smooth=True,
                    drawEdges=True  # Show edges for better visibility of the selected model
                )
                
        # Update visibility button text to reflect that a model is selected
        self.visibility_btn.setText("Hide Selected Model")
        self.visibility_btn.setEnabled(True)

    def find_coplanar_surfaces(self, mesh):
        """
        Find and group coplanar faces in the mesh
        Returns a list of lists, where each inner list contains indices of coplanar faces
        """
        if not hasattr(mesh, 'faces') or not hasattr(mesh, 'face_normals'):
            # Calculate face normals if not present
            if not hasattr(mesh, 'face_normals'):
                mesh._cache.clear()
                mesh.face_normals  # This will compute face normals
                
        face_normals = mesh.face_normals
        face_count = len(mesh.faces)
        
        # Dictionary to store the groups. Key: group_id, Value: list of face indices
        coplanar_groups = {}
        group_id = 0
        processed = set()
        
        # Threshold for determining if normals are parallel (dot product close to 1 or -1)
        normal_threshold = 0.999
        
        # Threshold for determining if faces are coplanar (distance from plane)
        distance_threshold = 0.001
        
        for face_idx in range(face_count):
            if face_idx in processed:
                continue
                
            # Start a new group with this face
            current_group = [face_idx]
            processed.add(face_idx)
            
            # Get face normal and a point on the face for plane equation
            current_normal = face_normals[face_idx]
            face_vertices = mesh.vertices[mesh.faces[face_idx]]
            point_on_plane = np.mean(face_vertices, axis=0)
            
            # Find all other faces with parallel normals
            for other_idx in range(face_count):
                if other_idx in processed:
                    continue
                    
                # Check if normals are parallel
                other_normal = face_normals[other_idx]
                dot_product = np.abs(np.dot(current_normal, other_normal))
                
                if dot_product > normal_threshold:
                    # Check if the faces are coplanar
                    other_vertices = mesh.vertices[mesh.faces[other_idx]]
                    other_centroid = np.mean(other_vertices, axis=0)
                    
                    # Distance from point to plane
                    distance = np.abs(np.dot(other_centroid - point_on_plane, current_normal))
                    
                    if distance < distance_threshold:
                        current_group.append(other_idx)
                        processed.add(other_idx)
            
            coplanar_groups[group_id] = current_group
            group_id += 1
            
        # Convert to list of lists for easier use
        return list(coplanar_groups.values())

    def _is_point_in_face_2d(self, point, vertices_2d):
        """Check if a 2D point is inside the 2D projection of a face
        
        Args:
            point: 2D point (x, y) as numpy array
            vertices_2d: List of 2D vertices representing the face projection
            
        Returns:
            Boolean indicating if the point is inside the face projection
        """
        if len(vertices_2d) < 3:
            return False
            
        # Ray casting algorithm for 2D polygon
        inside = False
        x, y = point[0], point[1]
        
        for i in range(len(vertices_2d)):
            j = (i + 1) % len(vertices_2d)
            xi, yi = vertices_2d[i][0], vertices_2d[i][1]
            xj, yj = vertices_2d[j][0], vertices_2d[j][1]
            
            # Check if ray intersects segment
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
                
        return inside

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