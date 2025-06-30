import os  # Add this missing import
import random
import json
from flask import Flask, request, render_template_string, jsonify
from Prototype_Structure_Generator import Generate

PORT = 8765
STORAGE_FOLDER = "Storage/"

app = Flask(__name__)

# Ensure Storage directory exists
os.makedirs(STORAGE_FOLDER, exist_ok=True)

@app.route('/<path:subpath>', methods=['GET', 'PUT'])
def main(subpath):
    filename = f"{STORAGE_FOLDER}/{subpath}"
    if request.method == 'PUT':
        # Received the file, can process it here
        dataRaw = request.get_data()
        with open(filename, "wb") as f:
            f.write(dataRaw)
        # Fix: Only pass the file path to Generate()
        Generate(glb_file_path=filename)  # Use full path including Storage/
        
        # After processing, redirect to visualization if structural_system.json exists
        json_path = "Server/Storage/structural_system.json"
        if os.path.exists(json_path):
            return f'<script>window.location.href="/visualize";</script>'
    elif request.method == 'GET':
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                dataRaw = file.read()
            return dataRaw
        else:
            return "File not found", 404
    return ""

@app.route('/api/json-data')
def get_json_data():
    json_path = "Server/Storage/structural_system.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format"})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "structural_system.json not found"})

@app.route('/visualize')
def visualize():
    # Check if currentModel.glb exists - if not, show upload message
    glb_path = f"{STORAGE_FOLDER}/currentModel.glb"
    json_path = "Server/Storage/structural_system.json"
    
    if not os.path.exists(glb_path):
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Structure Prediction</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; background: #f0f0f0; display: flex; justify-content: center; align-items: center; height: 100vh; }
                .message-container { text-align: center; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
                .message-container h1 { color: #333; margin-bottom: 20px; }
                .message-container p { color: #666; font-size: 16px; margin-bottom: 30px; }
                .upload-info { background: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2196f3; }
                .upload-info code { background: #f5f5f5; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
            </style>
        </head>
        <body>
            <div class="message-container">
                <h1>Upload a GLB Model</h1>
                <p>Please upload a GLB model file to see the structural analysis.</p>
                <div class="upload-info">
                    <strong>Upload your GLB file to:</strong><br>
                    <code>http://192.168.0.13:8765/currentModel.glb</code>
                    <br><br>
                    After upload, the analysis will be processed and you'll see the visualization.
                </div>
            </div>
            <script>
                // Check every 2 seconds if GLB is uploaded
                setInterval(() => {
                    fetch('/api/check-glb')
                        .then(response => response.json())
                        .then(data => {
                            if (data.exists) {
                                window.location.reload();
                            }
                        })
                        .catch(() => {});
                }, 2000);
            </script>
        </body>
        </html>
        ''')
    
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Structure Prediction</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; background: #f0f0f0; overflow: hidden; }
            .viewport { width: 100vw; height: 100vh; position: relative; }
            .header { position: absolute; top: 20px; left: 20px; background: rgba(255,255,255,0.9); padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header h1 { margin: 0; font-size: 18px; color: #333; }
            .controls { position: absolute; top: 20px; right: 20px; background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .control-btn { background: #6c5ce7; color: white; padding: 8px 15px; border: none; border-radius: 6px; cursor: pointer; margin: 2px; font-size: 12px; }
            .control-btn:hover { background: #5a4fcf; }
            .delete-btn { background: #e74c3c; color: white; padding: 8px 15px; border: none; border-radius: 6px; cursor: pointer; margin: 2px; font-size: 12px; }
            .delete-btn:hover { background: #c0392b; }
            #canvas-container { width: 100%; height: 100%; }
            .view-selector { position: absolute; top: 50%; left: 20px; transform: translateY(-50%); background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .view-btn { display: block; background: #74b9ff; color: white; padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; margin: 2px 0; font-size: 11px; }
            .view-btn:hover { background: #0984e3; }
        </style>
    </head>
    <body>
        <div class="viewport">
            <div class="header">
                <h1>Structure prediction</h1>
            </div>
            <div class="controls">
                <button class="control-btn" onclick="resetCamera()">Reset View</button>
                <button class="control-btn" onclick="toggleWireframe()">Wireframe</button>
                <button class="control-btn" onclick="loadData()">Refresh</button>
                <button class="delete-btn" onclick="deleteModel()">Delete Model</button>
            </div>
            <div class="view-selector">
                <button class="view-btn" onclick="setTopView()">Top</button>
                <button class="view-btn" onclick="setFrontView()">Front</button>
                <button class="view-btn" onclick="setSideView()">Side</button>
                <button class="view-btn" onclick="setIsoView()">3D</button>
            </div>
            <div id="canvas-container"></div>
        </div>
        
        <script>
            let scene, camera, renderer, controls;
            let structuralElements = [];
            let wireframeMode = false;
            let boundingBox = { min: { x: 0, y: 0, z: 0 }, max: { x: 10, y: 15, z: 5 } };
            
            function init3D() {
                // Scene setup
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf0f0f0);
                
                // Camera setup
                camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.set(30, 25, 30);
                
                // Renderer setup
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                document.getElementById('canvas-container').appendChild(renderer.domElement);
                
                // Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                
                // Lighting
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
                directionalLight.position.set(50, 50, 25);
                directionalLight.castShadow = true;
                directionalLight.shadow.mapSize.width = 2048;
                directionalLight.shadow.mapSize.height = 2048;
                scene.add(directionalLight);
                
                // Grid (similar to reference image)
                // const gridSize = 50;
                // const gridDivisions = 50;
                // const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, 0xcccccc, 0xcccccc);
                // gridHelper.position.y = 0;
                // scene.add(gridHelper); // <-- Remove or comment out this line to hide the plane
                
                animate();
            }
            
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            
            function clearStructure() {
                structuralElements.forEach(element => {
                    scene.remove(element);
                });
                structuralElements = [];
            }
            
            function visualizeStructure(data) {
                clearStructure();
                
                if (!data || typeof data !== 'object') return;
                
                // Create a lookup for nodes by ID
                const nodeMap = new Map();
                if (data.nodes) {
                    data.nodes.forEach(node => {
                        nodeMap.set(node.ID, node);
                    });
                }
                
                // Create beams as rectangular cross-sections
                if (data.beams && nodeMap.size > 0) {
                    data.beams.forEach(beam => createBeamRectangular(beam, nodeMap));
                }
                
                // Create columns as rectangular cross-sections
                if (data.columns && nodeMap.size > 0) {
                    data.columns.forEach(column => createColumnRectangular(column, nodeMap));
                }
                
                // Center camera on structure
                centerCameraOnStructure(data.nodes);
            }
            
            function createBeamRectangular(beam, nodeMap) {
                const startNode = nodeMap.get(beam.i_node);
                const endNode = nodeMap.get(beam.j_node);
                
                if (!startNode || !endNode) return;
                
                const start = new THREE.Vector3(startNode.X, startNode.Z, startNode.Y);
                const end = new THREE.Vector3(endNode.X, endNode.Z, endNode.Y);
                const direction = new THREE.Vector3().subVectors(end, start);
                const length = direction.length();
                
                // Create rectangular beam - length along Z-axis initially
                const geometry = new THREE.BoxGeometry(0.3, 0.15, length);
                const material = new THREE.MeshLambertMaterial({ 
                    color: 0x4CAF50,  // Green color like in reference
                    transparent: false
                });
                const beam3D = new THREE.Mesh(geometry, material);
                
                // Position at midpoint
                const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
                beam3D.position.copy(midpoint);
                
                // Align beam with direction vector using lookAt
                const target = new THREE.Vector3().addVectors(midpoint, direction.normalize());
                beam3D.lookAt(target);
                
                beam3D.userData = { type: 'beam', id: beam.ID };
                scene.add(beam3D);
                structuralElements.push(beam3D);
            }
            
            function createColumnRectangular(column, nodeMap) {
                const startNode = nodeMap.get(column.i_node);
                const endNode = nodeMap.get(column.j_node);
                
                if (!startNode || !endNode) return;
                
                const start = new THREE.Vector3(startNode.X, startNode.Z, startNode.Y);
                const end = new THREE.Vector3(endNode.X, endNode.Z, endNode.Y);
                const direction = new THREE.Vector3().subVectors(end, start);
                const length = direction.length();
                
                // Create rectangular column - length along Z-axis initially
                const geometry = new THREE.BoxGeometry(0.2, 0.2, length);
                const material = new THREE.MeshLambertMaterial({ 
                    color: 0xf44336,  // Red color like in reference
                    transparent: false
                });
                const column3D = new THREE.Mesh(geometry, material);
                
                // Position at midpoint
                const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
                column3D.position.copy(midpoint);
                
                // Align column with direction vector using lookAt
                const target = new THREE.Vector3().addVectors(midpoint, direction.normalize());
                column3D.lookAt(target);
                
                column3D.userData = { type: 'column', id: column.ID };
                scene.add(column3D);
                structuralElements.push(column3D);
            }
            
            function centerCameraOnStructure(nodes) {
                if (!nodes || nodes.length === 0) return;
                
                // Calculate bounding box
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;
                let minZ = Infinity, maxZ = -Infinity;
                
                nodes.forEach(node => {
                    minX = Math.min(minX, node.X);
                    maxX = Math.max(maxX, node.X);
                    minY = Math.min(minY, node.Y);
                    maxY = Math.max(maxY, node.Y);
                    minZ = Math.min(minZ, node.Z);
                    maxZ = Math.max(maxZ, node.Z);
                });
                
                const centerX = (minX + maxX) / 2;
                const centerY = (minY + maxY) / 2;
                const centerZ = (minZ + maxZ) / 2;
                
                boundingBox = {
                    min: { x: minX, y: minY, z: minZ },
                    max: { x: maxX, y: maxY, z: maxZ }
                };
                
                const sizeX = maxX - minX;
                const sizeY = maxY - minY;
                const sizeZ = maxZ - minZ;
                const maxSize = Math.max(sizeX, sizeY, sizeZ);
                
                // Position camera to view the entire structure
                const distance = maxSize * 1.5;
                camera.position.set(
                    centerX + distance,
                    centerZ + distance,
                    centerY + distance
                );
                
                controls.target.set(centerX, centerZ, centerY);
                controls.update();
            }
            
            // View controls
            function setTopView() {
                const center = {
                    x: (boundingBox.min.x + boundingBox.max.x) / 2,
                    y: (boundingBox.min.y + boundingBox.max.y) / 2,
                    z: (boundingBox.min.z + boundingBox.max.z) / 2
                };
                camera.position.set(center.x, center.z + 20, center.y);
                controls.target.set(center.x, center.z, center.y);
                controls.update();
            }
            
            function setFrontView() {
                const center = {
                    x: (boundingBox.min.x + boundingBox.max.x) / 2,
                    y: (boundingBox.min.y + boundingBox.max.y) / 2,
                    z: (boundingBox.min.z + boundingBox.max.z) / 2
                };
                camera.position.set(center.x, center.z, center.y + 25);
                controls.target.set(center.x, center.z, center.y);
                controls.update();
            }
            
            function setSideView() {
                const center = {
                    x: (boundingBox.min.x + boundingBox.max.x) / 2,
                    y: (boundingBox.min.y + boundingBox.max.y) / 2,
                    z: (boundingBox.min.z + boundingBox.max.z) / 2
                };
                camera.position.set(center.x + 25, center.z, center.y);
                controls.target.set(center.x, center.z, center.y);
                controls.update();
            }
            
            function setIsoView() {
                centerCameraOnStructure();
            }
            
            function resetCamera() {
                loadData(); // This will recenter the camera
            }
            
            function toggleWireframe() {
                wireframeMode = !wireframeMode;
                structuralElements.forEach(element => {
                    if (element.material) {
                        element.material.wireframe = wireframeMode;
                    }
                });
            }
            
            function loadData() {
                fetch('/api/json-data')
                    .then(response => response.json())
                    .then(data => {
                        if (!data.error) {
                            visualizeStructure(data);
                        }
                    })
                    .catch(error => {
                        console.error('Failed to load data:', error);
                    });
            }
            
            function deleteModel() {
                if (confirm('Are you sure you want to delete the current model? This action cannot be undone.')) {
                    fetch('/api/delete-model', {
                        method: 'DELETE'
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Model deleted successfully');
                            window.location.reload();
                        } else {
                            alert('Error deleting model: ' + data.error);
                        }
                    })
                    .catch(error => {
                        alert('Error deleting model: ' + error.message);
                    });
                }
            }
            
            // Window resize handler
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
            
            // Initialize
            init3D();
            loadData();
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/api/delete-model', methods=['DELETE'])
def delete_model():
    try:
        glb_path = f"{STORAGE_FOLDER}/currentModel.glb";
        
        # Only delete the GLB file, keep the JSON for re-use
        if os.path.exists(glb_path):
            os.remove(glb_path)
            return jsonify({
                "success": True,
                "message": "Model deleted successfully. Upload a new GLB file to see analysis."
            })
        else:
            return jsonify({
                "success": False,
                "error": "No GLB model file found to delete"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/check-glb')
def check_glb():
    glb_path = f"{STORAGE_FOLDER}/currentModel.glb"
    return jsonify({"exists": os.path.exists(glb_path)})

if __name__ == '__main__':
    try:
        print(f"Server starting on http://0.0.0.0:{PORT}")
        print(f"Upload files to: http://192.168.0.13:{PORT}/filename.glb")
        app.run(debug=True, port=PORT, host='0.0.0.0')
    except KeyboardInterrupt:
        pass