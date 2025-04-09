import React, { useEffect, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import axios from 'axios';
import './ModelViewer.css';

const ModelViewer = () => {
    const { filename } = useParams();
    const navigate = useNavigate();
    const containerRef = useRef(null);
    const rendererRef = useRef(null);
    const sceneRef = useRef(null);
    const cameraRef = useRef(null);
    const controlsRef = useRef(null);
    const loadingRef = useRef(null);
    const objectRef = useRef(null);
    
    const [isLoading, setIsLoading] = useState(true);
    const [loadingProgress, setLoadingProgress] = useState(0);
    const [error, setError] = useState('');
    const [modelInfo, setModelInfo] = useState(null);
    const [showGrid, setShowGrid] = useState(true);
    const [contextMenu, setContextMenu] = useState(null);
    
    // Initialize the Three.js scene
    useEffect(() => {
        if (!containerRef.current) return;
        
        let animationFrameId;
        
        try {
            // Create scene
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a192f); // Dark blue background
            sceneRef.current = scene;
            
            // Camera setup
            const camera = new THREE.PerspectiveCamera(
                45,
                containerRef.current.clientWidth / containerRef.current.clientHeight,
                0.1,
                1000
            );
            camera.position.set(10, 10, 10);
            camera.lookAt(0, 0, 0);
            cameraRef.current = camera;
            
            // Renderer setup with antialias for smoother edges
            const renderer = new THREE.WebGLRenderer({
                antialias: true,
                alpha: true
            });
            renderer.setSize(
                containerRef.current.clientWidth,
                containerRef.current.clientHeight
            );
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            containerRef.current.appendChild(renderer.domElement);
            rendererRef.current = renderer;
            
            // Enhanced lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight1.position.set(5, 10, 5);
            directionalLight1.castShadow = true;
            directionalLight1.shadow.camera.near = 0.5;
            directionalLight1.shadow.camera.far = 50;
            directionalLight1.shadow.mapSize.width = 1024;
            directionalLight1.shadow.mapSize.height = 1024;
            scene.add(directionalLight1);
            
            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight2.position.set(-5, 5, -5);
            scene.add(directionalLight2);
            
            // Grid helper - explicitly position at y=0
            const gridHelper = new THREE.GridHelper(20, 20, 0x888888, 0x444444);
            gridHelper.position.set(0, 0, 0); // Ensure grid is at y=0
            gridHelper.material.opacity = 0.5;
            gridHelper.material.transparent = true;
            scene.add(gridHelper);
            
            // Controls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = true;
            controls.minDistance = 3;
            controls.maxDistance = 50;
            controls.maxPolarAngle = Math.PI / 1.5;
            controlsRef.current = controls;
            
            // Load model info
            fetch(`/api/model/${filename}`)
                .then(response => response.json())
                .then(data => {
                    setModelInfo(data);
                })
                .catch(err => {
                    console.error("Error fetching model info:", err);
                });
            
            // Load the 3D model
            const loader = new OBJLoader();
            loader.load(
                `/static/models/${filename}`,
                (object) => {
                    // Calculate the bounding box of the model
                    const box = new THREE.Box3().setFromObject(object);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    
                    console.log('Model bounding box:', { 
                        min: box.min, 
                        max: box.max, 
                        size: size 
                    });
                    
                    // Set position to center the model horizontally but place bottom on the grid
                    object.position.x = -center.x;
                    object.position.z = -center.z;
                    
                    // Ensure the bottom of the model is exactly at y=0 (on the grid)
                    object.position.y = -box.min.y;
                    
                    console.log('Model position after adjustment:', object.position);
                    
                    // Scale model to fit view
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scale = 5 / maxDim;
                    object.scale.multiplyScalar(scale);
                    
                    // Add materials if none exist
                    object.traverse(function(child) {
                        if (child.isMesh) {
                            if (!child.material) {
                                child.material = new THREE.MeshPhongMaterial({
                                    color: 0x67d5d3,
                                    emissive: 0x072534,
                                    side: THREE.DoubleSide,
                                    flatShading: true
                                });
                            }
                            
                            child.castShadow = true;
                            child.receiveShadow = true;
                        }
                    });
                    
                    // Add a subtle highlight on the grid under the model
                    const scaledSize = new THREE.Vector3(
                        size.x * scale,
                        size.y * scale,
                        size.z * scale
                    );
                    
                    // Add a subtle ground marker (optional, can be removed if not wanted)
                    const planeGeometry = new THREE.CircleGeometry(Math.max(scaledSize.x, scaledSize.z) * 0.5, 32);
                    const planeMaterial = new THREE.MeshBasicMaterial({ 
                        color: 0x3498db, 
                        transparent: true,
                        opacity: 0.1,
                        side: THREE.DoubleSide
                    });
                    
                    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
                    plane.rotation.x = -Math.PI / 2; // Rotate to be horizontal
                    plane.position.y = 0.01; // Slightly above the grid to avoid z-fighting
                    scene.add(plane);
                    
                    scene.add(object);
                    objectRef.current = object;
                    setIsLoading(false);
                },
                (xhr) => {
                    const progress = Math.round((xhr.loaded / xhr.total) * 100);
                    setLoadingProgress(progress);
                },
                (error) => {
                    console.error('Error loading 3D model:', error);
                    setError(`Error loading 3D model: ${error.message}`);
                    setIsLoading(false);
                }
            );
            
            // Animation loop
            const animate = () => {
                animationFrameId = requestAnimationFrame(animate);
                
                if (controlsRef.current) {
                    controlsRef.current.update();
                }
                
                renderer.render(scene, camera);
            };
            
            animate();
            
            // Handle window resize
            const handleResize = () => {
                if (!containerRef.current) return;
                
                const width = containerRef.current.clientWidth;
                const height = containerRef.current.clientHeight;
                
                cameraRef.current.aspect = width / height;
                cameraRef.current.updateProjectionMatrix();
                rendererRef.current.setSize(width, height);
            };
            
            window.addEventListener('resize', handleResize);
            
            // Context menu event listener
            const handleContextMenu = (event) => {
                event.preventDefault();
                
                if (isLoading) return;
                
                setContextMenu({
                    x: event.clientX,
                    y: event.clientY
                });
            };
            
            // Close context menu when clicking anywhere else
            const handleClick = () => {
                if (contextMenu) setContextMenu(null);
            };
            
            containerRef.current.addEventListener('contextmenu', handleContextMenu);
            window.addEventListener('click', handleClick);
            
            // Cleanup
            return () => {
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                }
                
                window.removeEventListener('resize', handleResize);
                
                if (containerRef.current) {
                    containerRef.current.removeEventListener('contextmenu', handleContextMenu);
                }
                window.removeEventListener('click', handleClick);
                
                if (rendererRef.current && rendererRef.current.domElement) {
                    if (containerRef.current) {
                        containerRef.current.removeChild(rendererRef.current.domElement);
                    }
                    rendererRef.current.dispose();
                }
                
                // Dispose materials and geometries to prevent memory leaks
                if (sceneRef.current) {
                    sceneRef.current.traverse((object) => {
                        if (object.geometry) {
                            object.geometry.dispose();
                        }
                        
                        if (object.material) {
                            if (Array.isArray(object.material)) {
                                object.material.forEach(material => material.dispose());
                            } else {
                                object.material.dispose();
                            }
                        }
                    });
                }
            };
        } catch (err) {
            console.error('Error initializing 3D viewer:', err);
            setError(`Error initializing 3D viewer: ${err.message}`);
            setIsLoading(false);
        }
    }, [filename, contextMenu, isLoading]);
    
    const handleDelete = async () => {
        if (window.confirm('Are you sure you want to delete this model?')) {
            try {
                await axios.post(`/delete/${filename}`);
                navigate('/');
            } catch (error) {
                console.error('Error deleting model:', error);
                setError(`Failed to delete model: ${error.message}`);
            }
        }
    };
    
    // Camera position controls
    const setCameraPosition = (position) => {
        if (!cameraRef.current || !controlsRef.current) return;
        
        switch (position) {
            case 'front':
                cameraRef.current.position.set(0, 0, 15);
                break;
            case 'top':
                cameraRef.current.position.set(0, 15, 0);
                break;
            case 'side':
                cameraRef.current.position.set(15, 0, 0);
                break;
            case 'isometric':
                cameraRef.current.position.set(10, 10, 10);
                break;
            default:
                break;
        }
        
        cameraRef.current.lookAt(0, 0, 0);
        controlsRef.current.update();
    };
    
    // Toggle grid visibility
    const toggleGrid = () => {
        if (!sceneRef.current) return;
        
        sceneRef.current.traverse((object) => {
            if (object instanceof THREE.GridHelper) {
                object.visible = !showGrid;
            }
        });
        
        setShowGrid(!showGrid);
    };

    // Context menu for model manipulation
    const handleContextMenuAction = (action) => {
        if (!objectRef.current) return;
        
        switch (action) {
            case 'wireframe':
                objectRef.current.traverse((child) => {
                    if (child.isMesh && child.material) {
                        child.material.wireframe = !child.material.wireframe;
                    }
                });
                break;
            case 'color':
                const randomColor = new THREE.Color(Math.random() * 0xffffff);
                objectRef.current.traverse((child) => {
                    if (child.isMesh && child.material) {
                        child.material.color = randomColor;
                    }
                });
                break;
            default:
                break;
        }
        
        setContextMenu(null);
    };
    
    return (
        <div className="model-viewer-container">
            <div className="model-viewer-header">
                <button 
                    className="back-button" 
                    onClick={() => navigate('/')}
                >
                    ← Back to List
                </button>
                
                {modelInfo && (
                    <div className="model-info">
                        <h2>{modelInfo.original_name}</h2>
                        <p>Uploaded: {modelInfo.upload_date}</p>
                    </div>
                )}
                
                <button 
                    className="delete-model-button" 
                    onClick={handleDelete}
                >
                    Delete Model
                </button>
            </div>
            
            <div className="model-viewer-content" ref={containerRef}>
                {isLoading && (
                    <div className="loading-overlay" ref={loadingRef}>
                        <div className="loading-container">
                            <div className="loader"></div>
                            <div className="loading-text">
                                Loading Model: {loadingProgress}%
                            </div>
                        </div>
                    </div>
                )}
                
                {error && (
                    <div className="error-overlay">
                        <div className="error-container">
                            <div className="error-icon">⚠️</div>
                            <div className="error-text">{error}</div>
                        </div>
                    </div>
                )}
                
                {!isLoading && !error && (
                    <>
                        <div className="scene-grid">
                            Grid: {showGrid ? 'On' : 'Off'}
                        </div>
                        
                        <div className="camera-positions">
                            <button 
                                className="camera-button"
                                title="Front View"
                                onClick={() => setCameraPosition('front')}
                            >
                                F
                            </button>
                            <button 
                                className="camera-button"
                                title="Top View"
                                onClick={() => setCameraPosition('top')}
                            >
                                T
                            </button>
                            <button 
                                className="camera-button"
                                title="Side View"
                                onClick={() => setCameraPosition('side')}
                            >
                                S
                            </button>
                            <button 
                                className="camera-button"
                                title="Isometric View"
                                onClick={() => setCameraPosition('isometric')}
                            >
                                I
                            </button>
                            <button 
                                className="camera-button"
                                title="Toggle Grid"
                                onClick={toggleGrid}
                            >
                                G
                            </button>
                        </div>
                        
                        {contextMenu && (
                            <div 
                                className="context-menu"
                                style={{ 
                                    top: contextMenu.y,
                                    left: contextMenu.x
                                }}
                            >
                                <div 
                                    className="context-menu-item"
                                    onClick={() => handleContextMenuAction('wireframe')}
                                >
                                    Toggle Wireframe
                                </div>
                                <div className="context-menu-divider"></div>
                                <div 
                                    className="context-menu-item"
                                    onClick={() => handleContextMenuAction('color')}
                                >
                                    Random Color
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
            
            <div className="model-viewer-controls">
                <div className="control-hint">
                    <div className="hint-item">
                        <span className="hint-icon">🖱️</span>
                        <span className="hint-text">Rotate: Click + Drag</span>
                    </div>
                    <div className="hint-item">
                        <span className="hint-icon">⚙️</span>
                        <span className="hint-text">Zoom: Scroll</span>
                    </div>
                    <div className="hint-item">
                        <span className="hint-icon">👆</span>
                        <span className="hint-text">Pan: Shift + Drag</span>
                    </div>
                    <div className="hint-item">
                        <span className="hint-icon">📌</span>
                        <span className="hint-text">Options: Right-click</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ModelViewer;