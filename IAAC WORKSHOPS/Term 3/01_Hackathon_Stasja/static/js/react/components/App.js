 import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import './App.css';

const App = () => {
    const [models, setModels] = useState([]);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadError, setUploadError] = useState('');

    // Fetch models on component mount
    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const response = await axios.get('/api/models');
            setModels(response.data);
        } catch (error) {
            console.error('Error fetching models:', error);
        }
    };

    const handleFileUpload = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        const file = event.target.file.files[0];
        
        if (!file) {
            setUploadError('Please select a file');
            return;
        }
        
        if (!file.name.endsWith('.obj')) {
            setUploadError('Only .obj files are supported');
            return;
        }
        
        formData.append('file', file);
        setIsUploading(true);
        setUploadError('');
        
        try {
            await axios.post('/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                },
                onUploadProgress: (progressEvent) => {
                    const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setUploadProgress(progress);
                }
            });
            
            // Reset form and state
            event.target.reset();
            setIsUploading(false);
            setUploadProgress(0);
            
            // Refresh the model list
            fetchModels();
        } catch (error) {
            setIsUploading(false);
            setUploadError('Upload failed: ' + (error.response?.data || error.message));
            console.error('Upload error:', error);
        }
    };

    const deleteModel = async (filename) => {
        if (window.confirm('Are you sure you want to delete this model?')) {
            try {
                await axios.post(`/delete/${filename}`);
                fetchModels(); // Refresh list after delete
            } catch (error) {
                console.error('Error deleting model:', error);
                alert('Failed to delete model');
            }
        }
    };

    const deleteAllModels = async () => {
        if (window.confirm('Are you sure you want to delete ALL models? This cannot be undone.')) {
            try {
                await axios.post('/delete_all');
                fetchModels(); // Refresh list after delete
            } catch (error) {
                console.error('Error deleting all models:', error);
                alert('Failed to delete all models');
            }
        }
    };

    return (
        <div className="app-container">
            <header className="app-header">
                <h1>3D Model Viewer</h1>
            </header>

            <section className="upload-section">
                <h2>Upload New Model</h2>
                {uploadError && <div className="error-message">{uploadError}</div>}
                
                <form onSubmit={handleFileUpload} className="upload-form">
                    <div className="form-group">
                        <input 
                            type="file" 
                            name="file" 
                            accept=".obj" 
                            disabled={isUploading}
                            className="file-input"
                        />
                    </div>
                    
                    {isUploading && (
                        <div className="progress-bar-container">
                            <div 
                                className="progress-bar" 
                                style={{ width: `${uploadProgress}%` }}
                            >
                                {uploadProgress}%
                            </div>
                        </div>
                    )}
                    
                    <button 
                        type="submit" 
                        disabled={isUploading} 
                        className="upload-button"
                    >
                        {isUploading ? 'Uploading...' : 'Upload Model'}
                    </button>
                </form>
            </section>

            <section className="models-section">
                <div className="models-header">
                    <h2>Uploaded Models</h2>
                    {models.length > 1 && (
                        <button 
                            onClick={deleteAllModels}
                            className="delete-all-button"
                        >
                            Delete All Models
                        </button>
                    )}
                </div>
                
                {models.length > 0 ? (
                    <div className="models-grid">
                        {models.map((model) => (
                            <div key={model.filename} className="model-card">
                                <div className="model-card-header">
                                    <h3>{model.original_name}</h3>
                                </div>
                                <div className="model-card-body">
                                    <p><strong>Uploaded:</strong> {model.upload_date}</p>
                                    <p><strong>Last Modified:</strong> {model.last_modified}</p>
                                </div>
                                <div className="model-card-footer">
                                    <Link to={`/view/${model.filename}`} className="view-button">
                                        View Model
                                    </Link>
                                    <button 
                                        onClick={() => deleteModel(model.filename)}
                                        className="delete-button"
                                    >
                                        Delete
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="no-models">
                        <p>No models uploaded yet. Upload your first 3D model to get started!</p>
                    </div>
                )}
            </section>
        </div>
    );
};

export default App;