from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
from datetime import datetime
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database file path
DB_FILE = 'static/models/database.json'

def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return []

def save_database(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# API endpoint to get all models
@app.route('/api/models')
def get_models():
    models = load_database()
    return jsonify(models)

# API endpoint to get a specific model's info
@app.route('/api/model/<filename>')
def get_model(filename):
    database = load_database()
    for model in database:
        if model['filename'] == filename:
            return jsonify(model)
    return jsonify({'error': 'Model not found'}), 404

@app.route('/')
def index():
    # Serve the main React app
    return render_template('index.html')

@app.route('/view/<filename>')
def view_model(filename):
    # For React routing, just return the main template
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.obj'):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"{timestamp}_{filename}"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
        
        # Update database
        database = load_database()
        database.append({
            'filename': new_filename,
            'original_name': filename,
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_modified': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        save_database(database)
        
        return jsonify({'success': True, 'filename': new_filename}), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/delete/<filename>', methods=['POST'])
def delete_model(filename):
    try:
        # Remove file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Update database
        database = load_database()
        database = [model for model in database if model['filename'] != filename]
        save_database(database)
        
        return '', 204  # Success, no content
    except Exception as e:
        return str(e), 500

@app.route('/delete_all', methods=['POST'])
def delete_all_models():
    try:
        # Get all files from database
        database = load_database()
        
        # Delete all model files
        for model in database:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], model['filename'])
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clear the database
        save_database([])
        
        return '', 204  # Success, no content
    except Exception as e:
        return str(e), 500

# Handle all routes for single-page React app
@app.route('/<path:path>')
def catch_all(path):
    # This will handle all client-side routing
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=2000)