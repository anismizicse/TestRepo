#!/usr/bin/env python3
"""
Flask Web Application for Skin Type Analysis
Allows users to upload images and get detailed skin type analysis
"""

import os
import sys
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import base64
from io import BytesIO

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_unified import UnifiedSkinTypePredictor

app = Flask(__name__)
app.secret_key = 'skin_analyzer_secret_key_2025'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path):
    """Convert image to base64 for web display"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except:
        return None

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Handle image upload and analysis"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get model selection from form
            model_type = request.form.get('model_type', 'random_forest')
            detailed_analysis = request.form.get('detailed_analysis') == 'on'
            
            # Perform skin analysis
            predictor = UnifiedSkinTypePredictor(model_type=model_type)
            
            if detailed_analysis:
                result = predictor.analyze_skin_characteristics(filepath)
            else:
                result = predictor.predict_image(filepath)
            
            # Encode image for display
            image_base64 = encode_image_to_base64(filepath)
            
            # Prepare result data
            analysis_data = {
                'result': result,
                'image_base64': image_base64,
                'filename': filename,
                'model_type': model_type,
                'detailed_analysis': detailed_analysis,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Clean up uploaded file (optional)
            # os.remove(filepath)
            
            return render_template('results.html', data=analysis_data)
        
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.')
            return redirect(url_for('index'))
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        model_type = request.form.get('model_type', 'random_forest')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform analysis
            predictor = UnifiedSkinTypePredictor(model_type=model_type)
            result = predictor.analyze_skin_characteristics(filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(result)
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def demo():
    """Demo page with sample images"""
    sample_images = []
    sample_dir = 'sample_faces'
    
    if os.path.exists(sample_dir):
        for filename in os.listdir(sample_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(sample_dir, filename)
                image_base64 = encode_image_to_base64(image_path)
                if image_base64:
                    sample_images.append({
                        'filename': filename,
                        'base64': image_base64,
                        'path': image_path
                    })
    
    return render_template('demo.html', sample_images=sample_images)

@app.route('/demo/analyze/<path:image_path>')
def demo_analyze(image_path):
    """Analyze a demo image"""
    try:
        if not os.path.exists(image_path):
            flash('Demo image not found')
            return redirect(url_for('demo'))
        
        # Perform analysis with Random Forest (best for real photos)
        predictor = UnifiedSkinTypePredictor(model_type='random_forest')
        result = predictor.analyze_skin_characteristics(image_path)
        
        # Encode image for display
        image_base64 = encode_image_to_base64(image_path)
        
        analysis_data = {
            'result': result,
            'image_base64': image_base64,
            'filename': os.path.basename(image_path),
            'model_type': 'random_forest',
            'detailed_analysis': True,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'is_demo': True
        }
        
        return render_template('results.html', data=analysis_data)
    
    except Exception as e:
        flash(f'Error analyzing demo image: {str(e)}')
        return redirect(url_for('demo'))

@app.route('/about')
def about():
    """About page with system information"""
    return render_template('about.html')

if __name__ == '__main__':
    print("🚀 Starting Skin Type Analyzer Web Application")
    print("=" * 50)
    print("🌐 Open your browser and go to: http://localhost:8080")
    print("📱 Features:")
    print("   • Upload and analyze your own images")
    print("   • Try demo with sample images")
    print("   • Choose different ML models")
    print("   • Get detailed skin analysis")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=8080)
