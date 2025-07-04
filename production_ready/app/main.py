#!/usr/bin/env python3
"""
Production-Ready Skin Analyzer API for Google Cloud Deployment
============================================================

Clean architecture Flask API using modular components for better
maintainability and production deployment.
"""

import os
import sys
import logging
import base64
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import SkinAnalyzer, decode_base64_image, preprocess_image, enhance_image_quality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize the ML analyzer
analyzer = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_analyzer():
    """Initialize the ML analyzer."""
    global analyzer
    
    try:
        analyzer = SkinAnalyzer()
        logger.info("ML analyzer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return False

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Skin Type Analyzer API',
        'version': '1.0.0',
        'models_loaded': analyzer is not None
    })

@app.route('/index', methods=['GET'])
def index():
    """Serve the web interface (index route)."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({'error': 'Web interface not available'}), 500

@app.route('/web', methods=['GET'])
def web_interface():
    """Serve the web interface."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving web interface: {e}")
        return jsonify({'error': 'Web interface not available'}), 500

@app.route('/about', methods=['GET'])
def about():
    """Serve the about page."""
    try:
        return render_template('about.html')
    except Exception as e:
        logger.error(f"Error serving about: {e}")
        return jsonify({'error': 'About page not available'}), 500

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_skin():
    """Analyze skin type from uploaded image."""
    
    # Handle GET request - provide API documentation
    if request.method == 'GET':
        return jsonify({
            'endpoint': '/analyze',
            'method': 'POST',
            'description': 'Upload an image for skin type analysis',
            'content_type': 'multipart/form-data',
            'parameters': {
                'image': 'Image file (PNG, JPG, JPEG, GIF, BMP)'
            },
            'example_curl': 'curl -X POST -F "image=@your_image.jpg" https://skin-analyzer-uw5iaen7va-uc.a.run.app/analyze',
            'web_interface': 'https://skin-analyzer-uw5iaen7va-uc.a.run.app/web',
            'models_loaded': analyzer is not None,
            'max_file_size': '16MB',
            'supported_formats': list(ALLOWED_EXTENSIONS)
        })
    
    # Handle POST request - actual analysis
    try:
        # Check if analyzer is loaded
        if analyzer is None:
            return jsonify({
                'error': 'Analyzer not loaded',
                'status': 'error'
            }), 500
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'status': 'error'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'error': 'No image file selected',
                'status': 'error'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP',
                'status': 'error'
            }), 400
        
        # Process the image
        try:
            # Read image as bytes
            image_bytes = file.read()
            
            # Convert to base64 for processing
            import base64
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            
            # Decode and preprocess image
            image = decode_base64_image(base64_data)
            if image is None:
                return jsonify({
                    'error': 'Failed to decode image',
                    'status': 'error'
                }), 400
            
            # Enhance and preprocess
            enhanced_image = enhance_image_quality(image)
            processed_image = preprocess_image(enhanced_image)
            
            # Analyze the image
            result = analyzer.predict(processed_image)
            
            if result['success']:
                logger.info(f"Analysis complete: {result['predicted_type']} ({result['confidence']:.2f})")
                return jsonify({
                    'predicted_skin_type': result['predicted_type'],
                    'confidence': result['confidence'],
                    'confidence_scores': result['all_scores'],
                    'status': 'success'
                })
            else:
                return jsonify({
                    'error': result.get('error', 'Analysis failed'),
                    'status': 'error'
                }), 500
            
        except Exception as e:
            logger.error(f"Error processing uploaded image: {str(e)}")
            return jsonify({
                'error': 'Failed to process image',
                'details': str(e),
                'status': 'error'
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/analyze-base64', methods=['GET', 'POST'])
def analyze_skin_base64():
    """Analyze skin type from base64 encoded image."""
    
    # Handle GET request - provide API documentation
    if request.method == 'GET':
        return jsonify({
            'endpoint': '/analyze-base64',
            'method': 'POST',
            'description': 'Upload a base64 encoded image for skin type analysis',
            'content_type': 'application/json',
            'parameters': {
                'image': 'Base64 encoded image string'
            },
            'example_curl': '''curl -X POST -H "Content-Type: application/json" -d '{"image": "data:image/jpeg;base64,/9j/4AAQ..."}' https://skin-analyzer-uw5iaen7va-uc.a.run.app/analyze-base64''',
            'models_loaded': analyzer is not None,
            'supported_formats': list(ALLOWED_EXTENSIONS)
        })
    
    # Handle POST request - actual analysis
    try:
        # Check if analyzer is loaded
        if analyzer is None:
            return jsonify({
                'error': 'Analyzer not loaded',
                'status': 'error'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No base64 image data provided',
                'status': 'error'
            }), 400
        
        try:
            # Decode and preprocess image
            image = decode_base64_image(data['image'])
            if image is None:
                return jsonify({
                    'error': 'Failed to decode base64 image',
                    'status': 'error'
                }), 400
            
            # Enhance and preprocess
            enhanced_image = enhance_image_quality(image)
            processed_image = preprocess_image(enhanced_image)
            
            # Analyze the image
            result = analyzer.predict(processed_image)
            
            if result['success']:
                logger.info(f"Base64 analysis complete: {result['predicted_type']} ({result['confidence']:.2f})")
                return jsonify({
                    'predicted_skin_type': result['predicted_type'],
                    'confidence': result['confidence'],
                    'confidence_scores': result['all_scores'],
                    'status': 'success'
                })
            else:
                return jsonify({
                    'error': result.get('error', 'Analysis failed'),
                    'status': 'error'
                }), 500
            
        except Exception as e:
            logger.error(f"Error processing base64 image: {str(e)}")
            return jsonify({
                'error': 'Failed to process base64 image',
                'details': str(e),
                'status': 'error'
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in base64 endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large. Maximum size is 16MB.',
        'status': 'error'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

def init_app():
    """Initialize the application."""
    logger.info("🚀 Initializing Skin Analyzer API...")
    
    if initialize_analyzer():
        logger.info("✅ Analyzer initialized successfully")
    else:
        logger.error("❌ Failed to initialize analyzer")

if __name__ == '__main__':
    init_app()
    
    # Get port from environment variable (for Google Cloud Run)
    port = int(os.environ.get('PORT', 8080))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)

# For production WSGI servers (like gunicorn) - always initialize
init_app()
