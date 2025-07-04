#!/usr/bin/env python3
"""
Skin Analyzer API - Production Ready REST API
============================================

A Flask REST API for skin type analysis that can be consumed by mobile applications.
Optimized for production deployment with proper error handling and response formatting.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import joblib
from datetime import datetime
import base64
import io
from PIL import Image
import logging
from werkzeug.utils import secure_filename
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinAnalyzerAPI:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for mobile app access
        
        # Configuration
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.upload_folder = 'api_uploads'
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # Load ML models
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        
        # Load models on startup
        self._load_models()
        
        # Setup routes
        self._setup_routes()
    
    def _load_models(self):
        """Load all trained ML models and preprocessing components."""
        try:
            models_path = "trained_models"
            
            # Load preprocessing components
            self.scaler = joblib.load(f"{models_path}/scaler.pkl")
            self.label_encoder = joblib.load(f"{models_path}/label_encoder.pkl")
            self.feature_selector = joblib.load(f"{models_path}/feature_selector.pkl")
            
            # Load ML models
            model_files = {
                'random_forest': f"{models_path}/random_forest_optimized.pkl",
                'gradient_boost': f"{models_path}/gradient_boost_optimized.pkl",
                'svm': f"{models_path}/svm_optimized.pkl",
                'neural_network': f"{models_path}/neural_network_optimized.pkl",
                'ensemble': f"{models_path}/ensemble_optimized.pkl"
            }
            
            for model_name, model_path in model_files.items():
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model successfully")
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _extract_features(self, image_path):
        """Extract features from image for ML prediction."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Resize image
            image = cv2.resize(image, (224, 224))
            
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            features = []
            
            # Basic color statistics
            for channel in cv2.split(image):
                features.extend([
                    np.mean(channel), np.std(channel),
                    np.median(channel), np.percentile(channel, 25), np.percentile(channel, 75)
                ])
            
            # HSV color features
            for channel in cv2.split(hsv):
                features.extend([np.mean(channel), np.std(channel)])
            
            # LAB color features
            for channel in cv2.split(lab):
                features.extend([np.mean(channel), np.std(channel)])
            
            # Texture features
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(laplacian_var)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _predict_skin_type(self, image_path, model_name='random_forest'):
        """Predict skin type from image."""
        try:
            # Extract features
            features = self._extract_features(image_path)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Select features
            features_selected = self.feature_selector.transform(features_scaled)
            
            # Get model
            if model_name not in self.models:
                model_name = 'random_forest'  # Default fallback
            
            model = self.models[model_name]
            
            # Make prediction
            prediction = model.predict(features_selected)[0]
            probabilities = model.predict_proba(features_selected)[0]
            
            # Convert prediction back to label
            skin_type = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence scores for all classes
            class_labels = self.label_encoder.classes_
            confidence_scores = {
                class_labels[i]: float(probabilities[i])
                for i in range(len(class_labels))
            }
            
            return {
                'skin_type': skin_type,
                'confidence': float(max(probabilities)),
                'confidence_scores': confidence_scores,
                'model_used': model_name
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'models_loaded': len(self.models),
                'available_models': list(self.models.keys()),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/analyze', methods=['POST'])
        def analyze_image():
            """Analyze skin type from uploaded image."""
            try:
                # Check if image is provided
                if 'image' not in request.files:
                    return jsonify({
                        'error': 'No image provided',
                        'success': False
                    }), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({
                        'error': 'No image selected',
                        'success': False
                    }), 400
                
                # Get model preference
                model_name = request.form.get('model', 'random_forest')
                
                # Save uploaded file
                filename = secure_filename(f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                filepath = os.path.join(self.upload_folder, filename)
                file.save(filepath)
                
                try:
                    # Analyze image
                    result = self._predict_skin_type(filepath, model_name)
                    
                    # Add metadata
                    result.update({
                        'success': True,
                        'timestamp': datetime.now().isoformat(),
                        'filename': filename
                    })
                    
                    # Clean up uploaded file
                    os.remove(filepath)
                    
                    return jsonify(result)
                    
                except Exception as e:
                    # Clean up file on error
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    raise e
                    
            except Exception as e:
                logger.error(f"Error in analyze_image: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/analyze_base64', methods=['POST'])
        def analyze_base64_image():
            """Analyze skin type from base64 encoded image (for mobile apps)."""
            try:
                data = request.get_json()
                
                if not data or 'image' not in data:
                    return jsonify({
                        'error': 'No base64 image data provided',
                        'success': False
                    }), 400
                
                # Decode base64 image
                image_data = data['image']
                if image_data.startswith('data:image'):
                    # Remove data URL prefix
                    image_data = image_data.split(',')[1]
                
                # Decode and save image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Save as temporary file
                filename = f"mobile_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(self.upload_folder, filename)
                image.save(filepath, 'JPEG')
                
                try:
                    # Get model preference
                    model_name = data.get('model', 'random_forest')
                    
                    # Analyze image
                    result = self._predict_skin_type(filepath, model_name)
                    
                    # Add metadata
                    result.update({
                        'success': True,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'mobile_camera'
                    })
                    
                    # Clean up uploaded file
                    os.remove(filepath)
                    
                    return jsonify(result)
                    
                except Exception as e:
                    # Clean up file on error
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    raise e
                    
            except Exception as e:
                logger.error(f"Error in analyze_base64_image: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/models', methods=['GET'])
        def get_available_models():
            """Get list of available ML models."""
            return jsonify({
                'available_models': list(self.models.keys()),
                'default_model': 'random_forest',
                'success': True
            })
        
        @self.app.errorhandler(413)
        def too_large(e):
            return jsonify({
                'error': 'File too large. Maximum size is 16MB.',
                'success': False
            }), 413
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server."""
        logger.info(f"Starting Skin Analyzer API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Create and run API
if __name__ == '__main__':
    try:
        api = SkinAnalyzerAPI()
        
        # Run server
        print("🚀 SKIN ANALYZER API STARTING...")
        print("=" * 50)
        print("🌐 API Endpoints:")
        print("   • GET  /api/health          - Health check")
        print("   • POST /api/analyze         - Analyze image (file upload)")
        print("   • POST /api/analyze_base64  - Analyze image (base64)")
        print("   • GET  /api/models          - Available models")
        print("=" * 50)
        print("📱 Ready for Flutter app integration!")
        print("🔗 API will be available at: http://localhost:8081")
        
        api.run(host='0.0.0.0', port=8081, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        print(f"❌ Error: {e}")
        print("Make sure all model files are present in the trained_models directory")
