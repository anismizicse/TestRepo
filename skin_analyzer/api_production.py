#!/usr/bin/env python3
"""
Production-Ready Skin Analyzer API for Google Cloud Deployment
============================================================

Optimized Flask API for deployment on Google Cloud Run with proper
error handling, logging, and production configurations.
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
import traceback

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

# Global variables for models
model = None
scaler = None
label_encoder = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load the trained ML models."""
    global model, scaler, label_encoder
    
    try:
        # Try to load the best performing model
        model_files = [
            'random_forest_optimized.pkl',
            'ensemble_optimized.pkl',
            'gradient_boost_optimized.pkl'
        ]
        
        model_loaded = False
        for model_file in model_files:
            try:
                model = joblib.load(model_file)
                logger.info(f"✅ Loaded model: {model_file}")
                model_loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not model_loaded:
            raise FileNotFoundError("No model files found")
        
        # Load preprocessing components
        try:
            scaler = joblib.load('scaler.pkl')
            logger.info("✅ Loaded scaler")
        except FileNotFoundError:
            logger.warning("⚠️ Scaler not found, using StandardScaler")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        
        try:
            label_encoder = joblib.load('label_encoder.pkl')
            logger.info("✅ Loaded label encoder")
        except FileNotFoundError:
            logger.warning("⚠️ Label encoder not found, using default labels")
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.fit(['combination', 'dry', 'normal', 'oily', 'sensitive'])
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        return False

def extract_features(image):
    """Extract features from image for ML prediction."""
    try:
        # Resize image
        image_resized = cv2.resize(image, (224, 224))
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
        
        features = []
        
        # Basic color statistics (RGB)
        for channel in cv2.split(image_resized):
            features.extend([
                np.mean(channel), np.std(channel),
                np.median(channel), np.percentile(channel, 25), np.percentile(channel, 75)
            ])
        
        # HSV color features
        for channel in cv2.split(hsv):
            features.extend([np.mean(channel), np.std(channel)])
        
        # Texture features (simplified)
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Smoothness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        # Brightness and contrast
        features.extend([np.mean(gray), np.std(gray)])
        
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def process_image(image_data):
    """Process image data and return prediction."""
    try:
        # Convert PIL Image to numpy array
        if isinstance(image_data, Image.Image):
            image_array = np.array(image_data)
        else:
            image_array = image_data
        
        # Ensure RGB format
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_rgb = image_array
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # Convert RGBA to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        else:
            # Convert grayscale to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # Extract features
        features = extract_features(image_rgb)
        
        # Scale features if scaler is available
        if scaler is not None:
            try:
                features_scaled = scaler.transform(features)
            except:
                features_scaled = features
        else:
            features_scaled = features
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get class names
        try:
            classes = label_encoder.classes_
        except:
            classes = ['combination', 'dry', 'normal', 'oily', 'sensitive']
        
        # Create confidence scores
        confidence_scores = {}
        for i, class_name in enumerate(classes):
            if i < len(prediction_proba):
                confidence_scores[class_name] = float(prediction_proba[i])
        
        # Get predicted class name
        if isinstance(prediction, (int, np.integer)):
            if prediction < len(classes):
                predicted_class = classes[prediction]
            else:
                predicted_class = 'unknown'
        else:
            predicted_class = str(prediction)
        
        max_confidence = max(confidence_scores.values()) if confidence_scores else 0.0
        
        return {
            'predicted_skin_type': predicted_class,
            'confidence': float(max_confidence),
            'confidence_scores': confidence_scores,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Skin Type Analyzer API',
        'version': '1.0.0',
        'models_loaded': model is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze_skin():
    """Analyze skin type from uploaded image."""
    try:
        # Check if models are loaded
        if model is None:
            return jsonify({
                'error': 'Models not loaded',
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
            # Read image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Analyze the image
            result = process_image(image)
            
            logger.info(f"Analysis complete: {result['predicted_skin_type']} ({result['confidence']:.2f})")
            
            return jsonify(result)
            
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

@app.route('/analyze-base64', methods=['POST'])
def analyze_skin_base64():
    """Analyze skin type from base64 encoded image."""
    try:
        # Check if models are loaded
        if model is None:
            return jsonify({
                'error': 'Models not loaded',
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
            # Decode base64 image
            image_data = data['image']
            
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Analyze the image
            result = process_image(image)
            
            logger.info(f"Base64 analysis complete: {result['predicted_skin_type']} ({result['confidence']:.2f})")
            
            return jsonify(result)
            
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

# Initialize models when the app starts
if __name__ == '__main__':
    logger.info("🚀 Starting Skin Analyzer API...")
    
    # Load models
    if load_models():
        logger.info("✅ Models loaded successfully")
    else:
        logger.error("❌ Failed to load models")
    
    # Get port from environment variable (for Google Cloud Run)
    port = int(os.environ.get('PORT', 8080))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # For production WSGI servers
    load_models()
