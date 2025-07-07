import os
import torch
import torch.nn as nn
import pickle
import json
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
import io
import base64
import urllib.request
from google.cloud import storage

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables for model and label mappings
model = None
label_mappings = None
transform = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_mappings():
    """Load the trained model and label mappings"""
    global model, label_mappings, transform
    
    try:
        # Check if already loaded
        if model is not None and label_mappings is not None and transform is not None:
            return True
            
        print("Loading model and mappings...")
        
        # Download model files from Cloud Storage if they don't exist locally
        bucket_name = "chatapplication-983c8-models"
        
        def download_from_gcs(filename):
            try:
                # Use /tmp directory for downloaded files (App Engine writable directory)
                tmp_path = f"/tmp/{filename}"
                
                # Check if file already exists
                if os.path.exists(tmp_path):
                    print(f"File {filename} already exists at {tmp_path}")
                    return tmp_path
                
                # Try using Google Cloud Storage client first (better for App Engine)
                try:
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(filename)
                    blob.download_to_filename(tmp_path)
                    print(f"Successfully downloaded {filename} using GCS client to {tmp_path}")
                    return tmp_path
                except Exception as gcs_error:
                    print(f"GCS client failed for {filename}: {gcs_error}, trying HTTP...")
                    # Fallback to HTTP download
                    url = f"https://storage.googleapis.com/{bucket_name}/{filename}"
                    urllib.request.urlretrieve(url, tmp_path)
                    print(f"Successfully downloaded {filename} using HTTP to {tmp_path}")
                    return tmp_path
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                return None
        
        # Try to download files if they don't exist locally
        label_maps_path = '/tmp/label_maps.pkl'
        model_path = '/tmp/best_skin_model.pth'
        
        # Download label maps if needed
        if not os.path.exists(label_maps_path):
            print("Downloading label_maps.pkl from Cloud Storage...")
            downloaded_path = download_from_gcs('label_maps.pkl')
            if not downloaded_path:
                return False
            label_maps_path = downloaded_path
        
        # Download model if needed
        if not os.path.exists(model_path):
            print("Downloading best_skin_model.pth from Cloud Storage...")
            downloaded_path = download_from_gcs('best_skin_model.pth')
            if not downloaded_path:
                return False
            model_path = downloaded_path
        
        print("Loading label mappings...")
        # Load label mappings
        with open(label_maps_path, 'rb') as f:
            label_mappings = pickle.load(f)
        print(f"Label mappings loaded: {label_mappings}")
        
        print("Initializing model architecture...")
        # Initialize model architecture (ResNet50 with 2 output classes)
        model = resnet50(weights=None)  # Don't load pre-trained weights
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: dry, oily
        
        print("Loading trained model weights...")
        # Load trained model weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Define image preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Model and mappings loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def predict_image(image):
    """Predict skin type from image"""
    global model, transform
    
    try:
        # Auto-load model if not loaded
        if model is None or transform is None:
            print("Model not loaded, attempting to load...")
            if not load_model_and_mappings():
                return {"error": "Model not loaded. Failed to load the trained model files."}
        
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get label name
        predicted_label = label_mappings['index_label'][predicted_class]
        
        # Create detailed prediction result
        result = {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "probabilities": {
                "dry": float(probabilities[0][0]),
                "oily": float(probabilities[0][1])
            },
            "class_index": predicted_class
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for image prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
        
        # Load and process image
        try:
            image = Image.open(file.stream)
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Make prediction with timeout handling
        try:
            result = predict_image(image)
            
            if 'error' in result:
                if 'Model not loaded' in result['error']:
                    return jsonify({
                        'error': 'Model is loading. Please try again in a few seconds.',
                        'retry_after': 10
                    }), 503
                return jsonify(result), 500
            
            return jsonify({
                'success': True,
                'result': result
            })
        except Exception as pred_error:
            return jsonify({
                'error': f'Prediction failed: {str(pred_error)}',
                'retry_after': 5
            }), 500
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def web_predict():
    """Web interface for image prediction"""
    try:
        if 'image' not in request.files:
            return render_template('index.html', error='No image file provided')
        
        file = request.files['image']
        
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if not allowed_file(file.filename):
            return render_template('index.html', error='Invalid file type. Please upload png, jpg, jpeg, gif, or bmp files.')
        
        # Load and process image
        image = Image.open(file.stream)
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Make prediction
        result = predict_image(image)
        
        if 'error' in result:
            return render_template('index.html', error=result['error'])
        
        return render_template('index.html', 
                             result=result, 
                             image_data=img_str,
                             success=True)
        
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None and transform is not None and label_mappings is not None,
            'version': '1.0.0',
            'message': 'Model loads lazily on first prediction request' if model is None else 'Model ready'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'model_loaded': False,
            'version': '1.0.0',
            'error': str(e)
        }), 500

@app.route('/warmup', methods=['GET', 'POST'])
def warmup():
    """Warmup endpoint to pre-load the model"""
    try:
        if model is not None and transform is not None and label_mappings is not None:
            return jsonify({
                'status': 'success',
                'message': 'Model already loaded',
                'model_loaded': True
            })
        
        print("Warmup: Starting model loading...")
        success = load_model_and_mappings()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model loaded successfully',
                'model_loaded': True
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to load model',
                'model_loaded': False
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Warmup failed: {str(e)}',
            'model_loaded': False
        }), 500

# Don't load model on startup to avoid App Engine timeout issues
# Model will be loaded lazily on first prediction request
print("App initialized. Model will be loaded on first prediction request.")

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
