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

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        # Load label mappings
        with open('label_maps.pkl', 'rb') as f:
            label_mappings = pickle.load(f)
        
        # Initialize model architecture (ResNet50 with 2 output classes)
        model = resnet50(weights=None)  # Don't load pre-trained weights
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: dry, oily
        
        # Load trained model weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load('best_skin_model.pth', map_location=device))
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
        return False

def predict_image(image):
    """Predict skin type from image"""
    try:
        if model is None or transform is None:
            return {"error": "Model not loaded"}
        
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
        image = Image.open(file.stream)
        
        # Make prediction
        result = predict_image(image)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
        
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
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model_and_mappings():
        print("Starting Flask app...")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    else:
        print("Failed to load model. Exiting.")
