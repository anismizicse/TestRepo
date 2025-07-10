# -*- coding: utf-8 -*-
import gradio as gr
import torch
import torch.nn as nn
import pickle
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os

def load_model_and_labels():
    """Load model and label mappings"""
    # Load label mappings
    try:
        with open('label_maps.pkl', 'rb') as f:
            label_mappings = pickle.load(f)
    except:
        # Fallback if file not found
        label_mappings = {'index_label': {0: 'dry', 1: 'oily'}}
    
    # Initialize and load model
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load trained weights
    device = torch.device('cpu')
    try:
        state_dict = torch.load('best_skin_model.pth', map_location=device)
        model.load_state_dict(state_dict)
    except:
        print("Warning: Could not load model weights, using random initialization")
    
    model.eval()
    return model, label_mappings

def predict_skin_type(image):
    """Predict skin type from image"""
    if image is None:
        return "Please upload an image"
    
    try:
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get label
        predicted_label = label_mappings['index_label'][predicted_class]
        
        # Format result
        dry_prob = float(probabilities[0][0])
        oily_prob = float(probabilities[0][1])
        
        result = f"""Skin Type Prediction: {predicted_label.upper()}
Confidence: {confidence:.1%}

Detailed Probabilities:
- Dry Skin: {dry_prob:.1%}
- Oily Skin: {oily_prob:.1%}

Note: This is an AI prediction for educational purposes only."""
        
        return result
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Load model at startup
print("Loading model...")
model, label_mappings = load_model_and_labels()
print("Model loaded successfully!")

# Create interface
iface = gr.Interface(
    fn=predict_skin_type,
    inputs=gr.Image(type="pil", label="Upload facial skin image"),
    outputs=gr.Textbox(label="Prediction Results", lines=8),
    title="AI Skin Type Classifier",
    description="Upload a clear facial image to classify skin type as dry or oily."
)

# Launch
if __name__ == "__main__":
    iface.launch(share=False)
