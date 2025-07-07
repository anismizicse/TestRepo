#!/usr/bin/env python3
"""
Test script to verify the skin type classification model works correctly
"""

import torch
import torch.nn as nn
import pickle
import json
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np

def test_model():
    """Test the model loading and prediction functionality"""
    print("🧪 Testing Skin Type Classification Model")
    print("=" * 50)
    
    try:
        # Load label mappings
        print("📁 Loading label mappings...")
        with open('label_maps.pkl', 'rb') as f:
            label_mappings = pickle.load(f)
        print(f"✅ Label mappings loaded: {label_mappings}")
        
        # Initialize model architecture
        print("🏗️ Initializing model architecture...")
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes
        
        # Load trained weights
        print("⚖️ Loading trained model weights...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {device}")
        
        model.load_state_dict(torch.load('best_skin_model.pth', map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create a dummy image for testing
        print("🖼️ Creating test image...")
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        # Preprocess
        input_tensor = transform(dummy_image).unsqueeze(0).to(device)
        
        # Make prediction
        print("🔮 Making prediction...")
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get label
        predicted_label = label_mappings['index_label'][predicted_class]
        
        result = {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "probabilities": {
                "dry": float(probabilities[0][0]),
                "oily": float(probabilities[0][1])
            },
            "class_index": predicted_class
        }
        
        print("🎯 Prediction Results:")
        print(json.dumps(result, indent=2))
        print("\n✅ Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 All tests passed! Model is ready for deployment.")
    else:
        print("\n💥 Tests failed! Please check the errors above.")
