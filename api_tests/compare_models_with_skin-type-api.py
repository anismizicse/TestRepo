# -*- coding: utf-8 -*-
"""
Comprehensive API vs Training Comparison
"""

import os
import sys
import torch
import pickle
from PIL import Image
from torchvision import transforms
import numpy as np

def compare_models_and_predictions():
    print("🔍 Comprehensive API vs Training Model Comparison")
    print("=" * 60)
    
    # Load both models
    try:
        api_model = torch.load('/home/anis/Desktop/My_Files/kaggle/TestRepo/skin-type-api/model/best_skin_model_entire.pth', 
                              map_location='cpu')
        train_model = torch.load('/home/anis/Desktop/My_Files/kaggle/TestRepo/trained_models_updated/best_skin_model_entire.pth', 
                                map_location='cpu')
        
        api_model.eval()
        train_model.eval()
        
        print("✅ Both models loaded successfully")
        
        # Compare model architectures
        api_params = sum(p.numel() for p in api_model.parameters())
        train_params = sum(p.numel() for p in train_model.parameters())
        
        print(f"📊 API model parameters: {api_params}")
        print(f"📊 Training model parameters: {train_params}")
        print(f"📊 Parameters match: {api_params == train_params}")
        
        # Compare a few key layer weights
        api_state = api_model.state_dict()
        train_state = train_model.state_dict()
        
        # Check final layer (most important for classification)
        fc_weight_match = torch.equal(api_state['fc.weight'], train_state['fc.weight'])
        fc_bias_match = torch.equal(api_state['fc.bias'], train_state['fc.bias'])
        
        print(f"🎯 Final layer weights match: {fc_weight_match}")
        print(f"🎯 Final layer bias match: {fc_bias_match}")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Load label mappings
    try:
        with open('/home/anis/Desktop/My_Files/kaggle/TestRepo/trained_models_updated/label_maps.pkl', 'rb') as f:
            label_maps = pickle.load(f)
        
        class_names = [label_maps['index_label'][i] for i in sorted(label_maps['index_label'].keys())]
        print(f"📋 Class names: {class_names}")
        
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return
    
    # Define exact same transforms as training
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test with a few images
    test_base = '/home/anis/Desktop/My_Files/kaggle/TestRepo/Oily-Dry-Skin-Types/test'
    
    if not os.path.exists(test_base):
        print(f"❌ Test directory not found: {test_base}")
        return
    
    print(f"\n🧪 Testing predictions on sample images...")
    print("-" * 60)
    
    total_tests = 0
    api_correct = 0
    train_correct = 0
    predictions_match = 0
    
    for class_name in ['dry', 'normal', 'oily']:
        class_dir = os.path.join(test_base, class_name)
        if not os.path.exists(class_dir):
            continue
            
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Test first 3 images from each class
        for img_name in images[:3]:
            img_path = os.path.join(class_dir, img_name)
            
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                img_tensor = val_transforms(image).unsqueeze(0)
                
                # API-style prediction (using the corrected logic)
                with torch.no_grad():
                    api_outputs = api_model(img_tensor)
                    api_probs = torch.softmax(api_outputs[0], dim=0)
                    api_pred_idx = api_probs.argmax().item()
                    api_pred_class = class_names[api_pred_idx]
                    api_confidence = api_probs[api_pred_idx].item()
                
                # Training-style prediction  
                with torch.no_grad():
                    train_outputs = train_model(img_tensor)
                    train_pred_idx = train_outputs.argmax(1).item()
                    train_pred_class = class_names[train_pred_idx]
                    train_probs = torch.softmax(train_outputs[0], dim=0)
                    train_confidence = train_probs[train_pred_idx].item()
                
                # Compare results
                total_tests += 1
                
                if api_pred_class == class_name:
                    api_correct += 1
                if train_pred_class == class_name:
                    train_correct += 1
                if api_pred_class == train_pred_class:
                    predictions_match += 1
                
                # Status indicators
                api_status = "✅" if api_pred_class == class_name else "❌"
                train_status = "✅" if train_pred_class == class_name else "❌"
                match_status = "✅" if api_pred_class == train_pred_class else "❌"
                
                print(f"{class_name:6s} | True: {class_name:6s} | API: {api_pred_class:6s} {api_status} ({api_confidence:.3f}) | Train: {train_pred_class:6s} {train_status} ({train_confidence:.3f}) | Match: {match_status}")
                
                # If predictions don't match, show raw outputs
                if api_pred_class != train_pred_class:
                    print(f"       Raw outputs - API: {api_outputs[0].numpy()}")
                    print(f"                     Train: {train_outputs[0].numpy()}")
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
    
    print("-" * 60)
    print(f"📊 SUMMARY (Total tests: {total_tests})")
    print(f"🎯 API Accuracy: {api_correct}/{total_tests} = {api_correct/total_tests*100:.1f}%")
    print(f"🎯 Training Accuracy: {train_correct}/{total_tests} = {train_correct/total_tests*100:.1f}%")
    print(f"🔍 Predictions Match: {predictions_match}/{total_tests} = {predictions_match/total_tests*100:.1f}%")
    
    if predictions_match == total_tests:
        print("✅ All predictions match - API is working correctly!")
    else:
        print("⚠️ Predictions don't match - investigating issue...")
        
        # Check if it's a model difference
        if fc_weight_match and fc_bias_match:
            print("🔍 Models are identical - issue might be elsewhere")
        else:
            print("❌ Models are different - this explains the discrepancy")

if __name__ == "__main__":
    compare_models_and_predictions()
