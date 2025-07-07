#!/usr/bin/env python3
"""
Test script to verify the deployed API works correctly
"""

import requests
import json
from PIL import Image
import io
import base64

def test_api():
    """Test the deployed API"""
    base_url = "https://chatapplication-983c8.uc.r.appspot.com"
    
    print("Testing deployed skin type classification API...")
    
    # Test 1: Health check before prediction
    print("\n1. Testing health check (before model loading):")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Create a simple test image
    print("\n2. Creating test image...")
    # Create a simple 224x224 RGB image for testing
    test_image = Image.new('RGB', (224, 224), color='red')
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Test 3: API prediction
    print("\n3. Testing API prediction endpoint...")
    files = {'image': ('test.png', img_buffer, 'image/png')}
    
    try:
        response = requests.post(f"{base_url}/api/predict", files=files, timeout=60)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Prediction result:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Request timed out - this might happen on first prediction as model loads from Cloud Storage")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Health check after prediction
    print("\n4. Testing health check (after model loading):")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
