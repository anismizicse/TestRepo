#!/usr/bin/env python3
"""
Test Script for Deployed Skin Analyzer API
==========================================

This script tests the deployed Google Cloud API to ensure it's working correctly.
"""

import requests
import base64
import json
from pathlib import Path

def test_health_check(api_url):
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{api_url}/")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Service: {data.get('service', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
            print(f"   Models loaded: {data.get('models_loaded', False)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_file_upload(api_url, image_path):
    """Test image analysis with file upload."""
    print("📤 Testing file upload analysis...")
    
    try:
        if not Path(image_path).exists():
            print(f"❌ Image file not found: {image_path}")
            return False
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{api_url}/analyze", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ File upload analysis successful")
            print(f"   Predicted skin type: {data.get('predicted_skin_type', 'Unknown')}")
            print(f"   Confidence: {data.get('confidence', 0):.2f}")
            print(f"   Status: {data.get('status', 'Unknown')}")
            return True
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return False

def test_base64_upload(api_url, image_path):
    """Test image analysis with base64 encoding."""
    print("🔄 Testing base64 analysis...")
    
    try:
        if not Path(image_path).exists():
            print(f"❌ Image file not found: {image_path}")
            return False
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {'image': image_data}
        response = requests.post(
            f"{api_url}/analyze-base64",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Base64 analysis successful")
            print(f"   Predicted skin type: {data.get('predicted_skin_type', 'Unknown')}")
            print(f"   Confidence: {data.get('confidence', 0):.2f}")
            print(f"   Status: {data.get('status', 'Unknown')}")
            return True
        else:
            print(f"❌ Base64 upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Base64 upload error: {e}")
        return False

def generate_curl_examples(api_url):
    """Generate cURL command examples."""
    print("\n📋 cURL Examples for Testing:")
    print("=" * 40)
    
    print("\n1. Health Check:")
    print(f"curl {api_url}/")
    
    print("\n2. File Upload:")
    print(f"curl -X POST -F 'image=@your_image.jpg' {api_url}/analyze")
    
    print("\n3. Base64 Upload:")
    print(f"""curl -X POST \\
  -H "Content-Type: application/json" \\
  -d '{{"image":"$(base64 -i your_image.jpg)"}}' \\
  {api_url}/analyze-base64""")

def main():
    """Main test function."""
    print("🧪 Google Cloud API Testing Script")
    print("=" * 40)
    
    # Get API URL from user
    api_url = input("Enter your Google Cloud API URL (without trailing slash): ").strip()
    
    if not api_url:
        print("❌ API URL is required!")
        return
    
    if api_url.endswith('/'):
        api_url = api_url[:-1]
    
    print(f"\n🎯 Testing API: {api_url}")
    print("-" * 40)
    
    # Test health check
    health_ok = test_health_check(api_url)
    
    if not health_ok:
        print("\n❌ Health check failed. Please check your deployment.")
        return
    
    # Look for test images
    test_images = [
        "sample_images/sample_face.jpg",
        "test_output/original.jpg",
        "training_dataset/train/normal/normal_skin_unsplash_0322790c.jpg"
    ]
    
    test_image = None
    for img_path in test_images:
        if Path(img_path).exists():
            test_image = img_path
            break
    
    if test_image:
        print(f"\n📸 Using test image: {test_image}")
        print("-" * 40)
        
        # Test file upload
        test_file_upload(api_url, test_image)
        print()
        
        # Test base64 upload
        test_base64_upload(api_url, test_image)
        
    else:
        print("\n⚠️ No test images found. Skipping image analysis tests.")
        print("Available test paths checked:")
        for img_path in test_images:
            print(f"   - {img_path}")
    
    # Generate cURL examples
    generate_curl_examples(api_url)
    
    print("\n🎉 Testing complete!")
    print("\n💡 Tips:")
    print("   - Test with different image types (JPG, PNG)")
    print("   - Monitor response times and accuracy")
    print("   - Check Google Cloud Console for logs and metrics")
    print("   - Set up monitoring alerts for production use")

if __name__ == "__main__":
    main()
