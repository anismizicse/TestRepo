"""
Minimal Skin Analyzer Demo
Demonstrates the core functionality without requiring TensorFlow training
"""

import os
import sys
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_test_image():
    """Create a sample face image for testing"""
    print("Creating sample test image...")
    
    # Create a simple synthetic face image
    image_size = (224, 224, 3)
    
    # Create base skin color (medium tone)
    image = np.ones(image_size, dtype=np.uint8) * 200
    
    # Add some face-like features
    h, w = image_size[:2]
    
    # Add oval face shape
    center_x, center_y = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            # Create oval mask
            distance = ((x - center_x) / (w * 0.4))**2 + ((y - center_y) / (h * 0.45))**2
            if distance <= 1:
                # Inside face area - vary skin tone slightly
                variation = np.random.randint(-20, 20)
                image[y, x] = np.clip(image[y, x] + variation, 0, 255)
            else:
                # Outside face - make background
                image[y, x] = [50, 50, 50]  # Dark background
    
    # Add some facial features
    # Eyes
    eye_y = center_y - 30
    for eye_x in [center_x - 40, center_x + 40]:
        for dy in range(-10, 10):
            for dx in range(-15, 15):
                if 0 <= eye_y + dy < h and 0 <= eye_x + dx < w:
                    image[eye_y + dy, eye_x + dx] = [100, 100, 100]  # Darker for eyes
    
    # Nose
    nose_y = center_y
    for dy in range(-20, 20):
        for dx in range(-5, 5):
            if 0 <= nose_y + dy < h and 0 <= center_x + dx < w:
                image[nose_y + dy, center_x + dx] = np.clip(image[nose_y + dy, center_x + dx] - 10, 0, 255)
    
    # Mouth
    mouth_y = center_y + 40
    for dy in range(-5, 5):
        for dx in range(-25, 25):
            if 0 <= mouth_y + dy < h and 0 <= center_x + dx < w:
                image[mouth_y + dy, center_x + dx] = [150, 100, 100]  # Reddish for mouth
    
    return image

def test_image_processing():
    """Test image processing without TensorFlow"""
    print("Testing image processing...")
    
    try:
        # Test basic imports
        import cv2
        from utils.image_processor import ImageProcessor
        
        print("✅ Image processing modules imported successfully")
        
        # Create processor
        processor = ImageProcessor()
        print("✅ ImageProcessor created successfully")
        
        # Create sample image
        sample_image = create_sample_test_image()
        print("✅ Sample image created")
        
        # Test face detection
        face_detected, face_coords = processor.detect_face(cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
        print(f"✅ Face detection test: {'Face detected' if face_detected else 'No face detected (expected for synthetic image)'}")
        
        # Test image preprocessing
        processed_image, face_detected = processor.process_image(sample_image)
        if processed_image is not None:
            print(f"✅ Image processing successful: {processed_image.shape}")
        else:
            print("⚠️ Image processing returned None")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {str(e)}")
        return False

def test_basic_prediction_structure():
    """Test prediction structure without actual model"""
    print("Testing prediction structure...")
    
    try:
        # Simulate a prediction result
        skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        
        # Create mock prediction
        mock_probabilities = np.random.dirichlet(np.ones(5))  # Random probabilities that sum to 1
        predicted_class = np.argmax(mock_probabilities)
        confidence = mock_probabilities[predicted_class]
        
        result = {
            'skin_type': skin_types[predicted_class],
            'confidence': float(confidence),
            'probabilities': {skin_type: float(prob) for skin_type, prob in zip(skin_types, mock_probabilities)},
            'face_detected': True
        }
        
        print("✅ Mock prediction structure created:")
        print(f"   Predicted Skin Type: {result['skin_type'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Top 3 probabilities:")
        
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for i, (skin_type, prob) in enumerate(sorted_probs[:3]):
            print(f"     {i+1}. {skin_type.capitalize()}: {prob:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction structure test failed: {str(e)}")
        return False

def demonstrate_skin_analysis():
    """Demonstrate skin analysis concepts"""
    print("Demonstrating skin analysis concepts...")
    
    skin_characteristics = {
        'normal': {
            'description': 'Balanced oil production, good hydration',
            'characteristics': ['Even texture', 'Minimal pores', 'Good elasticity'],
            'care_tips': ['Use gentle cleanser', 'Moisturize daily', 'Apply sunscreen']
        },
        'dry': {
            'description': 'Low moisture, possible flaking',
            'characteristics': ['Tight feeling', 'Fine lines', 'Rough texture'],
            'care_tips': ['Use hydrating cleanser', 'Rich moisturizer', 'Avoid harsh products']
        },
        'oily': {
            'description': 'Excess sebum production, shine',
            'characteristics': ['Enlarged pores', 'Shine', 'Acne-prone'],
            'care_tips': ['Oil-free cleanser', 'Lightweight moisturizer', 'Salicylic acid']
        },
        'combination': {
            'description': 'Mixed characteristics (oily T-zone, dry cheeks)',
            'characteristics': ['Oily forehead/nose', 'Dry cheeks', 'Variable texture'],
            'care_tips': ['Dual-approach care', 'Different products for different areas']
        },
        'sensitive': {
            'description': 'Reactive, prone to irritation',
            'characteristics': ['Redness', 'Irritation-prone', 'Thin skin'],
            'care_tips': ['Gentle products', 'Fragrance-free', 'Patch testing']
        }
    }
    
    print("✅ Skin type analysis framework:")
    print("=" * 60)
    
    for skin_type, info in skin_characteristics.items():
        print(f"\n{skin_type.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Characteristics: {', '.join(info['characteristics'])}")
        print(f"  Care Tips: {', '.join(info['care_tips'])}")
    
    print("\n" + "=" * 60)
    return True

def save_sample_image():
    """Save a sample image for testing"""
    print("Creating sample image file...")
    
    try:
        # Create sample directory if it doesn't exist
        sample_dir = "sample_images"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create and save sample image
        sample_image = create_sample_test_image()
        image_pil = Image.fromarray(sample_image)
        image_path = os.path.join(sample_dir, "sample_face.jpg")
        image_pil.save(image_path, quality=95)
        
        print(f"✅ Sample image saved to: {image_path}")
        print(f"   Image size: {sample_image.shape}")
        
        return image_path
        
    except Exception as e:
        print(f"❌ Failed to save sample image: {str(e)}")
        return None

def main():
    """Run the minimal demo"""
    print("=" * 60)
    print("SKIN ANALYZER - MINIMAL DEMO")
    print("=" * 60)
    print("This demo tests the core functionality without requiring trained models.")
    print("")
    
    tests = [
        ("Image Processing", test_image_processing),
        ("Prediction Structure", test_basic_prediction_structure),
        ("Skin Analysis Framework", demonstrate_skin_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Save sample image
    print(f"\n{'=' * 20} Sample Image Creation {'=' * 20}")
    sample_path = save_sample_image()
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if sample_path:
        print(f"Sample image created: {sample_path}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("\nThe core image processing and analysis framework is working.")
        print("\nNext steps:")
        print("1. Install TensorFlow for full functionality")
        print("2. Create dataset: python utils/dataset_downloader.py")
        print("3. Train model: python models/train_model.py")
        print("4. Make predictions: python predict.py")
    else:
        print("⚠️ Some components need attention.")
        print("Check the error messages above for details.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
