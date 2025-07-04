"""
Simple Image Processing Test
Tests image processing without TensorFlow dependencies
"""

import os
import cv2
import numpy as np
from PIL import Image

def test_basic_opencv():
    """Test basic OpenCV functionality"""
    print("Testing basic OpenCV...")
    
    try:
        # Create a simple test image
        test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Gray image
        
        # Test basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(test_image, (112, 112))
        
        print(f"✅ Original image shape: {test_image.shape}")
        print(f"✅ Grayscale image shape: {gray.shape}")
        print(f"✅ Resized image shape: {resized.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {str(e)}")
        return False

def test_face_detection():
    """Test Haar cascade face detection"""
    print("Testing face detection...")
    
    try:
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create a simple test image with a face-like pattern
        test_image = np.ones((300, 300, 3), dtype=np.uint8) * 200
        
        # Add a simple rectangular face-like region
        cv2.rectangle(test_image, (100, 100), (200, 200), (180, 180, 180), -1)
        cv2.circle(test_image, (130, 140), 10, (100, 100, 100), -1)  # Left eye
        cv2.circle(test_image, (170, 140), 10, (100, 100, 100), -1)  # Right eye
        cv2.rectangle(test_image, (140, 170), (160, 180), (120, 120, 120), -1)  # Nose
        cv2.ellipse(test_image, (150, 190), (20, 10), 0, 0, 180, (100, 100, 100), -1)  # Mouth
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        print(f"✅ Face cascade loaded successfully")
        print(f"✅ Test image created: {test_image.shape}")
        print(f"✅ Face detection result: {len(faces)} face(s) detected")
        
        # Save test image
        os.makedirs('test_output', exist_ok=True)
        cv2.imwrite('test_output/test_face.jpg', cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        print("✅ Test image saved to: test_output/test_face.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {str(e)}")
        return False

def test_image_preprocessing():
    """Test image preprocessing pipeline"""
    print("Testing image preprocessing...")
    
    try:
        # Create sample image
        sample_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        
        # Preprocessing steps
        # 1. Resize
        resized = cv2.resize(sample_image, (224, 224))
        print(f"✅ Resized from {sample_image.shape} to {resized.shape}")
        
        # 2. Normalize
        normalized = resized.astype(np.float32) / 255.0
        print(f"✅ Normalized to range [0, 1]: min={normalized.min():.3f}, max={normalized.max():.3f}")
        
        # 3. Add augmentation
        # Horizontal flip
        flipped = cv2.flip(resized, 1)
        print(f"✅ Horizontal flip applied")
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(resized, alpha=1.2, beta=20)
        print(f"✅ Brightness adjustment applied")
        
        # Save examples
        os.makedirs('test_output', exist_ok=True)
        cv2.imwrite('test_output/original.jpg', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        cv2.imwrite('test_output/flipped.jpg', cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR))
        cv2.imwrite('test_output/bright.jpg', cv2.cvtColor(bright, cv2.COLOR_RGB2BGR))
        print("✅ Test images saved to test_output/")
        
        return True
        
    except Exception as e:
        print(f"❌ Image preprocessing test failed: {str(e)}")
        return False

def create_realistic_face_sample():
    """Create a more realistic face sample"""
    print("Creating realistic face sample...")
    
    try:
        # Create base face
        face_img = np.ones((224, 224, 3), dtype=np.uint8)
        
        # Skin tone variations for different types
        skin_tones = {
            'normal': (220, 180, 140),
            'dry': (200, 160, 120),
            'oily': (230, 190, 150),
            'combination': (215, 175, 135),
            'sensitive': (235, 185, 145)
        }
        
        os.makedirs('sample_faces', exist_ok=True)
        
        for skin_type, base_color in skin_tones.items():
            # Create face with specific skin characteristics
            face = np.ones((224, 224, 3), dtype=np.uint8)
            
            # Apply base skin color
            for i in range(3):
                face[:, :, i] = base_color[i]
            
            # Add skin type specific characteristics
            if skin_type == 'oily':
                # Add shine effect
                shine_mask = np.random.random((224, 224)) < 0.3
                for i in range(3):
                    face[:, :, i] = np.where(shine_mask, 
                                           np.clip(face[:, :, i] + 30, 0, 255), 
                                           face[:, :, i])
            
            elif skin_type == 'dry':
                # Add texture and reduce brightness
                texture = np.random.normal(0, 15, (224, 224))
                for i in range(3):
                    face[:, :, i] = np.clip(face[:, :, i] + texture, 0, 255)
                face = np.clip(face * 0.9, 0, 255).astype(np.uint8)
            
            elif skin_type == 'sensitive':
                # Add redness
                red_areas = np.random.random((224, 224)) < 0.2
                face[:, :, 0] = np.where(red_areas, 
                                       np.clip(face[:, :, 0] + 25, 0, 255), 
                                       face[:, :, 0])
            
            # Add face oval
            center_x, center_y = 112, 112
            for y in range(224):
                for x in range(224):
                    distance = ((x - center_x) / 90)**2 + ((y - center_y) / 110)**2
                    if distance > 1:
                        face[y, x] = [50, 50, 50]  # Background
            
            # Add basic facial features
            # Eyes
            cv2.circle(face, (85, 85), 8, (100, 100, 100), -1)
            cv2.circle(face, (140, 85), 8, (100, 100, 100), -1)
            
            # Nose
            cv2.ellipse(face, (112, 112), (8, 15), 0, 0, 360, (int(base_color[0]*0.9), int(base_color[1]*0.9), int(base_color[2]*0.9)), -1)
            
            # Mouth
            cv2.ellipse(face, (112, 140), (15, 8), 0, 0, 180, (150, 100, 100), -1)
            
            # Save sample
            sample_path = f'sample_faces/{skin_type}_sample.jpg'
            cv2.imwrite(sample_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            print(f"✅ {skin_type.capitalize()} skin sample saved to: {sample_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Realistic face sample creation failed: {str(e)}")
        return False

def main():
    """Run comprehensive image processing tests"""
    print("=" * 60)
    print("IMAGE PROCESSING TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Basic OpenCV", test_basic_opencv),
        ("Face Detection", test_face_detection),
        ("Image Preprocessing", test_image_preprocessing),
        ("Realistic Face Samples", create_realistic_face_sample)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"Status: {'✅ PASSED' if success else '❌ FAILED'}")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL IMAGE PROCESSING TESTS PASSED!")
        print("\nGenerated files:")
        print("- test_output/: Basic processing examples")
        print("- sample_faces/: Realistic skin type samples")
        print("\nYou can now proceed with model training!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
