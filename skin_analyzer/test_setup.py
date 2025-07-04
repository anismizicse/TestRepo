"""
Simple test script to verify the skin analyzer setup
Tests basic functionality without requiring trained models
"""

import os
import sys
import numpy as np

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError:
        print("✗ OpenCV not found - run: pip install opencv-python")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError:
        print("✗ TensorFlow not found - run: pip install tensorflow")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL imported successfully")
    except ImportError:
        print("✗ PIL not found - run: pip install Pillow")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError:
        print("✗ NumPy not found - run: pip install numpy")
        return False
    
    return True


def test_directory_structure():
    """Test if directory structure is correct"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data',
        'models',
        'utils',
        'models/saved_models'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/ directory exists")
        else:
            print(f"✗ {dir_path}/ directory missing")
            all_exist = False
    
    required_files = [
        'utils/image_processor.py',
        'utils/data_loader.py',
        'models/skin_classifier.py',
        'predict.py',
        'requirements.txt'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist


def test_basic_functionality():
    """Test basic functionality without requiring trained models"""
    print("\nTesting basic functionality...")
    
    try:
        # Test image processor
        sys.path.append('.')
        from utils.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        print("✓ ImageProcessor initialized successfully")
        
        # Test with synthetic image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed = processor.normalize_image(test_image)
        
        if processed.dtype == np.float32 and processed.max() <= 1.0:
            print("✓ Image normalization working correctly")
        else:
            print("✗ Image normalization failed")
            return False
        
        # Test skin classifier initialization
        from models.skin_classifier import SkinClassifier
        classifier = SkinClassifier()
        print("✓ SkinClassifier initialized successfully")
        
        # Test data loader
        from utils.data_loader import SkinDataLoader
        data_loader = SkinDataLoader('data')
        print("✓ SkinDataLoader initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {str(e)}")
        return False


def create_sample_data():
    """Create minimal sample data for testing"""
    print("\nCreating sample data...")
    
    try:
        from utils.data_loader import SkinDataLoader
        
        data_loader = SkinDataLoader('data')
        data_loader.create_dataset_structure()
        
        print("✓ Dataset structure created")
        
        # Create a few synthetic samples
        skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        
        for skin_type in skin_types:
            train_dir = os.path.join('data', 'train', skin_type)
            test_dir = os.path.join('data', 'test', skin_type)
            
            # Create 5 training and 2 test samples per type
            for i in range(5):
                synthetic_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                
                # Apply basic skin type characteristics
                if skin_type == 'oily':
                    synthetic_image[:, :, 1] = np.clip(synthetic_image[:, :, 1] + 30, 0, 255)
                elif skin_type == 'dry':
                    synthetic_image = np.clip(synthetic_image - 20, 0, 255)
                
                from PIL import Image
                image_path = os.path.join(train_dir, f'sample_{i:02d}.png')
                Image.fromarray(synthetic_image).save(image_path)
            
            for i in range(2):
                synthetic_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                
                if skin_type == 'oily':
                    synthetic_image[:, :, 1] = np.clip(synthetic_image[:, :, 1] + 30, 0, 255)
                elif skin_type == 'dry':
                    synthetic_image = np.clip(synthetic_image - 20, 0, 255)
                
                from PIL import Image
                image_path = os.path.join(test_dir, f'sample_{i:02d}.png')
                Image.fromarray(synthetic_image).save(image_path)
        
        print("✓ Sample data created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Sample data creation failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("SKIN ANALYZER SETUP TEST")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Directory structure
    if test_directory_structure():
        tests_passed += 1
    
    # Test 3: Basic functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Test 4: Sample data creation
    if create_sample_data():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("\n🎉 All tests passed! The skin analyzer is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train a model: python models/train_model.py")
        print("3. Make predictions: python predict.py --image path/to/image.jpg")
    else:
        print("\n⚠️  Some tests failed. Please check the requirements and setup.")
        print("\nTo install dependencies:")
        print("pip install -r requirements.txt")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    run_all_tests()
