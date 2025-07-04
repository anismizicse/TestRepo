#!/usr/bin/env python3
"""
Comprehensive Test of the Unified Skin Type Prediction System
Tests both ML-based prediction and detailed analysis capabilities
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_unified import UnifiedSkinTypePredictor, print_single_result


def test_single_image_prediction():
    """Test single image prediction with different models"""
    print("🔍 TESTING SINGLE IMAGE PREDICTION")
    print("=" * 60)
    
    test_image = "sample_images/sample_face.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    model_types = ['random_forest', 'gradient_boost', 'svm']
    
    for model_type in model_types:
        print(f"\n📊 Testing with {model_type.upper()} model:")
        print("-" * 40)
        
        try:
            predictor = UnifiedSkinTypePredictor(model_type=model_type)
            result = predictor.predict_image(test_image)
            print_single_result(result)
            print("✅ Success!")
            
        except Exception as e:
            print(f"❌ Error with {model_type}: {str(e)}")
            return False
    
    return True


def test_detailed_analysis():
    """Test detailed skin analysis"""
    print("\n🔬 TESTING DETAILED SKIN ANALYSIS")
    print("=" * 60)
    
    test_image = "sample_images/sample_face.jpg"
    
    try:
        predictor = UnifiedSkinTypePredictor(model_type='random_forest')
        result = predictor.analyze_skin_characteristics(test_image)
        
        print(f"\nDetailed Analysis for: {test_image}")
        print_single_result(result)
        print("✅ Detailed analysis successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in detailed analysis: {str(e)}")
        return False


def test_batch_prediction():
    """Test batch prediction with multiple images"""
    print("\n📚 TESTING BATCH PREDICTION")
    print("=" * 60)
    
    # Test with synthetic dataset images
    test_images = []
    skin_types = ['normal', 'oily', 'dry', 'combination', 'sensitive']
    
    for skin_type in skin_types:
        test_file = f"data/test/{skin_type}/{skin_type}_test_0000.jpg"
        if os.path.exists(test_file):
            test_images.append(test_file)
    
    if len(test_images) == 0:
        print("❌ No test images found in data/test/ directories")
        return False
    
    print(f"Testing with {len(test_images)} images from different skin types...")
    
    try:
        predictor = UnifiedSkinTypePredictor(model_type='random_forest')
        results = predictor.predict_batch(test_images)
        
        print(f"\nBatch Prediction Results:")
        for i, result in enumerate(results):
            print(f"\nImage {i+1}: {os.path.basename(test_images[i])}")
            print_single_result(result)
        
        print("✅ Batch prediction successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {str(e)}")
        return False


def test_model_comparison():
    """Compare predictions across different models"""
    print("\n⚖️  TESTING MODEL COMPARISON")
    print("=" * 60)
    
    test_image = "sample_images/sample_face.jpg"
    model_types = ['random_forest', 'gradient_boost', 'svm']
    
    results = {}
    
    try:
        for model_type in model_types:
            predictor = UnifiedSkinTypePredictor(model_type=model_type)
            result = predictor.predict_image(test_image)
            results[model_type] = result
        
        print(f"\nModel Comparison for: {os.path.basename(test_image)}")
        print("-" * 50)
        
        for model_type, result in results.items():
            if 'error' not in result:
                print(f"\n{model_type.upper()}: {result['skin_type']} ({result['confidence']:.1%})")
                # Show top 3 probabilities
                sorted_probs = sorted(result['probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                for skin_type, prob in sorted_probs:
                    print(f"  {skin_type}: {prob:.1%}")
            else:
                print(f"\n{model_type.upper()}: Error - {result['error']}")
        
        print("\n✅ Model comparison successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in model comparison: {str(e)}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 TESTING EDGE CASES")
    print("=" * 60)
    
    try:
        predictor = UnifiedSkinTypePredictor(model_type='random_forest')
        
        # Test with non-existent image
        print("Testing with non-existent image...")
        result = predictor.predict_image("non_existent_image.jpg")
        if 'error' in result:
            print("✅ Correctly handled non-existent image")
        else:
            print("❌ Should have returned error for non-existent image")
            return False
        
        # Test with empty batch
        print("\nTesting with empty batch...")
        results = predictor.predict_batch([])
        if len(results) == 0:
            print("✅ Correctly handled empty batch")
        else:
            print("❌ Should have returned empty results for empty batch")
            return False
        
        print("\n✅ Edge case testing successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in edge case testing: {str(e)}")
        return False


def save_test_report(test_results):
    """Save test results to a report file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"unified_prediction_test_report_{timestamp}.json"
    
    report = {
        "timestamp": timestamp,
        "test_results": test_results,
        "summary": {
            "total_tests": len(test_results),
            "passed_tests": sum(1 for result in test_results.values() if result),
            "failed_tests": sum(1 for result in test_results.values() if not result)
        }
    }
    
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Test report saved to: {report_file}")
    except Exception as e:
        print(f"❌ Failed to save test report: {str(e)}")


def main():
    """Run comprehensive tests of the unified prediction system"""
    print("🚀 COMPREHENSIVE UNIFIED SKIN TYPE PREDICTION TEST")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run all tests
    test_functions = [
        ("Single Image Prediction", test_single_image_prediction),
        ("Detailed Analysis", test_detailed_analysis),
        ("Batch Prediction", test_batch_prediction),
        ("Model Comparison", test_model_comparison),
        ("Edge Cases", test_edge_cases)
    ]
    
    test_results = {}
    
    for test_name, test_func in test_functions:
        print(f"\n{'='*70}")
        try:
            success = test_func()
            test_results[test_name] = success
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {str(e)}")
            test_results[test_name] = False
    
    # Print summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The unified prediction system is working perfectly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    # Save test report
    save_test_report(test_results)
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
