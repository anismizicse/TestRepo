#!/usr/bin/env python3
"""
Hugging Face API Endpoint Test Script
Tests the deployed skin type classification API with real images
"""

import requests
import os
import json
import time
from pathlib import Path
import glob
from PIL import Image
import io

# API Configuration
API_ENDPOINT = "https://anismizi-skin-type-api.hf.space/predict"
BASE_ENDPOINT = "https://anismizi-skin-type-api.hf.space"

def test_api_health():
    """Test if the API is healthy and running"""
    print("🔍 Testing API Health...")
    
    try:
        response = requests.get(BASE_ENDPOINT, timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API is healthy!")
            print(f"   Message: {health_data.get('message', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def predict_image(image_path, max_retries=3):
    """Send image to API for prediction"""
    
    for attempt in range(max_retries):
        try:
            # Prepare the image file
            with open(image_path, 'rb') as image_file:
                files = {
                    'file': (os.path.basename(image_path), image_file, 'image/jpeg')
                }
                
                # Send request to API
                response = requests.post(API_ENDPOINT, files=files, timeout=30)
                
            if response.status_code == 200:
                return response.json()
            else:
                print(f"   ⚠️ Attempt {attempt + 1} failed with status {response.status_code}")
                if attempt == max_retries - 1:
                    return {"error": f"API returned status {response.status_code}"}
                time.sleep(1)  # Wait before retry
                
        except Exception as e:
            print(f"   ⚠️ Attempt {attempt + 1} error: {e}")
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(1)  # Wait before retry
    
    return {"error": "Max retries exceeded"}

def test_single_image(image_path, expected_class=None):
    """Test a single image and return results"""
    
    print(f"📸 Testing: {os.path.basename(image_path)}")
    
    # Check if image exists and is valid
    if not os.path.exists(image_path):
        print(f"   ❌ Image not found: {image_path}")
        return None
    
    try:
        # Verify image can be opened
        with Image.open(image_path) as img:
            print(f"   📏 Image size: {img.size}, Mode: {img.mode}")
    except Exception as e:
        print(f"   ❌ Invalid image: {e}")
        return None
    
    # Get prediction from API
    start_time = time.time()
    result = predict_image(image_path)
    end_time = time.time()
    
    response_time = end_time - start_time
    
    if "error" in result:
        print(f"   ❌ Prediction failed: {result['error']}")
        return {
            "image": os.path.basename(image_path),
            "success": False,
            "error": result["error"],
            "response_time": response_time
        }
    
    # Parse successful result
    top_class = result.get("top_class", "unknown")
    predictions = result.get("predictions", [])
    
    print(f"   🎯 Predicted: {top_class}")
    print(f"   ⏱️ Response time: {response_time:.2f}s")
    
    # Show all predictions with confidence
    if predictions:
        print(f"   📊 All predictions:")
        for pred in predictions:
            confidence = pred.get("confidence", 0)
            class_name = pred.get("class", "unknown")
            print(f"      - {class_name}: {confidence:.3f} ({confidence*100:.1f}%)")
    
    # Check if prediction matches expected
    is_correct = None
    if expected_class:
        is_correct = (top_class == expected_class)
        status = "✅" if is_correct else "❌"
        print(f"   {status} Expected: {expected_class}, Got: {top_class}")
    
    return {
        "image": os.path.basename(image_path),
        "success": True,
        "predicted_class": top_class,
        "expected_class": expected_class,
        "correct": is_correct,
        "confidence": predictions[0].get("confidence", 0) if predictions else 0,
        "all_predictions": predictions,
        "response_time": response_time
    }

def test_batch_images(image_patterns, class_name=None):
    """Test multiple images from given patterns with enhanced real-world sampling"""
    
    print(f"\n📁 Testing {class_name or 'images'}...")
    
    all_images = []
    split_counts = {"test": 0, "valid": 0, "train": 0}
    
    for pattern in image_patterns:
        images = glob.glob(pattern)
        all_images.extend(images)
        
        # Track which split each image comes from for analysis
        for img in images:
            if "/test/" in img:
                split_counts["test"] += 1
            elif "/valid/" in img:
                split_counts["valid"] += 1
            elif "/train/" in img:
                split_counts["train"] += 1
    
    if not all_images:
        print(f"   ⚠️ No images found for patterns: {image_patterns}")
        return []
    
    print(f"   Found {len(all_images)} images across splits:")
    print(f"   📊 Test: {split_counts['test']}, Valid: {split_counts['valid']}, Train: {split_counts['train']}")
    
    # Sample intelligently from different splits for comprehensive testing
    test_images = []
    
    # Prioritize test and validation images (unseen during training)
    test_split_images = [img for img in all_images if "/test/" in img]
    valid_split_images = [img for img in all_images if "/valid/" in img]
    train_split_images = [img for img in all_images if "/train/" in img]
    
    # Take samples from each split for comprehensive evaluation
    test_images.extend(test_split_images[:8])  # Up to 8 from test split
    test_images.extend(valid_split_images[:8])  # Up to 8 from validation split
    test_images.extend(train_split_images[:4])  # Up to 4 from training split for comparison
    
    # If we don't have enough, take more from available images
    if len(test_images) < 15:
        remaining = [img for img in all_images if img not in test_images]
        test_images.extend(remaining[:15-len(test_images)])
    
    test_images = test_images[:20]  # Maximum 20 images per class
    
    print(f"   🎯 Testing {len(test_images)} carefully selected images")
    
    results = []
    correct_count = 0
    split_results = {"test": [], "valid": [], "train": []}
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n   [{i}/{len(test_images)}]", end=" ")
        
        # Determine which split this image is from
        split_type = "test" if "/test/" in image_path else ("valid" if "/valid/" in image_path else "train")
        
        result = test_single_image(image_path, class_name)
        
        if result:
            result["split"] = split_type  # Add split information
            results.append(result)
            split_results[split_type].append(result)
            
            if result.get("correct") is True:
                correct_count += 1
    
    # Enhanced summary with split-wise analysis
    if results:
        successful = len([r for r in results if r["success"]])
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results if r["success"]) / max(successful, 1)
        
        print(f"\n   📊 {class_name or 'Batch'} Summary:")
        print(f"      - Successful predictions: {successful}/{len(results)}")
        print(f"      - Average response time: {avg_response_time:.2f}s")
        print(f"      - Average confidence: {avg_confidence:.3f}")
        
        if class_name:
            accuracy = correct_count / successful if successful > 0 else 0
            print(f"      - Overall accuracy: {correct_count}/{successful} = {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Split-wise accuracy analysis
            for split_name, split_res in split_results.items():
                if split_res:
                    split_successful = len([r for r in split_res if r["success"]])
                    split_correct = len([r for r in split_res if r.get("correct") is True])
                    split_acc = split_correct / split_successful if split_successful > 0 else 0
                    print(f"      - {split_name.capitalize()} split accuracy: {split_correct}/{split_successful} = {split_acc:.3f} ({split_acc*100:.1f}%)")
    
    return results

def test_api_with_local_images():
    """Test API with real-world images from multiple dataset splits"""
    
    print("🧪 TESTING API WITH REAL-WORLD IMAGES")
    print("=" * 60)
    
    # Test with diverse real-world images from different splits for better coverage
    test_cases = [
        {
            "name": "dry",
            "patterns": [
                # Test split (original test set)
                "./Oily-Dry-Skin-Types/test/dry/*.jpg",
                "./Oily-Dry-Skin-Types/test/dry/*.jpeg",
                "./Oily-Dry-Skin-Types/test/dry/*.png",
                # Validation split (never used in training)
                "./Oily-Dry-Skin-Types/valid/dry/*.jpg",
                # Small sample from training split for comparison
                "./Oily-Dry-Skin-Types/train/dry/*.jpg"
            ]
        },
        {
            "name": "normal", 
            "patterns": [
                # Test split (original test set)
                "./Oily-Dry-Skin-Types/test/normal/*.jpg",
                "./Oily-Dry-Skin-Types/test/normal/*.jpeg",
                "./Oily-Dry-Skin-Types/test/normal/*.png",
                # Validation split (never used in training)
                "./Oily-Dry-Skin-Types/valid/normal/*.jpg",
                # Small sample from training split for comparison
                "./Oily-Dry-Skin-Types/train/normal/*.jpg"
            ]
        },
        {
            "name": "oily",
            "patterns": [
                # Test split (original test set)
                "./Oily-Dry-Skin-Types/test/oily/*.jpg",
                "./Oily-Dry-Skin-Types/test/oily/*.jpeg", 
                "./Oily-Dry-Skin-Types/test/oily/*.png",
                # Validation split (never used in training)
                "./Oily-Dry-Skin-Types/valid/oily/*.jpg",
                # Small sample from training split for comparison
                "./Oily-Dry-Skin-Types/train/oily/*.jpg"
            ]
        }
    ]
    
    all_results = []
    
    for test_case in test_cases:
        results = test_batch_images(test_case["patterns"], test_case["name"])
        all_results.extend(results)
    
    return all_results

def test_api_performance():
    """Test API performance metrics"""
    
    print("\n⚡ PERFORMANCE TESTING")
    print("=" * 40)
    
    # Find a test image
    test_patterns = [
        "./Oily-Dry-Skin-Types/test/*/*.jpg",
        "./Oily-Dry-Skin-Types/test/*/*.jpeg"
    ]
    
    test_image = None
    for pattern in test_patterns:
        images = glob.glob(pattern)
        if images:
            test_image = images[0]
            break
    
    if not test_image:
        print("❌ No test images found for performance testing")
        return
    
    print(f"🔄 Performance testing with: {os.path.basename(test_image)}")
    
    # Test multiple requests
    response_times = []
    successful_requests = 0
    
    for i in range(5):
        print(f"   Request {i+1}/5...", end=" ")
        start_time = time.time()
        result = predict_image(test_image)
        end_time = time.time()
        
        response_time = end_time - start_time
        response_times.append(response_time)
        
        if "error" not in result:
            successful_requests += 1
            print(f"✅ {response_time:.2f}s")
        else:
            print(f"❌ {response_time:.2f}s - {result.get('error', 'Unknown error')}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n📊 Performance Results:")
        print(f"   - Success rate: {successful_requests}/5 ({successful_requests/5*100:.0f}%)")
        print(f"   - Average response: {avg_time:.2f}s")
        print(f"   - Fastest response: {min_time:.2f}s")
        print(f"   - Slowest response: {max_time:.2f}s")

def generate_test_report(results):
    """Generate a comprehensive test report with split-wise analysis"""
    
    print("\n📋 COMPREHENSIVE TEST REPORT")
    print("=" * 50)
    
    if not results:
        print("❌ No test results to analyze")
        return
    
    # Overall statistics
    total_tests = len(results)
    successful_tests = len([r for r in results if r["success"]])
    failed_tests = total_tests - successful_tests
    
    print(f"📊 Overall Statistics:")
    print(f"   - Total tests: {total_tests}")
    print(f"   - Successful: {successful_tests}")
    print(f"   - Failed: {failed_tests}")
    print(f"   - Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests == 0:
        print("❌ No successful tests to analyze further")
        return
    
    # Performance metrics
    successful_results = [r for r in results if r["success"]]
    avg_response_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
    avg_confidence = sum(r["confidence"] for r in successful_results) / len(successful_results)
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   - Average response time: {avg_response_time:.2f}s")
    print(f"   - Average confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
    
    # Split-wise analysis
    print(f"\n🔍 Dataset Split Analysis:")
    split_stats = {}
    for split_name in ["test", "valid", "train"]:
        split_results = [r for r in successful_results if r.get("split") == split_name]
        if split_results:
            split_correct = len([r for r in split_results if r.get("correct") is True])
            split_accuracy = split_correct / len(split_results) if split_results else 0
            split_stats[split_name] = {
                "count": len(split_results),
                "correct": split_correct,
                "accuracy": split_accuracy
            }
            print(f"   - {split_name.capitalize()} split: {split_correct}/{len(split_results)} = {split_accuracy:.3f} ({split_accuracy*100:.1f}%)")
    
    # Accuracy analysis (only for tests with expected classes)
    accuracy_results = [r for r in successful_results if r["correct"] is not None]
    if accuracy_results:
        correct_predictions = len([r for r in accuracy_results if r["correct"]])
        accuracy = correct_predictions / len(accuracy_results)
        
        print(f"\n🎯 Accuracy Analysis:")
        print(f"   - Correct predictions: {correct_predictions}/{len(accuracy_results)}")
        print(f"   - Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Per-class accuracy
        for class_name in ["dry", "normal", "oily"]:
            class_results = [r for r in accuracy_results if r["expected_class"] == class_name]
            if class_results:
                class_correct = len([r for r in class_results if r["correct"]])
                class_accuracy = class_correct / len(class_results)
                print(f"   - {class_name} accuracy: {class_correct}/{len(class_results)} = {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
                
                # Per-class per-split accuracy
                for split_name in ["test", "valid", "train"]:
                    split_class_results = [r for r in class_results if r.get("split") == split_name]
                    if split_class_results:
                        split_class_correct = len([r for r in split_class_results if r["correct"]])
                        split_class_acc = split_class_correct / len(split_class_results)
                        print(f"     • {split_name} split: {split_class_correct}/{len(split_class_results)} = {split_class_acc:.3f} ({split_class_acc*100:.1f}%)")
    
    # Top predictions analysis
    print(f"\n🏆 Prediction Distribution:")
    prediction_counts = {}
    for result in successful_results:
        pred_class = result["predicted_class"]
        prediction_counts[pred_class] = prediction_counts.get(pred_class, 0) + 1
    
    for class_name, count in sorted(prediction_counts.items()):
        percentage = count / successful_tests * 100
        print(f"   - {class_name}: {count} predictions ({percentage:.1f}%)")
    
    # Key insights
    print(f"\n💡 Key Insights:")
    if split_stats:
        best_split = max(split_stats.keys(), key=lambda x: split_stats[x]["accuracy"])
        worst_split = min(split_stats.keys(), key=lambda x: split_stats[x]["accuracy"])
        print(f"   - Best performing split: {best_split} ({split_stats[best_split]['accuracy']*100:.1f}%)")
        print(f"   - Most challenging split: {worst_split} ({split_stats[worst_split]['accuracy']*100:.1f}%)")
        
        if "valid" in split_stats and "train" in split_stats:
            val_acc = split_stats["valid"]["accuracy"]
            train_acc = split_stats["train"]["accuracy"] 
            if val_acc > train_acc * 0.9:
                print(f"   - ✅ Model generalizes well (validation performance close to training)")
            else:
                print(f"   - ⚠️ Potential overfitting detected (validation << training performance)")
    
    print(f"   - API shows {'high' if avg_confidence > 0.8 else 'moderate' if avg_confidence > 0.6 else 'low'} confidence overall")
    print(f"   - Response time is {'excellent' if avg_response_time < 2.0 else 'good' if avg_response_time < 3.0 else 'acceptable'}")

def main():
    """Main test function"""
    
    print("🚀 HUGGING FACE API ENDPOINT TEST")
    print("=" * 60)
    print(f"🔗 API Endpoint: {API_ENDPOINT}")
    print(f"📅 Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Health check (skip if not available)
    health_ok = test_api_health()
    if not health_ok:
        print("⚠️ Health check endpoint not available, but continuing with main API tests...")
        print()
    
    # Step 2: Test with local images
    results = test_api_with_local_images()
    
    # Step 3: Performance testing
    test_api_performance()
    
    # Step 4: Generate comprehensive report
    generate_test_report(results)
    
    print("\n" + "=" * 60)
    print("✅ API testing completed!")
    print(f"🔗 Your API is deployed at: {API_ENDPOINT}")
    print("=" * 60)

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir('/home/anis/Desktop/My_Files/kaggle/TestRepo')
    
    # Run the tests
    main()
