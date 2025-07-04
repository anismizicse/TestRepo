"""
Working Skin Analyzer Demo
Complete demonstration using the generated sample images
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import json

class SimpleSkinAnalyzer:
    """
    Simplified skin analyzer for demonstration
    """
    
    def __init__(self):
        self.skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        self.skin_characteristics = {
            'normal': {
                'description': 'Balanced oil production, good hydration',
                'characteristics': ['Even texture', 'Minimal pores', 'Good elasticity'],
                'care_tips': ['Use gentle cleanser', 'Moisturize daily', 'Apply sunscreen'],
                'color_range': [(200, 160, 120), (240, 200, 160)]
            },
            'dry': {
                'description': 'Low moisture, possible flaking',
                'characteristics': ['Tight feeling', 'Fine lines', 'Rough texture'],
                'care_tips': ['Use hydrating cleanser', 'Rich moisturizer', 'Avoid harsh products'],
                'color_range': [(180, 140, 100), (220, 180, 140)]
            },
            'oily': {
                'description': 'Excess sebum production, shine',
                'characteristics': ['Enlarged pores', 'Shine', 'Acne-prone'],
                'care_tips': ['Oil-free cleanser', 'Lightweight moisturizer', 'Salicylic acid'],
                'color_range': [(210, 170, 130), (250, 210, 170)]
            },
            'combination': {
                'description': 'Mixed characteristics (oily T-zone, dry cheeks)',
                'characteristics': ['Oily forehead/nose', 'Dry cheeks', 'Variable texture'],
                'care_tips': ['Dual-approach care', 'Different products for different areas'],
                'color_range': [(195, 155, 115), (235, 195, 155)]
            },
            'sensitive': {
                'description': 'Reactive, prone to irritation',
                'characteristics': ['Redness', 'Irritation-prone', 'Thin skin'],
                'care_tips': ['Gentle products', 'Fragrance-free', 'Patch testing'],
                'color_range': [(215, 175, 135), (255, 215, 175)]
            }
        }
    
    def detect_face_region(self, image):
        """Simple face detection"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Return the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                return True, largest_face
            return False, None
        except:
            return False, None
    
    def extract_skin_features(self, image):
        """Extract simple skin features from image"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate statistics
        features = {
            'mean_rgb': np.mean(image, axis=(0, 1)),
            'std_rgb': np.std(image, axis=(0, 1)),
            'mean_hsv': np.mean(hsv, axis=(0, 1)),
            'mean_lab': np.mean(lab, axis=(0, 1)),
            'brightness': np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)),
            'texture_variance': np.var(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        }
        
        return features
    
    def analyze_skin_type(self, image_path):
        """Analyze skin type from image"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path
            
            # Detect face
            face_detected, face_coords = self.detect_face_region(image)
            
            if face_detected:
                x, y, w, h = face_coords
                face_image = image[y:y+h, x:x+w]
            else:
                # Use center crop
                h, w = image.shape[:2]
                size = min(h, w)
                start_h = (h - size) // 2
                start_w = (w - size) // 2
                face_image = image[start_h:start_h + size, start_w:start_w + size]
            
            # Resize to standard size
            face_image = cv2.resize(face_image, (224, 224))
            
            # Extract features
            features = self.extract_skin_features(face_image)
            
            # Simple rule-based classification
            probabilities = self.classify_skin_features(features)
            
            # Get prediction
            predicted_class = max(probabilities.keys(), key=lambda k: probabilities[k])
            confidence = probabilities[predicted_class]
            
            result = {
                'skin_type': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'face_detected': face_detected,
                'features': features,
                'detailed_analysis': self.skin_characteristics[predicted_class]
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def classify_skin_features(self, features):
        """Simple rule-based classification"""
        probabilities = {}
        
        brightness = features['brightness']
        texture_var = features['texture_variance']
        mean_rgb = features['mean_rgb']
        
        # Simple heuristic classification
        for skin_type in self.skin_types:
            score = 0.2  # Base probability
            
            # Analyze based on brightness
            if skin_type == 'oily' and brightness > 150:
                score += 0.3  # Oily skin tends to be brighter
            elif skin_type == 'dry' and brightness < 130:
                score += 0.3  # Dry skin tends to be darker
            elif skin_type == 'normal' and 130 <= brightness <= 150:
                score += 0.3  # Normal skin in middle range
            
            # Analyze based on texture variance
            if skin_type == 'dry' and texture_var > 200:
                score += 0.2  # Dry skin has more texture variation
            elif skin_type == 'oily' and texture_var < 150:
                score += 0.2  # Oily skin smoother
            
            # Analyze based on color
            if skin_type == 'sensitive' and mean_rgb[0] > mean_rgb[1]:
                score += 0.2  # More red in sensitive skin
            
            # Add some randomness for demonstration
            score += np.random.uniform(0, 0.1)
            
            probabilities[skin_type] = score
        
        # Normalize probabilities
        total = sum(probabilities.values())
        probabilities = {k: v/total for k, v in probabilities.items()}
        
        return probabilities

def test_sample_images():
    """Test analysis on our generated sample images"""
    print("Testing skin analysis on sample images...")
    
    analyzer = SimpleSkinAnalyzer()
    sample_dir = 'sample_faces'
    
    if not os.path.exists(sample_dir):
        print(f"❌ Sample directory {sample_dir} not found. Run test_image_processing.py first.")
        return False
    
    results = []
    
    for filename in os.listdir(sample_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(sample_dir, filename)
            true_skin_type = filename.split('_')[0]  # Extract true type from filename
            
            print(f"\nAnalyzing: {filename}")
            result = analyzer.analyze_skin_type(image_path)
            
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
                continue
            
            predicted_type = result['skin_type']
            confidence = result['confidence']
            face_detected = result['face_detected']
            
            print(f"  True Type: {true_skin_type.upper()}")
            print(f"  Predicted: {predicted_type.upper()}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Face Detected: {'Yes' if face_detected else 'No'}")
            print(f"  Top 3 Probabilities:")
            
            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            for i, (skin_type, prob) in enumerate(sorted_probs[:3]):
                print(f"    {i+1}. {skin_type.capitalize()}: {prob:.2%}")
            
            # Check if prediction matches
            correct = predicted_type == true_skin_type
            print(f"  Result: {'✅ Correct' if correct else '❌ Incorrect'}")
            
            results.append({
                'filename': filename,
                'true_type': true_skin_type,
                'predicted_type': predicted_type,
                'confidence': confidence,
                'correct': correct,
                'full_result': result
            })
    
    return results

def generate_analysis_report(results):
    """Generate analysis report"""
    print("\n" + "="*60)
    print("SKIN ANALYSIS REPORT")
    print("="*60)
    
    if not results:
        print("No results to analyze.")
        return
    
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r['correct'])
    accuracy = correct_predictions / total_samples
    
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Per-class analysis
    print(f"\nPer-Class Results:")
    skin_types = set(r['true_type'] for r in results)
    
    for skin_type in sorted(skin_types):
        type_results = [r for r in results if r['true_type'] == skin_type]
        type_correct = sum(1 for r in type_results if r['correct'])
        type_accuracy = type_correct / len(type_results) if type_results else 0
        avg_confidence = np.mean([r['confidence'] for r in type_results]) if type_results else 0
        
        print(f"  {skin_type.capitalize()}: {type_accuracy:.2%} accuracy, {avg_confidence:.2%} avg confidence")
    
    # Save detailed report
    report_path = 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed report saved to: {report_path}")

def demonstrate_skin_care_recommendations():
    """Demonstrate skin care recommendations"""
    print("\n" + "="*60)
    print("SKIN CARE RECOMMENDATIONS DEMO")
    print("="*60)
    
    analyzer = SimpleSkinAnalyzer()
    
    for skin_type, info in analyzer.skin_characteristics.items():
        print(f"\n{skin_type.upper()} SKIN:")
        print(f"  Description: {info['description']}")
        print(f"  Key Characteristics:")
        for char in info['characteristics']:
            print(f"    • {char}")
        print(f"  Recommended Care:")
        for tip in info['care_tips']:
            print(f"    • {tip}")

def main():
    """Run the complete working demo"""
    print("="*60)
    print("WORKING SKIN ANALYZER DEMO")
    print("="*60)
    print("This demo uses rule-based analysis on the generated sample images.")
    print("")
    
    # Step 1: Test sample images
    print("Step 1: Analyzing Sample Images")
    print("-" * 40)
    results = test_sample_images()
    
    if results:
        # Step 2: Generate report
        generate_analysis_report(results)
        
        # Step 3: Demonstrate recommendations
        demonstrate_skin_care_recommendations()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✅ Image loading and preprocessing")
        print("✅ Face detection (when possible)")
        print("✅ Feature extraction from skin images")
        print("✅ Skin type classification")
        print("✅ Confidence scoring")
        print("✅ Detailed skin analysis")
        print("✅ Care recommendations")
        print("✅ Performance reporting")
        
        print(f"\nFiles Generated:")
        print(f"• sample_faces/ - Sample skin type images")
        print(f"• analysis_report.json - Detailed analysis results")
        
        print(f"\nNext Steps:")
        print(f"1. Install TensorFlow for deep learning models")
        print(f"2. Create larger dataset for training")
        print(f"3. Train CNN models for better accuracy")
        print(f"4. Deploy with web interface")
        
    else:
        print("❌ Demo failed. Please run test_image_processing.py first to generate sample images.")
    
    print("="*60)

if __name__ == "__main__":
    main()
