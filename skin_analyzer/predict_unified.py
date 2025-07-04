"""
Unified Skin Type Prediction Interface
Easy-to-use interface for making skin type predictions on new images
Supports both deep learning (TensorFlow/Keras) and traditional ML (scikit-learn) approaches
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import argparse
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.image_processor import ImageProcessor

# Try to import deep learning classifier
try:
    from models.skin_classifier import SkinClassifier
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Import ML classifier
from ml_skin_classifier import MLSkinClassifier


class UnifiedSkinTypePredictor:
    """
    Unified interface for skin type prediction
    Automatically chooses between deep learning and ML approaches based on availability
    """
    
    def __init__(self, model_path=None, model_type='auto', model_architecture='efficientnet'):
        """
        Initialize the skin type predictor
        
        Args:
            model_path (str): Path to the trained model file
            model_type (str): 'auto', 'deep_learning', 'ml', 'random_forest', 'gradient_boost', 'svm'
            model_architecture (str): Deep learning model architecture (if using TensorFlow)
        """
        self.model_architecture = model_architecture
        self.image_processor = ImageProcessor(target_size=(224, 224))
        self.model_type = model_type
        self.classifier = None
        self.is_ml_based = False
        
        # Determine which model type to use
        if model_type == 'auto':
            # Auto-select based on availability
            if TENSORFLOW_AVAILABLE and self._find_dl_model():
                self.model_type = 'deep_learning'
            else:
                self.model_type = 'ml'
        
        # Initialize the appropriate classifier
        self._initialize_classifier(model_path)
    
    def _initialize_classifier(self, model_path):
        """Initialize the appropriate classifier based on model type"""
        if self.model_type == 'deep_learning':
            if not TENSORFLOW_AVAILABLE:
                print("Warning: TensorFlow not available, falling back to ML approach")
                self.model_type = 'ml'
                self._initialize_ml_classifier(model_path)
            else:
                self._initialize_dl_classifier(model_path)
        else:
            self._initialize_ml_classifier(model_path)
    
    def _initialize_dl_classifier(self, model_path):
        """Initialize deep learning classifier"""
        self.classifier = SkinClassifier(base_model=self.model_architecture)
        self.is_ml_based = False
        
        # Load the trained model
        if model_path is None:
            model_path = self._find_dl_model()
        
        if model_path and os.path.exists(model_path):
            self.classifier.load_model(model_path)
            print(f"Deep learning model loaded from: {model_path}")
        else:
            print(f"Warning: No trained deep learning model found")
            print("Falling back to ML approach...")
            self._initialize_ml_classifier(None)
    
    def _initialize_ml_classifier(self, model_path):
        """Initialize ML classifier"""
        # Determine ML model type
        ml_type = 'random_forest'  # default
        if self.model_type in ['random_forest', 'gradient_boost', 'svm']:
            ml_type = self.model_type
        
        self.classifier = MLSkinClassifier(model_type=ml_type)
        self.is_ml_based = True
        
        # Try to load trained ML model
        if model_path is None:
            model_path = self._find_ml_model(ml_type)
        
        if model_path and os.path.exists(model_path):
            try:
                self.classifier.load_model(model_path)
                print(f"ML model ({ml_type}) loaded from: {model_path}")
            except Exception as e:
                print(f"Failed to load ML model: {e}")
                print("Please train an ML model first using: python ml_skin_classifier.py")
        else:
            print(f"Warning: No trained ML model found")
            print("Please train an ML model first using: python ml_skin_classifier.py")
    
    def _find_dl_model(self):
        """Find the best available trained deep learning model"""
        models_dir = os.path.join('models', 'saved_models')
        
        if not os.path.exists(models_dir):
            return None
        
        # Look for model files
        model_files = [
            f'skin_classifier_{self.model_architecture}.h5',
            f'fine_tuned_model.h5',
            'best_model.h5'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                return model_path
        
        # If no specific model found, get the first .h5 file
        try:
            h5_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
            if h5_files:
                return os.path.join(models_dir, h5_files[0])
        except:
            pass
        
        return None
    
    def _find_ml_model(self, ml_type='random_forest'):
        """Find the best available trained ML model"""
        # Look for ML model files in current directory
        model_files = [
            f'ml_skin_classifier_{ml_type}.pkl',
            f'{ml_type}_skin_classifier.joblib',
            f'ml_skin_classifier_{ml_type}.joblib',
            'ml_skin_classifier.joblib'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                return model_file
        
        # Also check models directory
        models_dir = 'models'
        if os.path.exists(models_dir):
            for model_file in model_files:
                model_path = os.path.join(models_dir, model_file)
                if os.path.exists(model_path):
                    return model_path
        
        return None
    
    def predict_image(self, image_path, return_probabilities=True):
        """
        Predict skin type for a single image
        
        Args:
            image_path (str): Path to the image file
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        try:
            if self.is_ml_based:
                # Use ML classifier
                if not self.classifier.is_trained:
                    return {
                        'error': 'ML model is not trained. Please train a model first.',
                        'face_detected': False
                    }
                
                result = self.classifier.predict(image_path)
                
                # Add face detection status
                try:
                    processed_image, face_detected = self.image_processor.process_image(image_path)
                    result['face_detected'] = face_detected
                except:
                    result['face_detected'] = False
                
                # Add confidence level description
                confidence = result['confidence']
                if confidence >= 0.8:
                    result['confidence_level'] = 'High'
                elif confidence >= 0.6:
                    result['confidence_level'] = 'Medium'
                else:
                    result['confidence_level'] = 'Low'
                
                return result
            
            else:
                # Use deep learning classifier
                # Process the image
                processed_image, face_detected = self.image_processor.process_image(image_path)
                
                if processed_image is None:
                    return {
                        'error': 'Failed to process image',
                        'face_detected': False
                    }
                
                # Make prediction
                result = self.classifier.predict_single_image(processed_image)
                result['face_detected'] = face_detected
                result['image_path'] = image_path
                
                # Add confidence level description
                confidence = result['confidence']
                if confidence >= 0.8:
                    result['confidence_level'] = 'High'
                elif confidence >= 0.6:
                    result['confidence_level'] = 'Medium'
                else:
                    result['confidence_level'] = 'Low'
                
                return result
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'image_path': image_path
            }
    
    def predict_batch(self, image_paths):
        """
        Predict skin types for multiple images
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append(result)
        
        return results
    
    def analyze_skin_characteristics(self, image_path):
        """
        Provide detailed skin analysis beyond just type classification
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Detailed skin analysis
        """
        # Get basic prediction
        result = self.predict_image(image_path)
        
        if 'error' in result:
            return result
        
        # Add detailed analysis based on skin type
        skin_type = result['skin_type']
        analysis = self._get_skin_analysis(skin_type, result['probabilities'])
        
        result['detailed_analysis'] = analysis
        
        return result
    
    def _get_skin_analysis(self, skin_type, probabilities):
        """
        Generate detailed skin analysis based on predicted type
        
        Args:
            skin_type (str): Predicted skin type
            probabilities (dict): Class probabilities
            
        Returns:
            dict: Detailed analysis
        """
        skin_analysis = {
            'normal': {
                'characteristics': ['Balanced oil production', 'Good hydration', 'Even texture', 'Minimal sensitivity'],
                'care_tips': ['Use gentle cleanser', 'Moisturize daily', 'Apply sunscreen', 'Regular exfoliation'],
                'concerns': ['Maintain current routine', 'Protect from environmental damage']
            },
            'dry': {
                'characteristics': ['Low oil production', 'Possible flaking', 'Tight feeling', 'Fine lines'],
                'care_tips': ['Use hydrating cleanser', 'Rich moisturizer', 'Hyaluronic acid', 'Avoid harsh products'],
                'concerns': ['Dehydration', 'Premature aging', 'Sensitivity to weather']
            },
            'oily': {
                'characteristics': ['Excess sebum', 'Shine', 'Large pores', 'Acne-prone'],
                'care_tips': ['Oil-free cleanser', 'Lightweight moisturizer', 'Salicylic acid', 'Clay masks'],
                'concerns': ['Acne breakouts', 'Clogged pores', 'Excessive shine']
            },
            'combination': {
                'characteristics': ['Oily T-zone', 'Dry cheeks', 'Mixed texture', 'Variable needs'],
                'care_tips': ['Dual-approach care', 'Different products for different areas', 'Balanced routine'],
                'concerns': ['Managing different zones', 'Finding suitable products']
            },
            'sensitive': {
                'characteristics': ['Reactive to products', 'Redness', 'Irritation-prone', 'Thin skin'],
                'care_tips': ['Gentle products', 'Fragrance-free', 'Patch testing', 'Minimal ingredients'],
                'concerns': ['Product reactions', 'Environmental sensitivity', 'Inflammation']
            }
        }
        
        analysis = skin_analysis.get(skin_type, skin_analysis['normal']).copy()
        
        # Add probability-based insights
        analysis['secondary_characteristics'] = []
        
        # Check for mixed characteristics
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_probs) > 1 and sorted_probs[1][1] > 0.2:
            secondary_type = sorted_probs[1][0]
            analysis['secondary_characteristics'].append(
                f"Shows some {secondary_type} skin characteristics ({sorted_probs[1][1]:.1%})")
        
        return analysis
    
    def save_prediction_report(self, results, output_path):
        """
        Save prediction results to a JSON report
        
        Args:
            results (dict or list): Prediction results
            output_path (str): Path to save the report
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Prediction report saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save report: {str(e)}")


def main():
    """
    Command-line interface for skin type prediction
    """
    parser = argparse.ArgumentParser(description='Unified Skin Type Analysis Tool')
    parser.add_argument('--image', '-i', type=str, help='Path to image file')
    parser.add_argument('--batch', '-b', type=str, nargs='+', help='Multiple image paths')
    parser.add_argument('--model', '-m', type=str, help='Path to model file')
    parser.add_argument('--model_type', '-t', type=str, default='auto',
                       choices=['auto', 'deep_learning', 'ml', 'random_forest', 'gradient_boost', 'svm'],
                       help='Model type to use')
    parser.add_argument('--architecture', '-a', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet50', 'mobilenet', 'custom'],
                       help='Deep learning model architecture')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed analysis')
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = UnifiedSkinTypePredictor(
        model_path=args.model,
        model_type=args.model_type,
        model_architecture=args.architecture
    )
    
    results = None
    
    if args.batch:
        # Batch prediction
        print(f"Analyzing {len(args.batch)} images...")
        if args.detailed:
            results = [predictor.analyze_skin_characteristics(img) for img in args.batch]
        else:
            results = predictor.predict_batch(args.batch)
        
    elif args.image:
        # Single image prediction
        print(f"Analyzing image: {args.image}")
        if args.detailed:
            results = predictor.analyze_skin_characteristics(args.image)
        else:
            results = predictor.predict_image(args.image)
    
    else:
        print("No input specified. Use --help for usage information.")
        return
    
    # Display results
    if results:
        print("\n" + "="*60)
        print("SKIN TYPE ANALYSIS RESULTS")
        print("="*60)
        
        if isinstance(results, list):
            # Multiple results
            for i, result in enumerate(results):
                print(f"\nImage {i+1}:")
                print_single_result(result)
        else:
            # Single result
            print_single_result(results)
        
        # Save results if requested
        if args.output:
            predictor.save_prediction_report(results, args.output)
    
    print("\nAnalysis completed!")


def print_single_result(result):
    """
    Print a single prediction result in a formatted way
    """
    if 'error' in result:
        print(f"  Error: {result['error']}")
        return
    
    print(f"  Skin Type: {result['skin_type'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%} ({result.get('confidence_level', 'Unknown')})")
    print(f"  Face Detected: {'Yes' if result.get('face_detected', False) else 'No'}")
    
    # Show probabilities
    if 'probabilities' in result:
        print("  Class Probabilities:")
        for class_name, prob in sorted(result['probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"    {class_name.capitalize()}: {prob:.2%}")
    
    # Show detailed analysis if available
    if 'detailed_analysis' in result:
        analysis = result['detailed_analysis']
        print("\n  Detailed Analysis:")
        print(f"    Characteristics: {', '.join(analysis.get('characteristics', []))}")
        print(f"    Care Tips: {', '.join(analysis.get('care_tips', []))}")
        print(f"    Main Concerns: {', '.join(analysis.get('concerns', []))}")
        
        if analysis.get('secondary_characteristics'):
            print(f"    Additional Notes: {', '.join(analysis['secondary_characteristics'])}")


if __name__ == "__main__":
    main()
