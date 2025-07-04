"""
Skin Type Prediction Interface
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


class SkinTypePredictor:
    """
    Main interface for skin type prediction
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
        
        # Load label encoder if available (for backward compatibility)
        if not self.is_ml_based:
            self._load_label_encoder()
    
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
            print(f"Warning: No trained ML model found at {model_path}")
            print("Please train an ML model first using: python ml_skin_classifier.py")
    
    def _find_dl_model(self):
        """
        Find the best available trained deep learning model
        """
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
        h5_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        if h5_files:
            return os.path.join(models_dir, h5_files[0])
        
        return None
    
    def _find_ml_model(self, ml_type='random_forest'):
        """
        Find the best available trained ML model
        """
        models_dir = 'models'
        
        if not os.path.exists(models_dir):
            return None
        
        # Look for ML model files
        model_files = [
            f'{ml_type}_skin_classifier.joblib',
            f'ml_skin_classifier_{ml_type}.joblib',
            'ml_skin_classifier.joblib'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                return model_path
        
        return None
    
    def _find_best_model(self):
        """
        Find the best available trained model (backward compatibility)
        """
        return self._find_dl_model()\n    \n    def _load_label_encoder(self):\n        \"\"\"\n        Load label encoder if available\n        \"\"\"\n        encoder_path = os.path.join('models', 'saved_models', 'label_encoder.pkl')\n        \n        if os.path.exists(encoder_path):\n            try:\n                from utils.data_loader import SkinDataLoader\n                data_loader = SkinDataLoader('data')\n                data_loader.load_label_encoder(encoder_path)\n                print(\"Label encoder loaded successfully\")\n            except Exception as e:\n                print(f\"Could not load label encoder: {str(e)}\")\n    \n    def predict_image(self, image_path, return_probabilities=True):\n        \"\"\"\n        Predict skin type for a single image\n        \n        Args:\n            image_path (str): Path to the image file\n            return_probabilities (bool): Whether to return class probabilities\n            \n        Returns:\n            dict: Prediction results\n        \"\"\"\n        try:\n            # Process the image\n            processed_image, face_detected = self.image_processor.process_image(image_path)\n            \n            if processed_image is None:\n                return {\n                    'error': 'Failed to process image',\n                    'face_detected': False\n                }\n            \n            # Make prediction\n            result = self.classifier.predict_single_image(processed_image)\n            result['face_detected'] = face_detected\n            result['image_path'] = image_path\n            \n            # Add confidence level description\n            confidence = result['confidence']\n            if confidence >= 0.8:\n                result['confidence_level'] = 'High'\n            elif confidence >= 0.6:\n                result['confidence_level'] = 'Medium'\n            else:\n                result['confidence_level'] = 'Low'\n            \n            return result\n            \n        except Exception as e:\n            return {\n                'error': f'Prediction failed: {str(e)}',\n                'image_path': image_path\n            }\n    \n    def predict_batch(self, image_paths):\n        \"\"\"\n        Predict skin types for multiple images\n        \n        Args:\n            image_paths (list): List of image paths\n            \n        Returns:\n            list: List of prediction results\n        \"\"\"\n        results = []\n        \n        for image_path in image_paths:\n            result = self.predict_image(image_path)\n            results.append(result)\n        \n        return results\n    \n    def predict_from_camera(self, camera_index=0, num_frames=5):\n        \"\"\"\n        Predict skin type from camera feed\n        \n        Args:\n            camera_index (int): Camera index (usually 0 for default camera)\n            num_frames (int): Number of frames to capture and average\n            \n        Returns:\n            dict: Prediction results\n        \"\"\"\n        try:\n            # Initialize camera\n            cap = cv2.VideoCapture(camera_index)\n            \n            if not cap.isOpened():\n                return {'error': 'Could not open camera'}\n            \n            print(f\"Capturing {num_frames} frames from camera...\")\n            print(\"Press 'c' to capture or 'q' to quit\")\n            \n            captured_frames = []\n            \n            while len(captured_frames) < num_frames:\n                ret, frame = cap.read()\n                \n                if not ret:\n                    break\n                \n                # Display frame\n                cv2.imshow('Skin Type Analysis - Press C to Capture', frame)\n                \n                key = cv2.waitKey(1) & 0xFF\n                \n                if key == ord('c'):  # Capture frame\n                    captured_frames.append(frame.copy())\n                    print(f\"Captured frame {len(captured_frames)}/{num_frames}\")\n                \n                elif key == ord('q'):  # Quit\n                    break\n            \n            cap.release()\n            cv2.destroyAllWindows()\n            \n            if not captured_frames:\n                return {'error': 'No frames captured'}\n            \n            # Process captured frames\n            predictions = []\n            \n            for i, frame in enumerate(captured_frames):\n                # Convert BGR to RGB\n                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n                \n                # Process frame\n                processed_image, face_detected = self.image_processor.process_image(frame_rgb)\n                \n                if processed_image is not None:\n                    result = self.classifier.predict_single_image(processed_image)\n                    result['face_detected'] = face_detected\n                    predictions.append(result)\n            \n            if not predictions:\n                return {'error': 'No valid predictions from captured frames'}\n            \n            # Average the predictions\n            averaged_result = self._average_predictions(predictions)\n            averaged_result['num_frames_used'] = len(predictions)\n            \n            return averaged_result\n            \n        except Exception as e:\n            return {'error': f'Camera prediction failed: {str(e)}'}\n    \n    def _average_predictions(self, predictions):\n        \"\"\"\n        Average multiple predictions\n        \n        Args:\n            predictions (list): List of prediction dictionaries\n            \n        Returns:\n            dict: Averaged prediction results\n        \"\"\"\n        if not predictions:\n            return {'error': 'No predictions to average'}\n        \n        # Average probabilities\n        avg_probabilities = {}\n        class_names = self.classifier.class_names\n        \n        for class_name in class_names:\n            probs = [pred['probabilities'][class_name] for pred in predictions \n                    if 'probabilities' in pred and class_name in pred['probabilities']]\n            avg_probabilities[class_name] = np.mean(probs) if probs else 0.0\n        \n        # Find class with highest average probability\n        best_class = max(avg_probabilities.keys(), key=lambda k: avg_probabilities[k])\n        confidence = avg_probabilities[best_class]\n        \n        # Confidence level\n        if confidence >= 0.8:\n            confidence_level = 'High'\n        elif confidence >= 0.6:\n            confidence_level = 'Medium'\n        else:\n            confidence_level = 'Low'\n        \n        return {\n            'skin_type': best_class,\n            'confidence': float(confidence),\n            'confidence_level': confidence_level,\n            'probabilities': avg_probabilities,\n            'face_detected': any(pred.get('face_detected', False) for pred in predictions)\n        }\n    \n    def analyze_skin_characteristics(self, image_path):\n        \"\"\"\n        Provide detailed skin analysis beyond just type classification\n        \n        Args:\n            image_path (str): Path to the image file\n            \n        Returns:\n            dict: Detailed skin analysis\n        \"\"\"\n        # Get basic prediction\n        result = self.predict_image(image_path)\n        \n        if 'error' in result:\n            return result\n        \n        # Add detailed analysis based on skin type\n        skin_type = result['skin_type']\n        analysis = self._get_skin_analysis(skin_type, result['probabilities'])\n        \n        result['detailed_analysis'] = analysis\n        \n        return result\n    \n    def _get_skin_analysis(self, skin_type, probabilities):\n        \"\"\"\n        Generate detailed skin analysis based on predicted type\n        \n        Args:\n            skin_type (str): Predicted skin type\n            probabilities (dict): Class probabilities\n            \n        Returns:\n            dict: Detailed analysis\n        \"\"\"\n        skin_analysis = {\n            'normal': {\n                'characteristics': ['Balanced oil production', 'Good hydration', 'Even texture', 'Minimal sensitivity'],\n                'care_tips': ['Use gentle cleanser', 'Moisturize daily', 'Apply sunscreen', 'Regular exfoliation'],\n                'concerns': ['Maintain current routine', 'Protect from environmental damage']\n            },\n            'dry': {\n                'characteristics': ['Low oil production', 'Possible flaking', 'Tight feeling', 'Fine lines'],\n                'care_tips': ['Use hydrating cleanser', 'Rich moisturizer', 'Hyaluronic acid', 'Avoid harsh products'],\n                'concerns': ['Dehydration', 'Premature aging', 'Sensitivity to weather']\n            },\n            'oily': {\n                'characteristics': ['Excess sebum', 'Shine', 'Large pores', 'Acne-prone'],\n                'care_tips': ['Oil-free cleanser', 'Lightweight moisturizer', 'Salicylic acid', 'Clay masks'],\n                'concerns': ['Acne breakouts', 'Clogged pores', 'Excessive shine']\n            },\n            'combination': {\n                'characteristics': ['Oily T-zone', 'Dry cheeks', 'Mixed texture', 'Variable needs'],\n                'care_tips': ['Dual-approach care', 'Different products for different areas', 'Balanced routine'],\n                'concerns': ['Managing different zones', 'Finding suitable products']\n            },\n            'sensitive': {\n                'characteristics': ['Reactive to products', 'Redness', 'Irritation-prone', 'Thin skin'],\n                'care_tips': ['Gentle products', 'Fragrance-free', 'Patch testing', 'Minimal ingredients'],\n                'concerns': ['Product reactions', 'Environmental sensitivity', 'Inflammation']\n            }\n        }\n        \n        analysis = skin_analysis.get(skin_type, skin_analysis['normal']).copy()\n        \n        # Add probability-based insights\n        analysis['secondary_characteristics'] = []\n        \n        # Check for mixed characteristics\n        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)\n        \n        if len(sorted_probs) > 1 and sorted_probs[1][1] > 0.2:\n            secondary_type = sorted_probs[1][0]\n            analysis['secondary_characteristics'].append(\n                f\"Shows some {secondary_type} skin characteristics ({sorted_probs[1][1]:.1%})\")\n        \n        return analysis\n    \n    def save_prediction_report(self, results, output_path):\n        \"\"\"\n        Save prediction results to a JSON report\n        \n        Args:\n            results (dict or list): Prediction results\n            output_path (str): Path to save the report\n        \"\"\"\n        try:\n            with open(output_path, 'w') as f:\n                json.dump(results, f, indent=2, default=str)\n            print(f\"Prediction report saved to: {output_path}\")\n        except Exception as e:\n            print(f\"Failed to save report: {str(e)}\")\n\n\ndef main():\n    \"\"\"\n    Command-line interface for skin type prediction\n    \"\"\"\n    parser = argparse.ArgumentParser(description='Skin Type Analysis Tool')\n    parser.add_argument('--image', '-i', type=str, help='Path to image file')\n    parser.add_argument('--batch', '-b', type=str, nargs='+', help='Multiple image paths')\n    parser.add_argument('--camera', '-c', action='store_true', help='Use camera for prediction')\n    parser.add_argument('--model', '-m', type=str, help='Path to model file')\n    parser.add_argument('--architecture', '-a', type=str, default='efficientnet',\n                       choices=['efficientnet', 'resnet50', 'mobilenet', 'custom'],\n                       help='Model architecture')\n    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed analysis')\n    parser.add_argument('--output', '-o', type=str, help='Output file for results')\n    \n    args = parser.parse_args()\n    \n    # Initialize predictor\n    predictor = SkinTypePredictor(\n        model_path=args.model,\n        model_architecture=args.architecture\n    )\n    \n    results = None\n    \n    if args.camera:\n        # Camera prediction\n        print(\"Starting camera-based skin type analysis...\")\n        results = predictor.predict_from_camera()\n        \n    elif args.batch:\n        # Batch prediction\n        print(f\"Analyzing {len(args.batch)} images...\")\n        if args.detailed:\n            results = [predictor.analyze_skin_characteristics(img) for img in args.batch]\n        else:\n            results = predictor.predict_batch(args.batch)\n        \n    elif args.image:\n        # Single image prediction\n        print(f\"Analyzing image: {args.image}\")\n        if args.detailed:\n            results = predictor.analyze_skin_characteristics(args.image)\n        else:\n            results = predictor.predict_image(args.image)\n    \n    else:\n        print(\"No input specified. Use --help for usage information.\")\n        return\n    \n    # Display results\n    if results:\n        print(\"\\n\" + \"=\"*60)\n        print(\"SKIN TYPE ANALYSIS RESULTS\")\n        print(\"=\"*60)\n        \n        if isinstance(results, list):\n            # Multiple results\n            for i, result in enumerate(results):\n                print(f\"\\nImage {i+1}:\")\n                print_single_result(result)\n        else:\n            # Single result\n            print_single_result(results)\n        \n        # Save results if requested\n        if args.output:\n            predictor.save_prediction_report(results, args.output)\n    \n    print(\"\\nAnalysis completed!\")\n\n\ndef print_single_result(result):\n    \"\"\"\n    Print a single prediction result in a formatted way\n    \"\"\"\n    if 'error' in result:\n        print(f\"  Error: {result['error']}\")\n        return\n    \n    print(f\"  Skin Type: {result['skin_type'].upper()}\")\n    print(f\"  Confidence: {result['confidence']:.2%} ({result.get('confidence_level', 'Unknown')})\")\n    print(f\"  Face Detected: {'Yes' if result.get('face_detected', False) else 'No'}\")\n    \n    # Show probabilities\n    if 'probabilities' in result:\n        print(\"  Class Probabilities:\")\n        for class_name, prob in sorted(result['probabilities'].items(), \n                                     key=lambda x: x[1], reverse=True):\n            print(f\"    {class_name.capitalize()}: {prob:.2%}\")\n    \n    # Show detailed analysis if available\n    if 'detailed_analysis' in result:\n        analysis = result['detailed_analysis']\n        print(\"\\n  Detailed Analysis:\")\n        print(f\"    Characteristics: {', '.join(analysis.get('characteristics', []))}\")\n        print(f\"    Care Tips: {', '.join(analysis.get('care_tips', []))}\")\n        print(f\"    Main Concerns: {', '.join(analysis.get('concerns', []))}\")\n        \n        if analysis.get('secondary_characteristics'):\n            print(f\"    Additional Notes: {', '.join(analysis['secondary_characteristics'])}\")\n\n\nif __name__ == \"__main__\":\n    main()
