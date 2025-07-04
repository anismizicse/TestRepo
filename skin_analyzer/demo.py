"""
Demo script showing how to use the skin analyzer
"""

import os
import sys
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_downloader import DatasetDownloader
from models.train_model import ModelTrainer
from predict import SkinTypePredictor
from evaluate_model import ModelEvaluator


def demo_dataset_creation():
    """
    Demonstrate dataset creation
    """
    print("="*60)
    print("DEMO: Dataset Creation")
    print("="*60)
    
    data_dir = 'data'
    downloader = DatasetDownloader(data_dir)
    
    print("Creating curated skin type dataset...")
    downloader.create_curated_dataset()
    
    print("Dataset creation completed!")


def demo_model_training():
    """
    Demonstrate model training
    """
    print("="*60)
    print("DEMO: Model Training")
    print("="*60)
    
    data_dir = 'data'
    
    # Train a lightweight model for demo
    trainer = ModelTrainer(data_dir, model_architecture='mobilenet')
    
    print("Starting model training...")
    results = trainer.run_complete_training_pipeline()
    
    print(f"Training completed! Test accuracy: {results['test_accuracy']:.4f}")


def demo_prediction():
    """
    Demonstrate prediction on sample images
    """
    print("="*60)
    print("DEMO: Skin Type Prediction")
    print("="*60)
    
    # Initialize predictor
    predictor = SkinTypePredictor(model_architecture='mobilenet')
    
    # Check if we have test images
    test_dir = os.path.join('data', 'test')
    if not os.path.exists(test_dir):
        print("No test images found. Please run dataset creation first.")
        return
    
    # Find sample images
    sample_images = []
    for skin_type in ['normal', 'dry', 'oily', 'combination', 'sensitive']:
        type_dir = os.path.join(test_dir, skin_type)
        if os.path.exists(type_dir):
            images = [f for f in os.listdir(type_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                sample_images.append(os.path.join(type_dir, images[0]))
                if len(sample_images) >= 3:  # Limit to 3 samples for demo
                    break
    
    if not sample_images:
        print("No sample images found.")
        return
    
    print(f"Analyzing {len(sample_images)} sample images...")
    
    for i, image_path in enumerate(sample_images):
        print(f"\nSample {i+1}: {os.path.basename(image_path)}")
        result = predictor.analyze_skin_characteristics(image_path)
        
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Predicted Skin Type: {result['skin_type'].upper()}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Face Detected: {'Yes' if result['face_detected'] else 'No'}")
            
            # Show top 2 probabilities
            probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            print(f"  Top predictions:")
            for j, (skin_type, prob) in enumerate(probs[:2]):
                print(f"    {j+1}. {skin_type.capitalize()}: {prob:.2%}")


def demo_model_evaluation():
    """
    Demonstrate model evaluation
    """
    print("="*60)
    print("DEMO: Model Evaluation")
    print("="*60)
    
    data_dir = 'data'
    
    try:
        evaluator = ModelEvaluator(data_dir, model_architecture='mobilenet')
        results = evaluator.run_comprehensive_evaluation()
        
        print("Model evaluation completed!")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print("Please ensure a trained model exists.")


def main():
    """
    Run complete demo
    """
    print("SKIN TYPE ANALYZER - COMPLETE DEMO")
    print("="*60)
    
    # Check if user wants to run specific demos
    import argparse
    parser = argparse.ArgumentParser(description='Skin Analyzer Demo')
    parser.add_argument('--step', type=str, choices=['dataset', 'train', 'predict', 'evaluate', 'all'],
                       default='all', help='Which demo step to run')
    
    args = parser.parse_args()
    
    if args.step in ['dataset', 'all']:
        demo_dataset_creation()
        print("\n")
    
    if args.step in ['train', 'all']:
        demo_model_training()
        print("\n")
    
    if args.step in ['predict', 'all']:
        demo_prediction()
        print("\n")
    
    if args.step in ['evaluate', 'all']:
        demo_model_evaluation()
        print("\n")
    
    print("Demo completed!")
    print("\nTo use the skin analyzer:")
    print("1. Create dataset: python utils/dataset_downloader.py")
    print("2. Train model: python models/train_model.py")
    print("3. Make predictions: python predict.py --image path/to/image.jpg")
    print("4. Evaluate model: python evaluate_model.py")


if __name__ == "__main__":
    main()
