"""
Training Script for Skin Type Classification Model
Handles the complete training pipeline including data loading, model training, and evaluation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import SkinDataLoader
from utils.dataset_downloader import DatasetDownloader
from models.skin_classifier import SkinClassifier
import json


class ModelTrainer:
    """
    Handles the complete model training pipeline
    """
    
    def __init__(self, data_dir, model_architecture='efficientnet'):
        """
        Initialize the model trainer
        
        Args:
            data_dir (str): Path to the data directory
            model_architecture (str): Model architecture to use
        """
        self.data_dir = data_dir
        self.model_architecture = model_architecture
        self.data_loader = SkinDataLoader(data_dir, batch_size=32)
        self.classifier = SkinClassifier(base_model=model_architecture)
        
        # Create models directory if it doesn't exist
        self.models_dir = os.path.join(os.path.dirname(data_dir), 'models', 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Training configuration
        self.config = {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'fine_tune_epochs': 20,
            'image_size': (224, 224)
        }
    
    def prepare_data(self):
        """
        Prepare and load the dataset
        """
        print("Preparing dataset...")
        
        # Check if dataset exists, if not create it
        if not self._check_dataset_exists():
            print("Dataset not found. Creating new dataset...")
            downloader = DatasetDownloader(self.data_dir)
            downloader.create_curated_dataset()
        
        # Load training data
        print("Loading training data...")
        train_images, train_labels, val_images, val_labels = self.data_loader.load_dataset_from_directory(
            split='train',
            validation_split=self.config['validation_split']
        )
        
        # Load test data
        print("Loading test data...")
        test_images, test_labels = self.data_loader.load_dataset_from_directory(
            split='test',
            validation_split=0.0
        )
        
        # Create TensorFlow datasets
        print("Creating TensorFlow datasets...")
        self.train_dataset = self.data_loader.create_tf_dataset(
            train_images, train_labels, 
            shuffle=True, augment=True
        )
        
        self.val_dataset = self.data_loader.create_tf_dataset(
            val_images, val_labels, 
            shuffle=False, augment=False
        )
        
        self.test_dataset = self.data_loader.create_tf_dataset(
            test_images, test_labels, 
            shuffle=False, augment=False
        )
        
        # Store for later use
        self.test_images = test_images
        self.test_labels = test_labels
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(train_images)}")
        print(f"Validation samples: {len(val_images)}")
        print(f"Test samples: {len(test_images)}")
    
    def _check_dataset_exists(self):
        """
        Check if the dataset exists and has the required structure
        """
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            return False
        
        # Check if all skin type directories exist and have images
        skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        
        for skin_type in skin_types:
            train_type_dir = os.path.join(train_dir, skin_type)
            test_type_dir = os.path.join(test_dir, skin_type)
            
            if not os.path.exists(train_type_dir) or not os.path.exists(test_type_dir):
                return False
            
            # Check if directories have images
            train_images = [f for f in os.listdir(train_type_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            test_images = [f for f in os.listdir(test_type_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(train_images) == 0 or len(test_images) == 0:
                return False
        
        return True
    
    def train_model(self):
        """
        Train the skin classification model
        """
        print(f"Training {self.model_architecture} model...")
        
        # Build and compile the model
        self.classifier.build_model()
        
        # Train the model
        model_save_path = os.path.join(self.models_dir, f'skin_classifier_{self.model_architecture}.h5')
        
        history = self.classifier.train(
            self.train_dataset,
            self.val_dataset,
            epochs=self.config['epochs'],
            model_save_path=model_save_path
        )
        
        # Fine-tune if using pre-trained model
        if self.model_architecture != 'custom':
            print("Starting fine-tuning...")
            self.classifier.fine_tune_model(
                self.train_dataset,
                self.val_dataset,
                epochs=self.config['fine_tune_epochs']
            )
        
        return history
    
    def evaluate_model(self):
        """
        Evaluate the trained model
        """
        print("Evaluating model...")
        
        # Evaluate on test set
        results = self.classifier.evaluate(self.test_dataset)
        
        # Generate detailed predictions for analysis
        predictions = self.classifier.model.predict(self.test_dataset)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.test_labels, axis=1)
        
        # Classification report
        class_names = self.classifier.class_names
        report = classification_report(
            true_classes, predicted_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Save detailed results
        detailed_results = {
            'test_accuracy': float(results['test_accuracy']),
            'test_loss': float(results['test_loss']),
            'test_top2_accuracy': float(results['test_top2_accuracy']),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'model_architecture': self.model_architecture,
            'config': self.config
        }
        
        # Save results
        results_path = os.path.join(self.models_dir, f'evaluation_results_{self.model_architecture}.json')
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"Detailed results saved to: {results_path}")
        
        return detailed_results
    
    def plot_training_history(self, history):
        """
        Plot training history
        """
        if history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 3, 1)\n        plt.plot(history.history['accuracy'], label='Training Accuracy')\n        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')\n        plt.title('Model Accuracy')\n        plt.xlabel('Epoch')\n        plt.ylabel('Accuracy')\n        plt.legend()\n        plt.grid(True)\n        \n        # Plot loss\n        plt.subplot(1, 3, 2)\n        plt.plot(history.history['loss'], label='Training Loss')\n        plt.plot(history.history['val_loss'], label='Validation Loss')\n        plt.title('Model Loss')\n        plt.xlabel('Epoch')\n        plt.ylabel('Loss')\n        plt.legend()\n        plt.grid(True)\n        \n        # Plot top-2 accuracy\n        plt.subplot(1, 3, 3)\n        if 'top_2_accuracy' in history.history:\n            plt.plot(history.history['top_2_accuracy'], label='Training Top-2 Accuracy')\n            plt.plot(history.history['val_top_2_accuracy'], label='Validation Top-2 Accuracy')\n            plt.title('Model Top-2 Accuracy')\n            plt.xlabel('Epoch')\n            plt.ylabel('Top-2 Accuracy')\n            plt.legend()\n            plt.grid(True)\n        \n        plt.tight_layout()\n        \n        # Save plot\n        plot_path = os.path.join(self.models_dir, f'training_history_{self.model_architecture}.png')\n        plt.savefig(plot_path, dpi=300, bbox_inches='tight')\n        plt.show()\n        \n        print(f\"Training history plot saved to: {plot_path}\")\n    \n    def plot_confusion_matrix(self, detailed_results):\n        \"\"\"\n        Plot confusion matrix\n        \"\"\"\n        cm = np.array(detailed_results['confusion_matrix'])\n        class_names = detailed_results['class_names']\n        \n        plt.figure(figsize=(10, 8))\n        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n                    xticklabels=class_names, yticklabels=class_names)\n        plt.title(f'Confusion Matrix - {self.model_architecture.upper()}')\n        plt.xlabel('Predicted Label')\n        plt.ylabel('True Label')\n        \n        # Save plot\n        plot_path = os.path.join(self.models_dir, f'confusion_matrix_{self.model_architecture}.png')\n        plt.savefig(plot_path, dpi=300, bbox_inches='tight')\n        plt.show()\n        \n        print(f\"Confusion matrix plot saved to: {plot_path}\")\n    \n    def generate_classification_report(self, detailed_results):\n        \"\"\"\n        Print and save classification report\n        \"\"\"\n        report = detailed_results['classification_report']\n        class_names = detailed_results['class_names']\n        \n        print(\"\\n\" + \"=\"*60)\n        print(f\"CLASSIFICATION REPORT - {self.model_architecture.upper()}\")\n        print(\"=\"*60)\n        \n        # Print per-class metrics\n        print(f\"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\")\n        print(\"-\" * 60)\n        \n        for class_name in class_names:\n            if class_name in report:\n                metrics = report[class_name]\n                print(f\"{class_name:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} \"\n                      f\"{metrics['f1-score']:<10.3f} {metrics['support']:<10.0f}\")\n        \n        # Print overall metrics\n        print(\"-\" * 60)\n        accuracy = report['accuracy']\n        macro_avg = report['macro avg']\n        weighted_avg = report['weighted avg']\n        \n        print(f\"{'Accuracy':<12} {'':<10} {'':<10} {accuracy:<10.3f} {report['macro avg']['support']:<10.0f}\")\n        print(f\"{'Macro Avg':<12} {macro_avg['precision']:<10.3f} {macro_avg['recall']:<10.3f} \"\n              f\"{macro_avg['f1-score']:<10.3f} {macro_avg['support']:<10.0f}\")\n        print(f\"{'Weighted Avg':<12} {weighted_avg['precision']:<10.3f} {weighted_avg['recall']:<10.3f} \"\n              f\"{weighted_avg['f1-score']:<10.3f} {weighted_avg['support']:<10.0f}\")\n        \n        print(\"=\"*60)\n    \n    def save_label_encoder(self):\n        \"\"\"\n        Save the label encoder for later use\n        \"\"\"\n        encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')\n        self.data_loader.save_label_encoder(encoder_path)\n        print(f\"Label encoder saved to: {encoder_path}\")\n    \n    def run_complete_training_pipeline(self):\n        \"\"\"\n        Run the complete training pipeline\n        \"\"\"\n        print(\"Starting complete training pipeline...\")\n        print(f\"Model Architecture: {self.model_architecture}\")\n        print(f\"Data Directory: {self.data_dir}\")\n        \n        try:\n            # Step 1: Prepare data\n            self.prepare_data()\n            \n            # Step 2: Train model\n            history = self.train_model()\n            \n            # Step 3: Evaluate model\n            detailed_results = self.evaluate_model()\n            \n            # Step 4: Generate visualizations and reports\n            self.plot_training_history(history)\n            self.plot_confusion_matrix(detailed_results)\n            self.generate_classification_report(detailed_results)\n            \n            # Step 5: Save label encoder\n            self.save_label_encoder()\n            \n            # Final summary\n            print(\"\\n\" + \"=\"*80)\n            print(\"TRAINING PIPELINE COMPLETED SUCCESSFULLY!\")\n            print(\"=\"*80)\n            print(f\"Model Architecture: {self.model_architecture}\")\n            print(f\"Test Accuracy: {detailed_results['test_accuracy']:.4f}\")\n            print(f\"Test Top-2 Accuracy: {detailed_results['test_top2_accuracy']:.4f}\")\n            print(f\"Model saved to: {self.models_dir}\")\n            print(\"=\"*80)\n            \n            return detailed_results\n            \n        except Exception as e:\n            print(f\"Error in training pipeline: {str(e)}\")\n            raise e\n\n\ndef main():\n    \"\"\"\n    Main function to run model training\n    \"\"\"\n    # Configuration\n    current_dir = os.path.dirname(os.path.abspath(__file__))\n    data_dir = os.path.join(os.path.dirname(current_dir), 'data')\n    \n    # Model architectures to train\n    architectures = ['efficientnet', 'mobilenet', 'custom']\n    \n    results_summary = {}\n    \n    for architecture in architectures:\n        print(f\"\\n{'='*80}\")\n        print(f\"TRAINING {architecture.upper()} MODEL\")\n        print(f\"{'='*80}\")\n        \n        try:\n            # Create trainer\n            trainer = ModelTrainer(data_dir, architecture)\n            \n            # Run training pipeline\n            results = trainer.run_complete_training_pipeline()\n            \n            # Store results\n            results_summary[architecture] = {\n                'test_accuracy': results['test_accuracy'],\n                'test_top2_accuracy': results['test_top2_accuracy'],\n                'test_loss': results['test_loss']\n            }\n            \n        except Exception as e:\n            print(f\"Failed to train {architecture} model: {str(e)}\")\n            results_summary[architecture] = {'error': str(e)}\n    \n    # Print final summary\n    print(\"\\n\" + \"=\"*80)\n    print(\"FINAL TRAINING SUMMARY\")\n    print(\"=\"*80)\n    \n    for architecture, results in results_summary.items():\n        if 'error' in results:\n            print(f\"{architecture.upper():<15} - FAILED: {results['error']}\")\n        else:\n            print(f\"{architecture.upper():<15} - Accuracy: {results['test_accuracy']:.4f}, \"\n                  f\"Top-2: {results['test_top2_accuracy']:.4f}, Loss: {results['test_loss']:.4f}\")\n    \n    print(\"=\"*80)\n    print(\"Training completed! Check the models/saved_models/ directory for trained models.\")\n\n\nif __name__ == \"__main__\":\n    main()
