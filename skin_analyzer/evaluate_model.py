"""
Model Evaluation Script
Comprehensive evaluation of trained skin type classification models
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import SkinDataLoader
from models.skin_classifier import SkinClassifier
from predict import SkinTypePredictor


class ModelEvaluator:
    """
    Comprehensive model evaluation tool
    """
    
    def __init__(self, data_dir, model_path=None, model_architecture='efficientnet'):
        """
        Initialize the model evaluator
        
        Args:
            data_dir (str): Path to the data directory
            model_path (str): Path to the trained model
            model_architecture (str): Model architecture
        """
        self.data_dir = data_dir
        self.model_architecture = model_architecture
        self.data_loader = SkinDataLoader(data_dir, batch_size=32)
        
        # Initialize classifier and load model
        self.classifier = SkinClassifier(base_model=model_architecture)
        
        if model_path and os.path.exists(model_path):
            self.classifier.load_model(model_path)
        else:
            # Try to find a model automatically
            models_dir = os.path.join('models', 'saved_models')
            model_files = [f'skin_classifier_{model_architecture}.h5', 'fine_tuned_model.h5']
            
            for model_file in model_files:
                full_path = os.path.join(models_dir, model_file)
                if os.path.exists(full_path):
                    self.classifier.load_model(full_path)
                    break
            else:
                raise ValueError(f"No model found for architecture: {model_architecture}")
        
        self.class_names = self.classifier.class_names
        self.num_classes = len(self.class_names)
    
    def load_test_data(self):
        """
        Load test dataset for evaluation
        """
        print("Loading test dataset...")
        
        # Load test data
        test_images, test_labels = self.data_loader.load_dataset_from_directory(
            split='test', validation_split=0.0
        )
        
        # Create TensorFlow dataset
        self.test_dataset = self.data_loader.create_tf_dataset(
            test_images, test_labels, shuffle=False, augment=False
        )
        
        # Store for detailed analysis
        self.test_images = test_images
        self.test_labels = test_labels
        self.true_classes = np.argmax(test_labels, axis=1)
        
        print(f"Test dataset loaded: {len(test_images)} samples")
        
        return test_images, test_labels
    
    def evaluate_model_performance(self):
        """
        Evaluate model performance on test set
        
        Returns:
            dict: Comprehensive evaluation results
        """
        print("Evaluating model performance...")
        
        # Get model predictions
        predictions = self.classifier.model.predict(self.test_dataset, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Basic metrics
        test_loss, test_accuracy, test_top2_accuracy = self.classifier.model.evaluate(
            self.test_dataset, verbose=0
        )
        
        # Classification report
        report = classification_report(
            self.true_classes, predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.true_classes, predicted_classes)
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Calculate additional metrics
        results = {\n            'test_accuracy': float(test_accuracy),\n            'test_top2_accuracy': float(test_top2_accuracy),\n            'test_loss': float(test_loss),\n            'classification_report': report,\n            'confusion_matrix': cm.tolist(),\n            'per_class_accuracy': {class_name: float(acc) for class_name, acc in \n                                 zip(self.class_names, per_class_accuracy)},\n            'predictions': predictions.tolist(),\n            'predicted_classes': predicted_classes.tolist(),\n            'true_classes': self.true_classes.tolist(),\n            'class_names': self.class_names,\n            'model_architecture': self.model_architecture\n        }\n        \n        return results\n    \n    def calculate_roc_metrics(self, predictions):\n        \"\"\"\n        Calculate ROC curves and AUC scores for multi-class classification\n        \n        Args:\n            predictions (numpy.ndarray): Model predictions\n            \n        Returns:\n            dict: ROC metrics\n        \"\"\"\n        print(\"Calculating ROC metrics...\")\n        \n        # Binarize the true labels for multi-class ROC\n        y_test_binarized = label_binarize(self.true_classes, classes=range(self.num_classes))\n        \n        # Calculate ROC curve and AUC for each class\n        fpr = {}\n        tpr = {}\n        roc_auc = {}\n        \n        for i, class_name in enumerate(self.class_names):\n            fpr[class_name], tpr[class_name], _ = roc_curve(\n                y_test_binarized[:, i], predictions[:, i]\n            )\n            roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])\n        \n        # Calculate macro-average ROC curve and AUC\n        all_fpr = np.unique(np.concatenate([fpr[class_name] for class_name in self.class_names]))\n        mean_tpr = np.zeros_like(all_fpr)\n        \n        for class_name in self.class_names:\n            mean_tpr += np.interp(all_fpr, fpr[class_name], tpr[class_name])\n        \n        mean_tpr /= self.num_classes\n        \n        fpr[\"macro\"] = all_fpr\n        tpr[\"macro\"] = mean_tpr\n        roc_auc[\"macro\"] = auc(fpr[\"macro\"], tpr[\"macro\"])\n        \n        return {\n            'fpr': {k: v.tolist() for k, v in fpr.items()},\n            'tpr': {k: v.tolist() for k, v in tpr.items()},\n            'auc': roc_auc\n        }\n    \n    def analyze_prediction_confidence(self, predictions):\n        \"\"\"\n        Analyze prediction confidence distribution\n        \n        Args:\n            predictions (numpy.ndarray): Model predictions\n            \n        Returns:\n            dict: Confidence analysis\n        \"\"\"\n        confidence_scores = np.max(predictions, axis=1)\n        predicted_classes = np.argmax(predictions, axis=1)\n        \n        # Overall confidence statistics\n        confidence_stats = {\n            'mean_confidence': float(np.mean(confidence_scores)),\n            'std_confidence': float(np.std(confidence_scores)),\n            'min_confidence': float(np.min(confidence_scores)),\n            'max_confidence': float(np.max(confidence_scores))\n        }\n        \n        # Confidence by class\n        confidence_by_class = {}\n        for i, class_name in enumerate(self.class_names):\n            class_mask = predicted_classes == i\n            if np.any(class_mask):\n                class_confidences = confidence_scores[class_mask]\n                confidence_by_class[class_name] = {\n                    'mean': float(np.mean(class_confidences)),\n                    'std': float(np.std(class_confidences)),\n                    'count': int(np.sum(class_mask))\n                }\n        \n        # Confidence level distribution\n        high_conf = np.sum(confidence_scores >= 0.8)\n        medium_conf = np.sum((confidence_scores >= 0.6) & (confidence_scores < 0.8))\n        low_conf = np.sum(confidence_scores < 0.6)\n        \n        confidence_distribution = {\n            'high_confidence': int(high_conf),\n            'medium_confidence': int(medium_conf),\n            'low_confidence': int(low_conf),\n            'high_confidence_pct': float(high_conf / len(confidence_scores)),\n            'medium_confidence_pct': float(medium_conf / len(confidence_scores)),\n            'low_confidence_pct': float(low_conf / len(confidence_scores))\n        }\n        \n        return {\n            'confidence_stats': confidence_stats,\n            'confidence_by_class': confidence_by_class,\n            'confidence_distribution': confidence_distribution\n        }\n    \n    def find_misclassified_samples(self, predictions, top_n=10):\n        \"\"\"\n        Find and analyze the most confidently misclassified samples\n        \n        Args:\n            predictions (numpy.ndarray): Model predictions\n            top_n (int): Number of top misclassified samples to return\n            \n        Returns:\n            list: Misclassified samples with details\n        \"\"\"\n        predicted_classes = np.argmax(predictions, axis=1)\n        confidence_scores = np.max(predictions, axis=1)\n        \n        # Find misclassified samples\n        misclassified_mask = predicted_classes != self.true_classes\n        misclassified_indices = np.where(misclassified_mask)[0]\n        \n        if len(misclassified_indices) == 0:\n            return []\n        \n        # Sort by confidence (most confident mistakes first)\n        misclassified_confidences = confidence_scores[misclassified_indices]\n        sorted_indices = misclassified_indices[np.argsort(misclassified_confidences)[::-1]]\n        \n        # Get top N misclassified samples\n        top_misclassified = []\n        for idx in sorted_indices[:top_n]:\n            sample_info = {\n                'sample_index': int(idx),\n                'true_class': self.class_names[self.true_classes[idx]],\n                'predicted_class': self.class_names[predicted_classes[idx]],\n                'confidence': float(confidence_scores[idx]),\n                'probabilities': {class_name: float(prob) \n                                for class_name, prob in zip(self.class_names, predictions[idx])}\n            }\n            top_misclassified.append(sample_info)\n        \n        return top_misclassified\n    \n    def generate_evaluation_plots(self, results, output_dir='evaluation_plots'):\n        \"\"\"\n        Generate comprehensive evaluation plots\n        \n        Args:\n            results (dict): Evaluation results\n            output_dir (str): Directory to save plots\n        \"\"\"\n        os.makedirs(output_dir, exist_ok=True)\n        \n        # Set style\n        plt.style.use('default')\n        sns.set_palette(\"husl\")\n        \n        # 1. Confusion Matrix\n        self._plot_confusion_matrix(results, output_dir)\n        \n        # 2. Per-class accuracy\n        self._plot_per_class_accuracy(results, output_dir)\n        \n        # 3. ROC curves\n        if 'roc_metrics' in results:\n            self._plot_roc_curves(results['roc_metrics'], output_dir)\n        \n        # 4. Confidence distribution\n        if 'confidence_analysis' in results:\n            self._plot_confidence_distribution(results['confidence_analysis'], output_dir)\n        \n        # 5. Classification report heatmap\n        self._plot_classification_report_heatmap(results, output_dir)\n        \n        print(f\"Evaluation plots saved to: {output_dir}\")\n    \n    def _plot_confusion_matrix(self, results, output_dir):\n        \"\"\"Plot confusion matrix\"\"\"\n        cm = np.array(results['confusion_matrix'])\n        \n        plt.figure(figsize=(10, 8))\n        \n        # Normalize confusion matrix\n        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]\n        \n        # Create heatmap\n        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',\n                    xticklabels=self.class_names, yticklabels=self.class_names)\n        \n        plt.title(f'Normalized Confusion Matrix - {self.model_architecture.upper()}')\n        plt.xlabel('Predicted Label')\n        plt.ylabel('True Label')\n        plt.tight_layout()\n        \n        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')\n        plt.close()\n    \n    def _plot_per_class_accuracy(self, results, output_dir):\n        \"\"\"Plot per-class accuracy\"\"\"\n        per_class_acc = results['per_class_accuracy']\n        \n        plt.figure(figsize=(12, 6))\n        \n        classes = list(per_class_acc.keys())\n        accuracies = list(per_class_acc.values())\n        \n        bars = plt.bar(classes, accuracies, color=sns.color_palette(\"husl\", len(classes)))\n        \n        # Add value labels on bars\n        for bar, acc in zip(bars, accuracies):\n            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n                    f'{acc:.2%}', ha='center', va='bottom')\n        \n        plt.title(f'Per-Class Accuracy - {self.model_architecture.upper()}')\n        plt.xlabel('Skin Type')\n        plt.ylabel('Accuracy')\n        plt.ylim(0, 1.1)\n        plt.xticks(rotation=45)\n        plt.grid(axis='y', alpha=0.3)\n        plt.tight_layout()\n        \n        plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')\n        plt.close()\n    \n    def _plot_roc_curves(self, roc_metrics, output_dir):\n        \"\"\"Plot ROC curves\"\"\"\n        plt.figure(figsize=(12, 8))\n        \n        # Plot ROC curve for each class\n        for class_name in self.class_names:\n            plt.plot(roc_metrics['fpr'][class_name], roc_metrics['tpr'][class_name],\n                    label=f'{class_name} (AUC = {roc_metrics[\"auc\"][class_name]:.3f})')\n        \n        # Plot macro-average ROC curve\n        plt.plot(roc_metrics['fpr']['macro'], roc_metrics['tpr']['macro'],\n                label=f'Macro-average (AUC = {roc_metrics[\"auc\"][\"macro\"]:.3f})',\n                linestyle='--', linewidth=2)\n        \n        # Plot random classifier line\n        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')\n        \n        plt.xlabel('False Positive Rate')\n        plt.ylabel('True Positive Rate')\n        plt.title(f'ROC Curves - {self.model_architecture.upper()}')\n        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n        plt.grid(alpha=0.3)\n        plt.tight_layout()\n        \n        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')\n        plt.close()\n    \n    def _plot_confidence_distribution(self, confidence_analysis, output_dir):\n        \"\"\"Plot confidence distribution\"\"\"\n        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n        \n        # Overall confidence distribution\n        conf_dist = confidence_analysis['confidence_distribution']\n        labels = ['High (≥80%)', 'Medium (60-80%)', 'Low (<60%)']\n        sizes = [conf_dist['high_confidence'], conf_dist['medium_confidence'], conf_dist['low_confidence']]\n        colors = ['#2ecc71', '#f39c12', '#e74c3c']\n        \n        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)\n        ax1.set_title('Prediction Confidence Distribution')\n        \n        # Confidence by class\n        conf_by_class = confidence_analysis['confidence_by_class']\n        classes = list(conf_by_class.keys())\n        mean_confidences = [conf_by_class[c]['mean'] for c in classes]\n        std_confidences = [conf_by_class[c]['std'] for c in classes]\n        \n        bars = ax2.bar(classes, mean_confidences, yerr=std_confidences, capsize=5,\n                      color=sns.color_palette(\"husl\", len(classes)))\n        \n        ax2.set_title('Mean Confidence by Class')\n        ax2.set_xlabel('Skin Type')\n        ax2.set_ylabel('Mean Confidence')\n        ax2.set_ylim(0, 1)\n        ax2.tick_params(axis='x', rotation=45)\n        ax2.grid(axis='y', alpha=0.3)\n        \n        plt.tight_layout()\n        plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')\n        plt.close()\n    \n    def _plot_classification_report_heatmap(self, results, output_dir):\n        \"\"\"Plot classification report as heatmap\"\"\"\n        report = results['classification_report']\n        \n        # Extract metrics for each class\n        metrics_data = []\n        for class_name in self.class_names:\n            if class_name in report:\n                metrics_data.append([\n                    report[class_name]['precision'],\n                    report[class_name]['recall'],\n                    report[class_name]['f1-score']\n                ])\n        \n        # Create DataFrame\n        df = pd.DataFrame(metrics_data, \n                         index=self.class_names,\n                         columns=['Precision', 'Recall', 'F1-Score'])\n        \n        plt.figure(figsize=(8, 6))\n        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', \n                   cbar_kws={'label': 'Score'})\n        plt.title(f'Classification Metrics - {self.model_architecture.upper()}')\n        plt.ylabel('Skin Type')\n        plt.tight_layout()\n        \n        plt.savefig(os.path.join(output_dir, 'classification_metrics.png'), dpi=300, bbox_inches='tight')\n        plt.close()\n    \n    def run_comprehensive_evaluation(self):\n        \"\"\"\n        Run comprehensive model evaluation\n        \n        Returns:\n            dict: Complete evaluation results\n        \"\"\"\n        print(f\"Starting comprehensive evaluation for {self.model_architecture} model...\")\n        \n        # Load test data\n        self.load_test_data()\n        \n        # Basic performance evaluation\n        results = self.evaluate_model_performance()\n        \n        # ROC analysis\n        predictions = np.array(results['predictions'])\n        results['roc_metrics'] = self.calculate_roc_metrics(predictions)\n        \n        # Confidence analysis\n        results['confidence_analysis'] = self.analyze_prediction_confidence(predictions)\n        \n        # Misclassified samples analysis\n        results['top_misclassified'] = self.find_misclassified_samples(predictions)\n        \n        # Generate plots\n        output_dir = f'evaluation_plots_{self.model_architecture}'\n        self.generate_evaluation_plots(results, output_dir)\n        \n        # Print summary\n        self._print_evaluation_summary(results)\n        \n        # Save detailed results\n        results_file = f'detailed_evaluation_{self.model_architecture}.json'\n        with open(results_file, 'w') as f:\n            json.dump(results, f, indent=2, default=str)\n        \n        print(f\"\\nDetailed evaluation results saved to: {results_file}\")\n        \n        return results\n    \n    def _print_evaluation_summary(self, results):\n        \"\"\"\n        Print evaluation summary\n        \"\"\"\n        print(\"\\n\" + \"=\"*80)\n        print(f\"EVALUATION SUMMARY - {self.model_architecture.upper()} MODEL\")\n        print(\"=\"*80)\n        \n        # Overall metrics\n        print(f\"Test Accuracy: {results['test_accuracy']:.4f}\")\n        print(f\"Test Top-2 Accuracy: {results['test_top2_accuracy']:.4f}\")\n        print(f\"Test Loss: {results['test_loss']:.4f}\")\n        \n        # Per-class accuracy\n        print(\"\\nPer-Class Accuracy:\")\n        for class_name, acc in results['per_class_accuracy'].items():\n            print(f\"  {class_name.capitalize()}: {acc:.4f}\")\n        \n        # ROC AUC scores\n        if 'roc_metrics' in results:\n            print(\"\\nROC AUC Scores:\")\n            for class_name, auc_score in results['roc_metrics']['auc'].items():\n                if class_name != 'macro':\n                    print(f\"  {class_name.capitalize()}: {auc_score:.4f}\")\n            print(f\"  Macro Average: {results['roc_metrics']['auc']['macro']:.4f}\")\n        \n        # Confidence analysis\n        if 'confidence_analysis' in results:\n            conf_stats = results['confidence_analysis']['confidence_stats']\n            conf_dist = results['confidence_analysis']['confidence_distribution']\n            print(f\"\\nConfidence Analysis:\")\n            print(f\"  Mean Confidence: {conf_stats['mean_confidence']:.4f}\")\n            print(f\"  High Confidence Predictions: {conf_dist['high_confidence_pct']:.2%}\")\n            print(f\"  Medium Confidence Predictions: {conf_dist['medium_confidence_pct']:.2%}\")\n            print(f\"  Low Confidence Predictions: {conf_dist['low_confidence_pct']:.2%}\")\n        \n        # Top misclassified\n        if results['top_misclassified']:\n            print(f\"\\nTop 3 Misclassified Samples:\")\n            for i, sample in enumerate(results['top_misclassified'][:3]):\n                print(f\"  {i+1}. True: {sample['true_class']} → Predicted: {sample['predicted_class']} \"\n                      f\"(Confidence: {sample['confidence']:.2%})\")\n        \n        print(\"=\"*80)\n\n\ndef main():\n    \"\"\"\n    Main function for model evaluation\n    \"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(description='Evaluate Skin Type Classification Model')\n    parser.add_argument('--data_dir', '-d', type=str, default='data',\n                       help='Path to data directory')\n    parser.add_argument('--model', '-m', type=str, help='Path to model file')\n    parser.add_argument('--architecture', '-a', type=str, default='efficientnet',\n                       choices=['efficientnet', 'resnet50', 'mobilenet', 'custom'],\n                       help='Model architecture')\n    \n    args = parser.parse_args()\n    \n    try:\n        # Create evaluator\n        evaluator = ModelEvaluator(\n            data_dir=args.data_dir,\n            model_path=args.model,\n            model_architecture=args.architecture\n        )\n        \n        # Run evaluation\n        results = evaluator.run_comprehensive_evaluation()\n        \n        print(\"\\nEvaluation completed successfully!\")\n        \n    except Exception as e:\n        print(f\"Evaluation failed: {str(e)}\")\n        raise e\n\n\nif __name__ == \"__main__\":\n    main()
