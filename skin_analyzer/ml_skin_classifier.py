"""
Machine Learning Skin Classifier - Using Scikit-Learn
Works without TensorFlow, uses traditional ML approaches
"""

import os
import numpy as np
import cv2
from PIL import Image
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

class MLSkinClassifier:
    """
    Machine Learning based skin type classifier using traditional ML algorithms
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the ML classifier
        
        Args:
            model_type (str): Type of ML model ('random_forest', 'gradient_boost', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        self.feature_names = []
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def extract_features(self, image_path):
        """
        Extract comprehensive features from an image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Feature vector
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path
            
            # Resize to standard size
            image = cv2.resize(image, (224, 224))
            
            features = []
            
            # 1. Color features (RGB)
            mean_rgb = np.mean(image, axis=(0, 1))
            std_rgb = np.std(image, axis=(0, 1))
            features.extend(mean_rgb)
            features.extend(std_rgb)
            
            # 2. HSV color features
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            mean_hsv = np.mean(hsv, axis=(0, 1))
            std_hsv = np.std(hsv, axis=(0, 1))
            features.extend(mean_hsv)
            features.extend(std_hsv)
            
            # 3. LAB color features
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            mean_lab = np.mean(lab, axis=(0, 1))
            std_lab = np.std(lab, axis=(0, 1))
            features.extend(mean_lab)
            features.extend(std_lab)
            
            # 4. Texture features
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Texture variance
            texture_variance = np.var(gray)
            features.append(texture_variance)
            
            # Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            mean_edge = np.mean(edge_magnitude)
            std_edge = np.std(edge_magnitude)
            features.extend([mean_edge, std_edge])
            
            # 5. Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            features.extend([brightness, contrast])
            
            # 6. Histogram features
            hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])
            
            # Histogram statistics
            features.extend([
                np.mean(hist_r), np.std(hist_r),
                np.mean(hist_g), np.std(hist_g),
                np.mean(hist_b), np.std(hist_b)
            ])
            
            # 7. Skin-specific features
            # Red/Green ratio (for redness detection)
            rg_ratio = np.mean(mean_rgb[0] / (mean_rgb[1] + 1e-6))
            features.append(rg_ratio)
            
            # Shine detection (bright spots)
            bright_threshold = np.percentile(gray, 90)
            shine_pixels = np.sum(gray > bright_threshold) / gray.size
            features.append(shine_pixels)
            
            # Texture uniformity
            glcm = self._calculate_glcm_features(gray)
            features.extend(glcm)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return None
    
    def _calculate_glcm_features(self, gray_image):
        """
        Calculate Gray-Level Co-occurrence Matrix features
        """
        try:
            # Simplified GLCM calculation
            # Calculate local binary patterns for texture
            def local_binary_pattern(image, radius=1):
                h, w = image.shape
                lbp = np.zeros_like(image)
                
                for i in range(radius, h - radius):
                    for j in range(radius, w - radius):
                        center = image[i, j]
                        pattern = 0
                        for k, (di, dj) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), 
                                                     (1,1), (1,0), (1,-1), (0,-1)]):
                            if image[i+di, j+dj] >= center:
                                pattern |= (1 << k)
                        lbp[i, j] = pattern
                
                return lbp
            
            # Calculate LBP
            lbp = local_binary_pattern(gray_image)
            
            # LBP statistics
            lbp_mean = np.mean(lbp)
            lbp_std = np.std(lbp)
            lbp_uniformity = len(np.unique(lbp))
            
            return [lbp_mean, lbp_std, lbp_uniformity]
            
        except:
            return [0, 0, 0]
    
    def load_dataset(self, data_dir):
        """
        Load dataset and extract features
        
        Args:
            data_dir (str): Path to the dataset directory
            
        Returns:
            tuple: (features, labels, file_paths)
        """
        print("Loading dataset and extracting features...")
        
        features = []
        labels = []
        file_paths = []
        
        # Load training data
        train_dir = os.path.join(data_dir, 'train')
        
        for skin_type in self.skin_types:
            type_dir = os.path.join(train_dir, skin_type)
            if not os.path.exists(type_dir):
                continue
            
            image_files = [f for f in os.listdir(type_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {skin_type} samples: {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=f"Extracting {skin_type} features"):
                img_path = os.path.join(type_dir, img_file)
                
                # Extract features
                feature_vector = self.extract_features(img_path)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(skin_type)
                    file_paths.append(img_path)
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"Feature extraction completed: {len(features)} samples")
        print(f"Feature vector size: {features.shape[1]}")
        
        return features, labels, file_paths
    
    def train(self, data_dir, test_size=0.2):
        """
        Train the ML model
        
        Args:
            data_dir (str): Path to the dataset directory
            test_size (float): Fraction of data to use for testing
        """
        print(f"Training {self.model_type} classifier...")
        
        # Load and prepare data
        features, labels, file_paths = self.load_dataset(data_dir)
        
        if len(features) == 0:
            raise ValueError("No features extracted. Please check the dataset.")
        
        # Store feature names for later reference
        self.feature_names = [
            'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
            'mean_h', 'mean_s', 'mean_v', 'std_h', 'std_s', 'std_v',
            'mean_l', 'mean_a', 'mean_b_lab', 'std_l', 'std_a', 'std_b_lab',
            'texture_variance', 'mean_edge', 'std_edge', 'brightness', 'contrast',
            'hist_r_mean', 'hist_r_std', 'hist_g_mean', 'hist_g_std', 
            'hist_b_mean', 'hist_b_std', 'rg_ratio', 'shine_pixels',
            'lbp_mean', 'lbp_std', 'lbp_uniformity'
        ]
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=test_size, 
            random_state=42, stratify=encoded_labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        test_predictions_labels = self.label_encoder.inverse_transform(test_predictions)
        test_true_labels = self.label_encoder.inverse_transform(y_test)
        
        print("\nClassification Report:")
        print(classification_report(test_true_labels, test_predictions_labels))
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance()
        
        # Confusion matrix
        self._plot_confusion_matrix(test_true_labels, test_predictions_labels)
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(
                test_true_labels, test_predictions_labels, output_dict=True
            )
        }
    
    def predict(self, image_path):
        """
        Predict skin type for a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Please train the model first.")
        
        # Extract features
        features = self.extract_features(image_path)
        
        if features is None:
            return {'error': 'Failed to extract features from image'}
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Convert to skin type names
        predicted_skin_type = self.label_encoder.inverse_transform([prediction])[0]
        prob_dict = {
            skin_type: float(prob) 
            for skin_type, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        confidence = float(probabilities.max())
        
        return {
            'skin_type': predicted_skin_type,
            'confidence': confidence,
            'probabilities': prob_dict,
            'image_path': image_path
        }
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        if not hasattr(self.model, 'feature_importances_'):
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importance - {self.model_type.title()}")
        plt.bar(range(min(20, len(importances))), importances[indices[:20]])
        plt.xticks(range(min(20, len(importances))), 
                   [self.feature_names[i] for i in indices[:20]], rotation=45)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=self.skin_types)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.skin_types, yticklabels=self.skin_types)
        plt.title(f'Confusion Matrix - {self.model_type.title()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'skin_types': self.skin_types
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.skin_types = model_data['skin_types']
            self.is_trained = True
            
            print(f"Model loaded from: {model_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")


def main():
    """Main function to train and test the ML classifier"""
    print("🤖 MACHINE LEARNING SKIN TYPE CLASSIFIER")
    print("=" * 60)
    
    data_dir = 'data'
    
    # Train different models
    models = ['random_forest', 'gradient_boost', 'svm']
    results = {}
    
    for model_type in models:
        print(f"\n{'=' * 20} Training {model_type.upper()} {'=' * 20}")
        
        try:
            # Create and train classifier
            classifier = MLSkinClassifier(model_type=model_type)
            
            # Train model
            training_results = classifier.train(data_dir)
            results[model_type] = training_results
            
            # Save model
            model_path = f'ml_skin_classifier_{model_type}.pkl'
            classifier.save_model(model_path)
            
            print(f"✅ {model_type.title()} training completed!")
            
        except Exception as e:
            print(f"❌ {model_type.title()} training failed: {str(e)}")
            results[model_type] = {'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for model_type, result in results.items():
        if 'error' in result:
            print(f"{model_type.upper():<15} - FAILED: {result['error']}")
        else:
            print(f"{model_type.upper():<15} - Test Accuracy: {result['test_accuracy']:.4f}")
    
    print("=" * 60)
    print("Training completed! Models saved as .pkl files.")


if __name__ == "__main__":
    main()
