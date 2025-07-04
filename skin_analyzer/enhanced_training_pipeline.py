#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Skin Type Classification
Comprehensive approach to improve model accuracy through:
1. Better data collection and augmentation
2. Advanced feature engineering
3. Hyperparameter optimization
4. Cross-validation and ensemble methods
5. Real-world data validation
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, learning_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_skin_classifier import MLSkinClassifier

class EnhancedSkinTrainingPipeline:
    """
    Enhanced training pipeline for better skin type classification accuracy
    """
    
    def __init__(self, data_dir="dataset", models_dir="trained_models"):
        """
        Initialize the enhanced training pipeline
        
        Args:
            data_dir (str): Directory containing training data
            models_dir (str): Directory to save trained models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(f"{self.models_dir}/reports", exist_ok=True)
        
        # Training configuration
        self.config = {
            'test_size': 0.2,
            'validation_size': 0.2,
            'cv_folds': 5,
            'random_state': 42,
            'n_jobs': -1,
            'image_size': (224, 224),
            'augmentation_factor': 3,  # How many augmented versions per image
        }
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.best_models = {}
        
        print("🚀 Enhanced Skin Type Training Pipeline Initialized")
        print(f"📁 Data Directory: {self.data_dir}")
        print(f"💾 Models Directory: {self.models_dir}")
    
    def collect_and_prepare_data(self):
        """
        Comprehensive data collection and preparation strategy
        """
        print("\n📊 STEP 1: DATA COLLECTION & PREPARATION")
        print("="*50)
        
        # Check existing data
        if not os.path.exists(self.data_dir):
            print(f"❌ Data directory {self.data_dir} not found!")
            self._create_sample_dataset()
        
        # Load and analyze dataset
        X, y, image_paths = self._load_dataset_with_augmentation()
        
        # Dataset analysis
        self._analyze_dataset(y, image_paths)
        
        # Split data strategically
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config['test_size'] + self.config['validation_size'],
            stratify=y, random_state=self.config['random_state']
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5,
            stratify=y_temp, random_state=self.config['random_state']
        )
        
        print(f"📈 Final Dataset Split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _load_dataset_with_augmentation(self):
        """
        Load dataset with intelligent augmentation
        """
        print("📁 Loading dataset with augmentation...")
        
        X = []
        y = []
        image_paths = []
        
        # Load original images
        for skin_type in self.skin_types:
            skin_dir = os.path.join(self.data_dir, skin_type)
            if not os.path.exists(skin_dir):
                print(f"⚠️  Directory {skin_dir} not found, skipping...")
                continue
            
            image_files = [f for f in os.listdir(skin_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"   {skin_type}: {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=f"Processing {skin_type}"):
                img_path = os.path.join(skin_dir, img_file)
                
                # Original image
                features = self._extract_enhanced_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(skin_type)
                    image_paths.append(img_path)
                
                # Generate augmented versions
                augmented_images = self._generate_augmentations(img_path)
                for aug_img in augmented_images:
                    aug_features = self._extract_enhanced_features(aug_img, is_array=True)
                    if aug_features is not None:
                        X.append(aug_features)
                        y.append(skin_type)
                        image_paths.append(f"{img_path}_augmented")
        
        return np.array(X), np.array(y), image_paths
    
    def _generate_augmentations(self, image_path):
        """
        Generate realistic augmentations for skin images
        """
        try:
            img = Image.open(image_path).convert('RGB')
            augmented = []
            
            # 1. Brightness variations (lighting conditions)
            enhancer = ImageEnhance.Brightness(img)
            augmented.append(np.array(enhancer.enhance(0.8)))  # Darker
            augmented.append(np.array(enhancer.enhance(1.2)))  # Brighter
            
            # 2. Contrast variations (skin texture emphasis)
            enhancer = ImageEnhance.Contrast(img)
            augmented.append(np.array(enhancer.enhance(0.9)))
            augmented.append(np.array(enhancer.enhance(1.1)))
            
            # 3. Saturation (skin tone variations)
            enhancer = ImageEnhance.Color(img)
            augmented.append(np.array(enhancer.enhance(0.9)))
            augmented.append(np.array(enhancer.enhance(1.1)))
            
            # 4. Slight blur (camera focus variations)
            blurred = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            augmented.append(np.array(blurred))
            
            # 5. Noise addition (real-world photo conditions)
            img_array = np.array(img)
            noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
            noisy = np.clip(img_array + noise, 0, 255)
            augmented.append(noisy)
            
            return augmented[:self.config['augmentation_factor']]
            
        except Exception as e:
            print(f"Error in augmentation for {image_path}: {e}")
            return []
    
    def _extract_enhanced_features(self, image_path, is_array=False):
        """
        Extract comprehensive features with domain-specific enhancements
        """
        try:
            # Load image
            if is_array:
                image = image_path
            else:
                image = cv2.imread(image_path)
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            image = cv2.resize(image, self.config['image_size'])
            features = []
            
            # 1. ADVANCED COLOR ANALYSIS
            features.extend(self._extract_color_features(image))
            
            # 2. SKIN TEXTURE ANALYSIS
            features.extend(self._extract_texture_features(image))
            
            # 3. DERMATOLOGICAL FEATURES
            features.extend(self._extract_dermatological_features(image))
            
            # 4. STATISTICAL FEATURES
            features.extend(self._extract_statistical_features(image))
            
            # 5. FREQUENCY DOMAIN FEATURES
            features.extend(self._extract_frequency_features(image))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _extract_color_features(self, image):
        """Extract advanced color-based features"""
        features = []
        
        # RGB analysis
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                np.var(channel_data)
            ])
        
        # HSV analysis (important for skin tone)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(['H', 'S', 'V']):
            channel_data = hsv[:, :, i]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75)
            ])
        
        # LAB analysis (perceptual color space)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        for i in range(3):
            channel_data = lab[:, :, i]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        # Color ratios (skin-specific)
        mean_rgb = np.mean(image, axis=(0, 1))
        features.extend([
            mean_rgb[0] / (mean_rgb[1] + 1e-6),  # R/G ratio
            mean_rgb[1] / (mean_rgb[2] + 1e-6),  # G/B ratio
            mean_rgb[0] / (mean_rgb[2] + 1e-6),  # R/B ratio
        ])
        
        return features
    
    def _extract_texture_features(self, image):
        """Extract skin texture features"""
        features = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection for pore/wrinkle analysis
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.extend([
            np.mean(edge_magnitude),
            np.std(edge_magnitude),
            np.percentile(edge_magnitude, 90),  # Strong edges
            np.sum(edge_magnitude > np.percentile(edge_magnitude, 95)) / edge_magnitude.size
        ])
        
        # Laplacian (texture sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([
            np.var(laplacian),  # Texture variance
            np.mean(np.abs(laplacian))
        ])
        
        # Local Binary Pattern approximation
        lbp_features = self._calculate_lbp_features(gray)
        features.extend(lbp_features)
        
        return features
    
    def _extract_dermatological_features(self, image):
        """Extract features relevant to dermatological analysis"""
        features = []
        
        # Shine/oil detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bright_threshold = np.percentile(gray, 95)
        shine_ratio = np.sum(gray > bright_threshold) / gray.size
        features.append(shine_ratio)
        
        # Redness analysis (sensitive skin indicator)
        redness = image[:, :, 0] - np.mean([image[:, :, 1], image[:, :, 2]], axis=0)
        features.extend([
            np.mean(redness),
            np.std(redness),
            np.sum(redness > np.percentile(redness, 90)) / redness.size
        ])
        
        # Uniformity analysis
        for channel in range(3):
            channel_data = image[:, :, channel]
            # Calculate local variance
            kernel = np.ones((5, 5)) / 25
            local_mean = cv2.filter2D(channel_data.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((channel_data - local_mean)**2, -1, kernel)
            features.append(np.mean(local_var))
        
        return features
    
    def _extract_statistical_features(self, image):
        """Extract statistical distribution features"""
        features = []
        
        # Histogram features for each channel
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            features.extend([
                np.mean(hist),
                np.std(hist),
                np.max(hist),
                np.argmax(hist),  # Peak position
                len(hist[hist > 0.01])  # Number of significant bins
            ])
        
        return features
    
    def _extract_frequency_features(self, image):
        """Extract frequency domain features"""
        features = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # FFT analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        features.extend([
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.percentile(magnitude_spectrum, 90)
        ])
        
        return features
    
    def _calculate_lbp_features(self, gray_image):
        """Calculate Local Binary Pattern features"""
        try:
            h, w = gray_image.shape
            lbp = np.zeros_like(gray_image)
            
            # Simplified LBP calculation
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray_image[i, j]
                    pattern = 0
                    for k, (di, dj) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), 
                                                 (1,1), (1,0), (1,-1), (0,-1)]):
                        if gray_image[i+di, j+dj] >= center:
                            pattern |= (1 << k)
                    lbp[i, j] = pattern
            
            return [
                np.mean(lbp),
                np.std(lbp),
                len(np.unique(lbp[lbp > 0]))  # Texture patterns
            ]
        except:
            return [0, 0, 0]
    
    def _analyze_dataset(self, y, image_paths):
        """Analyze dataset distribution and quality"""
        print("\n📊 Dataset Analysis:")
        print("-" * 30)
        
        # Class distribution
        class_counts = Counter(y)
        total_samples = len(y)
        
        for skin_type in self.skin_types:
            count = class_counts.get(skin_type, 0)
            percentage = (count / total_samples) * 100
            print(f"   {skin_type:12}: {count:4} samples ({percentage:5.1f}%)")
        
        # Check for imbalance
        max_count = max(class_counts.values())
        min_count = min(class_counts.values()) if class_counts else 0
        imbalance_ratio = max_count / max(min_count, 1)
        
        if imbalance_ratio > 2:
            print(f"⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f})")
            print("   Consider using class weighting or SMOTE")
        else:
            print("✅ Dataset is reasonably balanced")
    
    def train_and_optimize_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Train multiple models with hyperparameter optimization
        """
        print("\n🤖 STEP 2: MODEL TRAINING & OPTIMIZATION")
        print("="*50)
        
        # Prepare data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Feature selection
        print("🔍 Performing feature selection...")
        self.feature_selector = SelectKBest(f_classif, k='all')
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_encoded)
        X_val_selected = self.feature_selector.transform(X_val_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Get top features
        feature_scores = self.feature_selector.scores_
        top_features_idx = np.argsort(feature_scores)[-20:]  # Top 20 features
        print(f"✅ Selected top features based on F-score")
        
        # Define models and hyperparameters
        models_config = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.config['random_state']),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                }
            },
            'gradient_boost': {
                'model': GradientBoostingClassifier(random_state=self.config['random_state']),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'svm': {
                'model': SVC(probability=True, random_state=self.config['random_state']),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced', None]
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=self.config['random_state'], max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (150, 75)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Train and optimize each model
        for model_name, config in models_config.items():
            print(f"\n🔧 Training {model_name.upper()}...")
            
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, 
                               random_state=self.config['random_state'])
            
            grid_search = GridSearchCV(
                config['model'], 
                config['params'],
                cv=cv,
                scoring='f1_weighted',
                n_jobs=self.config['n_jobs'],
                verbose=0
            )
            
            # Fit model
            grid_search.fit(X_train_selected, y_train_encoded)
            
            # Store best model
            self.best_models[model_name] = grid_search.best_estimator_
            
            # Evaluate on validation set
            val_score = grid_search.best_estimator_.score(X_val_selected, y_val_encoded)
            
            print(f"   Best params: {grid_search.best_params_}")
            print(f"   Validation accuracy: {val_score:.3f}")
            print(f"   CV score: {grid_search.best_score_:.3f}")
        
        # Create ensemble model
        print(f"\n🏆 Creating ensemble model...")
        ensemble_models = [
            ('rf', self.best_models['random_forest']),
            ('gb', self.best_models['gradient_boost']),
            ('svm', self.best_models['svm'])
        ]
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        ensemble.fit(X_train_selected, y_train_encoded)
        self.best_models['ensemble'] = ensemble
        
        ensemble_score = ensemble.score(X_val_selected, y_val_encoded)
        print(f"   Ensemble validation accuracy: {ensemble_score:.3f}")
        
        # Final evaluation on test set
        print(f"\n📊 FINAL TEST SET EVALUATION:")
        print("-" * 40)
        
        for model_name, model in self.best_models.items():
            test_pred = model.predict(X_test_selected)
            test_accuracy = accuracy_score(y_test_encoded, test_pred)
            
            print(f"{model_name:15}: {test_accuracy:.3f}")
        
        return X_test_selected, y_test_encoded
    
    def generate_comprehensive_report(self, X_test, y_test):
        """Generate detailed training and evaluation report"""
        print("\n📈 STEP 3: GENERATING COMPREHENSIVE REPORT")
        print("="*50)
        
        # Create report directory
        report_dir = f"{self.models_dir}/reports"
        
        # Best model analysis
        best_model_name = max(self.best_models.keys(), 
                            key=lambda x: self.best_models[x].score(X_test, y_test))
        best_model = self.best_models[best_model_name]
        
        print(f"🏆 Best performing model: {best_model_name.upper()}")
        
        # Detailed classification report
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Save detailed report
        with open(f"{report_dir}/classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"{report_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
            
            plt.figure(figsize=(12, 8))
            top_features_idx = np.argsort(feature_importance)[-20:]
            plt.barh(range(20), feature_importance[top_features_idx])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importance - {best_model_name.upper()}')
            plt.tight_layout()
            plt.savefig(f"{report_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Model comparison
        model_scores = {}
        for name, model in self.best_models.items():
            score = model.score(X_test, y_test)
            model_scores[name] = score
        
        plt.figure(figsize=(10, 6))
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        bars = plt.bar(models, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        plt.xlabel('Models')
        plt.ylabel('Test Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{report_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return best_model_name, report
    
    def save_production_models(self):
        """Save optimized models for production use"""
        print("\n💾 STEP 4: SAVING PRODUCTION MODELS")
        print("="*50)
        
        # Save each model
        for model_name, model in self.best_models.items():
            model_path = f"{self.models_dir}/{model_name}_optimized.pkl"
            joblib.dump(model, model_path)
            print(f"✅ Saved {model_name} to {model_path}")
        
        # Save preprocessing components
        joblib.dump(self.scaler, f"{self.models_dir}/scaler.pkl")
        joblib.dump(self.label_encoder, f"{self.models_dir}/label_encoder.pkl")
        joblib.dump(self.feature_selector, f"{self.models_dir}/feature_selector.pkl")
        
        # Save training configuration
        with open(f"{self.models_dir}/training_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print("✅ All preprocessing components saved")
    
    def _create_sample_dataset(self):
        """Create a sample dataset structure for demonstration"""
        print("📁 Creating sample dataset structure...")
        
        os.makedirs(self.data_dir, exist_ok=True)
        for skin_type in self.skin_types:
            os.makedirs(f"{self.data_dir}/{skin_type}", exist_ok=True)
        
        print(f"✅ Created dataset structure in {self.data_dir}")
        print("📝 Please add images to each skin type folder:")
        for skin_type in self.skin_types:
            print(f"   📁 {self.data_dir}/{skin_type}/")
    
    def run_complete_pipeline(self):
        """Run the complete enhanced training pipeline"""
        print("🚀 STARTING ENHANCED SKIN TYPE TRAINING PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Data preparation
            X_train, X_val, X_test, y_train, y_val, y_test = self.collect_and_prepare_data()
            
            # Step 2: Model training and optimization
            X_test_processed, y_test_processed = self.train_and_optimize_models(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            
            # Step 3: Comprehensive evaluation
            best_model_name, report = self.generate_comprehensive_report(
                X_test_processed, y_test_processed
            )
            
            # Step 4: Save production models
            self.save_production_models()
            
            print("\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"🏆 Best Model: {best_model_name.upper()}")
            print(f"📊 Overall Accuracy: {report['accuracy']:.3f}")
            print(f"📁 Models saved in: {self.models_dir}")
            print(f"📈 Reports saved in: {self.models_dir}/reports")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {str(e)}")
            return False

def main():
    """Main function to run the enhanced training pipeline"""
    
    # Initialize pipeline
    pipeline = EnhancedSkinTrainingPipeline()
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n💡 RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
        print("-" * 50)
        print("1. 🖼️  Collect more diverse, high-quality images")
        print("2. 👥 Include images from different demographics")
        print("3. 🔬 Add dermatologist-validated labels")
        print("4. 📱 Test with images from different devices/lighting")
        print("5. 🧠 Consider deep learning approaches for complex features")
        print("6. 🔄 Implement active learning for continuous improvement")
        print("7. 👨‍⚕️ Validate predictions with medical professionals")

if __name__ == "__main__":
    main()
