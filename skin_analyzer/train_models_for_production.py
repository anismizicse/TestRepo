#!/usr/bin/env python3
"""
Train models for production deployment
Generates the required .pkl files for the API
"""

import os
import sys
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import glob
from PIL import Image
from tqdm import tqdm

class ProductionModelTrainer:
    """Train models for production deployment"""
    
    def __init__(self, data_dir="training_dataset", models_dir="."):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_features(self, image_path):
        """Extract features from image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Resize image
            image_resized = cv2.resize(image, (224, 224))
            
            # Convert to different color spaces
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
            
            features = []
            
            # Basic color statistics (RGB)
            for channel in cv2.split(image_resized):
                features.extend([
                    np.mean(channel), np.std(channel),
                    np.median(channel), np.percentile(channel, 25), np.percentile(channel, 75)
                ])
            
            # HSV color features
            for channel in cv2.split(hsv):
                features.extend([np.mean(channel), np.std(channel)])
            
            # Texture features
            # Edge density with multiple thresholds for better texture analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Additional edge detection
            edges_low = cv2.Canny(gray, 30, 100)
            edge_density_low = np.sum(edges_low > 0) / (edges_low.shape[0] * edges_low.shape[1])
            features.append(edge_density_low)
            
            # Smoothness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(laplacian_var)
            
            # Sobel gradients for texture
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            features.extend([np.mean(sobel_magnitude), np.std(sobel_magnitude)])
            
            # Brightness and contrast
            features.extend([np.mean(gray), np.std(gray), np.var(gray)])
            
            # Additional skin-specific features
            # Redness and yellowness ratios (important for skin analysis)
            r_channel = image_resized[:, :, 2].astype(float)
            g_channel = image_resized[:, :, 1].astype(float)
            b_channel = image_resized[:, :, 0].astype(float)
            
            redness_ratio = np.mean(r_channel / (g_channel + b_channel + 1e-7))
            yellowness_ratio = np.mean((r_channel + g_channel) / (b_channel + 1e-7))
            features.extend([redness_ratio, yellowness_ratio])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_data(self):
        """Load training data"""
        print("📥 Loading training data...")
        
        X = []
        y = []
        
        train_dir = os.path.join(self.data_dir, 'train')
        classes = ['combination', 'dry', 'normal', 'oily', 'sensitive']
        
        for class_name in classes:
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found")
                continue
                
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(class_dir, ext)))
                image_files.extend(glob.glob(os.path.join(class_dir, ext.upper())))
            
            print(f"Found {len(image_files)} images for class '{class_name}'")
            
            for image_path in tqdm(image_files, desc=f"Processing {class_name}"):
                features = self.extract_features(image_path)
                if features is not None:
                    X.append(features)
                    y.append(class_name)
        
        print(f"✅ Loaded {len(X)} samples")
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train the Random Forest model"""
        print("🚀 Starting model training...")
        
        # Load data
        X, y = self.load_data()
        
        if len(X) == 0:
            raise ValueError("No training data found!")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Split data with stratification for balanced classes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train optimized Random Forest
        print("🌳 Training optimized Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate with cross-validation for more reliable assessment
        train_score = rf_model.score(X_train_scaled, y_train)
        test_score = rf_model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"📊 Training Accuracy: {train_score:.3f}")
        print(f"📊 Test Accuracy: {test_score:.3f}")
        print(f"📊 Cross-validation Score: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Detailed evaluation
        y_pred = rf_model.predict(X_test_scaled)
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return rf_model
    
    def save_models(self, model):
        """Save trained models"""
        print("💾 Saving models...")
        
        # Save model
        joblib.dump(model, os.path.join(self.models_dir, 'random_forest_optimized.pkl'))
        print("✅ Saved random_forest_optimized.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        print("✅ Saved scaler.pkl")
        
        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(self.models_dir, 'label_encoder.pkl'))
        print("✅ Saved label_encoder.pkl")
        
        print("🎉 All models saved successfully!")

def main():
    """Main training function"""
    trainer = ProductionModelTrainer()
    
    try:
        # Train model
        model = trainer.train_model()
        
        # Save models
        trainer.save_models(model)
        
        print("\n🎯 Model training completed successfully!")
        print("Models are ready for production deployment.")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
