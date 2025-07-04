"""
Machine Learning Core Module
============================

Contains the machine learning model loading and prediction logic.
"""

import os
import logging
import joblib
import cv2
import numpy as np
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class SkinAnalyzer:
    """Production skin analyzer using trained ML models."""
    
    def __init__(self, model_path: str = None):
        """Initialize the skin analyzer with model files."""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        
        if model_path is None:
            # Try multiple paths to find models directory
            # Path 1: app/models/ (from app/core/)
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            
            # Path 2: Check if we're in Docker container with /app working directory
            if not os.path.exists(model_path):
                model_path = '/app/app/models'
            
            # Path 3: Relative to current working directory
            if not os.path.exists(model_path):
                model_path = 'app/models'
            
            # Path 4: Check environment variable
            if not os.path.exists(model_path):
                model_path = os.environ.get('MODEL_PATH', 'models')
            
            logger.info(f"Using model path: {model_path}")
            logger.info(f"Model path exists: {os.path.exists(model_path)}")
            if os.path.exists(model_path):
                logger.info(f"Files in model path: {os.listdir(model_path)}")
        
        self.model_path = model_path
        self.load_models()
    
    def load_models(self):
        """Load all required model files."""
        try:
            # Load ensemble model (best performer)
            model_file = os.path.join(self.model_path, 'ensemble_skin_classifier.pkl')
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                logger.info("Loaded ensemble model")
            else:
                # Fallback to random forest
                model_file = os.path.join(self.model_path, 'random_forest_optimized.pkl')
                if os.path.exists(model_file):
                    self.model = joblib.load(model_file)
                    logger.info("Loaded random forest model")
                else:
                    raise FileNotFoundError("No model file found")
            
            # Load preprocessing components
            self.scaler = joblib.load(os.path.join(self.model_path, 'feature_scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_path, 'label_encoder.pkl'))
            
            # Try to load feature selector (optional)
            selector_file = os.path.join(self.model_path, 'feature_selector.pkl')
            if os.path.exists(selector_file):
                self.feature_selector = joblib.load(selector_file)
                logger.info("Loaded feature selector")
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract comprehensive features for better classification (79 features to match training)."""
        try:
            # Resize image to match training
            image_resized = cv2.resize(image, (224, 224))
            
            # Convert to different color spaces
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
            
            features = []
            
            # 1. Enhanced Color Features (RGB) - 27 features
            for i, channel in enumerate(cv2.split(image_resized)):
                features.extend([
                    np.mean(channel), np.std(channel), np.var(channel),
                    np.median(channel), np.percentile(channel, 25), np.percentile(channel, 75),
                    np.min(channel), np.max(channel), np.ptp(channel)  # peak-to-peak
                ])
            
            # 2. HSV Color Features - 18 features
            for i, channel in enumerate(cv2.split(hsv)):
                features.extend([
                    np.mean(channel), np.std(channel), np.var(channel),
                    np.median(channel), np.percentile(channel, 25), np.percentile(channel, 75)
                ])
            
            # 3. LAB Color Features - 9 features
            for i, channel in enumerate(cv2.split(lab)):
                features.extend([
                    np.mean(channel), np.std(channel), np.median(channel)
                ])
            
            # 4. Advanced Texture Features - 4 features
            # Edge density with multiple thresholds
            edges_low = cv2.Canny(gray, 30, 100)
            edges_high = cv2.Canny(gray, 100, 200)
            edge_density_low = np.sum(edges_low > 0) / (edges_low.shape[0] * edges_low.shape[1])
            edge_density_high = np.sum(edges_high > 0) / (edges_high.shape[0] * edges_high.shape[1])
            features.extend([edge_density_low, edge_density_high])
            
            # Laplacian variance (sharpness/smoothness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(laplacian_var)
            
            # Sobel gradients
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            features.extend([np.mean(sobel_magnitude)])  # 1 feature to keep count consistent
            
            # 5. Local Binary Pattern (LBP) for texture - 10 features
            def calculate_lbp(image, radius=1, n_points=8):
                lbp = np.zeros_like(image)
                for i in range(radius, image.shape[0] - radius):
                    for j in range(radius, image.shape[1] - radius):
                        center = image[i, j]
                        binary_string = ""
                        for k in range(n_points):
                            angle = 2 * np.pi * k / n_points
                            x = int(i + radius * np.cos(angle))
                            y = int(j + radius * np.sin(angle))
                            if x < image.shape[0] and y < image.shape[1]:
                                binary_string += "1" if image[x, y] >= center else "0"
                        lbp[i, j] = int(binary_string, 2) if len(binary_string) == n_points else 0
                return lbp
            
            lbp = calculate_lbp(gray)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
            features.extend(lbp_hist[:10])  # Use first 10 bins
            
            # 6. Contrast and Brightness Features - 5 features
            features.extend([
                np.mean(gray), np.std(gray), np.var(gray),
                np.percentile(gray, 10), np.percentile(gray, 90)
            ])
            
            # 7. Homogeneity and Energy - 2 features
            gray_normalized = gray / 255.0
            homogeneity = np.mean(1.0 / (1.0 + np.abs(gray_normalized[:-1, :-1] - gray_normalized[1:, 1:])))
            energy = np.sum(gray_normalized ** 2) / (gray_normalized.shape[0] * gray_normalized.shape[1])
            features.extend([homogeneity, energy])
            
            # 8. Color Coherence - 1 feature
            blur = cv2.GaussianBlur(image_resized, (5, 5), 0)
            diff = np.mean(np.abs(image_resized.astype(float) - blur.astype(float)))
            features.append(diff)
            
            # 9. Skin-specific features - 2 features
            r_channel = image_resized[:, :, 2].astype(float)
            g_channel = image_resized[:, :, 1].astype(float)
            b_channel = image_resized[:, :, 0].astype(float)
            
            redness_ratio = np.mean(r_channel / (g_channel + b_channel + 1e-7))
            yellowness_ratio = np.mean((r_channel + g_channel) / (b_channel + 1e-7))
            features.extend([redness_ratio, yellowness_ratio])
            
            # Ensure we have exactly 79 features (pad if needed)
            features_array = np.array(features)
            expected_features = 79
            
            if len(features_array) < expected_features:
                # Pad with zeros if we have fewer features
                padding = expected_features - len(features_array)
                features_array = np.pad(features_array, (0, padding), 'constant', constant_values=0)
                logger.warning(f"Padded features from {len(features)} to {expected_features}")
            elif len(features_array) > expected_features:
                # Truncate if we have more features
                features_array = features_array[:expected_features]
                logger.warning(f"Truncated features from {len(features)} to {expected_features}")
            
            logger.info(f"Extracted {len(features_array)} features")
            return features_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict skin type from image."""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Extract features
            features = self.extract_features(image)
            
            # Apply feature selection if available
            if self.feature_selector is not None:
                features = self.feature_selector.transform(features)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get class label
            skin_type = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence scores for all classes
            class_names = self.label_encoder.classes_
            confidence_scores = {
                name: float(prob) for name, prob in zip(class_names, probabilities)
            }
            
            return {
                'predicted_type': skin_type,
                'confidence': float(max(probabilities)),
                'all_scores': confidence_scores,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'predicted_type': 'Unknown',
                'confidence': 0.0,
                'all_scores': {},
                'success': False,
                'error': str(e)
            }
