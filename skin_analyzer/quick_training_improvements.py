#!/usr/bin/env python3
"""
Quick-Start Training Script for Immediate Improvement
Run this to improve your current models with existing data
"""

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_skin_classifier import MLSkinClassifier

class QuickTrainingImprover:
    """
    Quick training improvements you can apply immediately
    """
    
    def __init__(self, data_dir="sample_faces"):
        self.data_dir = data_dir
        self.models_dir = "optimized_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def improve_existing_models(self):
        """
        Apply immediate improvements to existing models
        """
        print("🚀 QUICK MODEL IMPROVEMENTS")
        print("="*40)
        
        # Load existing data from sample faces
        X, y = self._load_sample_data()
        
        if len(X) == 0:
            print("❌ No sample data found. Please add images to sample_faces/")
            return False
        
        print(f"📊 Loaded {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Improved Random Forest with optimized parameters
        print("🌳 Training optimized Random Forest...")
        
        rf_params = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1_weighted')
        rf_grid.fit(X_train_scaled, y_train)
        
        # Evaluate
        rf_pred = rf_grid.best_estimator_.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        print(f"✅ Optimized Random Forest Accuracy: {rf_accuracy:.3f}")
        print(f"📋 Best Parameters: {rf_grid.best_params_}")
        
        # Save optimized model
        joblib.dump(rf_grid.best_estimator_, f"{self.models_dir}/optimized_random_forest.pkl")
        joblib.dump(scaler, f"{self.models_dir}/optimized_scaler.pkl")
        
        # Generate report
        report = classification_report(y_test, rf_pred, output_dict=True)
        
        with open(f"{self.models_dir}/optimization_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Saved optimized model to {self.models_dir}/")
        return True
    
    def _load_sample_data(self):
        """Load data from sample_faces directory"""
        X = []
        y = []
        
        classifier = MLSkinClassifier()
        
        # Look for images in sample_faces
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.data_dir, filename)
                    
                    # Extract features
                    features = classifier.extract_features(img_path)
                    if features is not None:
                        X.append(features)
                        
                        # Simple labeling based on filename (you should improve this)
                        if 'oily' in filename.lower():
                            y.append('oily')
                        elif 'dry' in filename.lower():
                            y.append('dry')
                        elif 'sensitive' in filename.lower():
                            y.append('sensitive')
                        elif 'combination' in filename.lower():
                            y.append('combination')
                        else:
                            y.append('normal')
        
        return np.array(X), np.array(y)
    
    def create_enhanced_predictor(self):
        """Create an enhanced version of the current predictor"""
        
        enhanced_predictor_code = '''
#!/usr/bin/env python3
"""
Enhanced Predictor with Optimized Models
Uses the optimized models for better accuracy
"""

import os
import numpy as np
import joblib
from ml_skin_classifier import MLSkinClassifier

class EnhancedSkinPredictor:
    """Enhanced skin type predictor using optimized models"""
    
    def __init__(self):
        self.models_dir = "optimized_models"
        self.base_classifier = MLSkinClassifier()
        
        # Load optimized components
        try:
            self.model = joblib.load(f"{self.models_dir}/optimized_random_forest.pkl")
            self.scaler = joblib.load(f"{self.models_dir}/optimized_scaler.pkl")
            self.is_loaded = True
            print("✅ Loaded optimized model")
        except:
            print("❌ Optimized model not found, using default")
            self.is_loaded = False
    
    def predict_skin_type(self, image_path):
        """Predict skin type with enhanced accuracy"""
        
        # Extract features
        features = self.base_classifier.extract_features(image_path)
        if features is None:
            return {"error": "Could not process image"}
        
        if self.is_loaded:
            # Use optimized model
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get class names
            classes = self.model.classes_
            prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}
            
            return {
                "skin_type": prediction,
                "confidence": max(probabilities),
                "probabilities": prob_dict,
                "model": "optimized_random_forest"
            }
        else:
            # Fallback to default
            return {"error": "Optimized model not available"}

# Usage example
if __name__ == "__main__":
    predictor = EnhancedSkinPredictor()
    
    # Test with an image
    result = predictor.predict_skin_type("sample_faces/sample_face_image1.jpg")
    print(f"Result: {result}")
        '''
        
        with open("enhanced_predictor.py", 'w') as f:
            f.write(enhanced_predictor_code)
        
        print("✅ Created enhanced_predictor.py")

def generate_training_recommendations():
    """Generate specific recommendations for improving accuracy"""
    
    recommendations = {
        "IMMEDIATE_IMPROVEMENTS": [
            "🎯 Use class_weight='balanced' in RandomForest to handle imbalanced data",
            "📊 Apply StandardScaler to normalize features before training", 
            "🔧 Tune hyperparameters using GridSearchCV or RandomizedSearchCV",
            "🎲 Use stratified sampling to maintain class distribution",
            "📈 Increase n_estimators to 200-300 for Random Forest"
        ],
        
        "DATA_IMPROVEMENTS": [
            "📸 Collect more diverse images (different lighting, angles, skin tones)",
            "🏷️ Get expert dermatologist validation for labels",
            "🔄 Use data augmentation (brightness, contrast, rotation)",
            "⚖️ Balance your dataset - aim for equal samples per class",
            "🧹 Clean data by removing poor quality or mislabeled images"
        ],
        
        "FEATURE_IMPROVEMENTS": [
            "🎨 Add more color space features (LAB, HSV, YUV)",
            "🔍 Include texture analysis (GLCM, LBP, Gabor filters)",
            "📏 Extract dermatology-specific features (shine, redness, pores)",
            "📊 Use feature selection to identify most important features",
            "🧮 Consider deep learning features from pre-trained CNNs"
        ],
        
        "MODEL_IMPROVEMENTS": [
            "🎭 Use ensemble methods (Voting, Bagging, Stacking)",
            "🧠 Try advanced algorithms (XGBoost, CatBoost, Neural Networks)",
            "🎯 Implement cross-validation for robust evaluation",
            "📈 Use learning curves to detect overfitting/underfitting",
            "🔄 Apply techniques like SMOTE for minority class oversampling"
        ],
        
        "VALIDATION_IMPROVEMENTS": [
            "👨‍⚕️ Get dermatologist validation on predictions",
            "📊 Use stratified k-fold cross-validation",
            "🔍 Analyze confusion matrix for specific error patterns",
            "📈 Track multiple metrics (precision, recall, F1-score)",
            "🧪 Test on completely separate validation set"
        ]
    }
    
    return recommendations

def main():
    """Main function to run quick improvements"""
    
    print("🚀 SKIN TYPE CLASSIFICATION - QUICK TRAINING IMPROVEMENTS")
    print("="*65)
    
    # Initialize improver
    improver = QuickTrainingImprover()
    
    # Run improvements
    success = improver.improve_existing_models()
    
    if success:
        # Create enhanced predictor
        improver.create_enhanced_predictor()
        
        print("\n🎉 IMPROVEMENTS COMPLETED!")
        print("-" * 30)
        print("✅ Optimized Random Forest model saved")
        print("✅ Enhanced predictor created")
        print("✅ Training report generated")
    
    # Generate recommendations
    recommendations = generate_training_recommendations()
    
    print("\n💡 RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
    print("="*50)
    
    for category, items in recommendations.items():
        print(f"\n{category.replace('_', ' ')}:")
        for item in items:
            print(f"  {item}")
    
    print("\n🎯 NEXT STEPS:")
    print("-" * 20)
    print("1. Run: python enhanced_training_pipeline.py (for comprehensive training)")
    print("2. Run: python data_collection_guide.py (for data collection help)")
    print("3. Use enhanced_predictor.py in your web application")
    print("4. Collect more diverse, high-quality training data")
    print("5. Get expert validation for your predictions")

if __name__ == "__main__":
    main()
