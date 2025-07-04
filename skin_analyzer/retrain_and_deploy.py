#!/usr/bin/env python3
"""
Complete Retraining and Redeployment Pipeline
Automates the entire process of training with new data and redeploying models
"""

import os
import sys
import shutil
import json
import subprocess
from datetime import datetime
import glob

class RetrainingPipeline:
    """Handles complete retraining and redeployment workflow"""
    
    def __init__(self, new_data_dir="new_training_data", backup_dir="model_backups"):
        self.new_data_dir = new_data_dir
        self.backup_dir = backup_dir
        self.training_dataset_dir = "training_dataset"
        self.models_dir = "."
        
        # Create necessary directories
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(os.path.join(self.training_dataset_dir, "train"), exist_ok=True)
        
    def backup_existing_models(self):
        """Backup current models before retraining"""
        print("💾 Backing up existing models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = os.path.join(self.backup_dir, f"models_backup_{timestamp}")
        os.makedirs(backup_folder, exist_ok=True)
        
        # List of model files to backup
        model_files = [
            'ensemble_skin_classifier.pkl',
            'random_forest_optimized.pkl',
            'feature_scaler.pkl',
            'feature_selector.pkl',
            'label_encoder.pkl',
            'scaler.pkl',
            'model_metadata.json',
            'training_report.json'
        ]
        
        backed_up_files = []
        for file in model_files:
            if os.path.exists(file):
                shutil.copy2(file, backup_folder)
                backed_up_files.append(file)
                print(f"  ✅ Backed up {file}")
        
        print(f"📁 Backup saved to: {backup_folder}")
        return backup_folder, backed_up_files
    
    def organize_new_dataset(self):
        """Organize new dataset into training structure"""
        print("📂 Organizing new dataset...")
        
        if not os.path.exists(self.new_data_dir):
            raise FileNotFoundError(f"New data directory '{self.new_data_dir}' not found!")
        
        # Expected skin type classes
        classes = ['combination', 'dry', 'normal', 'oily', 'sensitive']
        
        # Check if new data is already organized
        organized = True
        for class_name in classes:
            class_path = os.path.join(self.new_data_dir, class_name)
            if not os.path.exists(class_path):
                organized = False
                break
        
        if organized:
            print("  ✅ New data is already organized by skin type")
            # Copy to training dataset
            train_dir = os.path.join(self.training_dataset_dir, "train")
            
            for class_name in classes:
                src_dir = os.path.join(self.new_data_dir, class_name)
                dst_dir = os.path.join(train_dir, class_name)
                
                # Create destination directory
                os.makedirs(dst_dir, exist_ok=True)
                
                # Copy all images
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
                total_copied = 0
                
                for ext in image_extensions:
                    files = glob.glob(os.path.join(src_dir, ext)) + glob.glob(os.path.join(src_dir, ext.upper()))
                    for file in files:
                        dst_file = os.path.join(dst_dir, os.path.basename(file))
                        shutil.copy2(file, dst_file)
                        total_copied += 1
                
                print(f"  📁 {class_name}: {total_copied} images")
        
        else:
            print("  ⚠️  New data needs manual organization")
            print(f"  Please organize images in {self.new_data_dir} into subfolders:")
            for class_name in classes:
                print(f"    - {self.new_data_dir}/{class_name}/")
            print("  Then run this script again.")
            return False
        
        return True
    
    def validate_dataset(self):
        """Validate the dataset structure and content"""
        print("🔍 Validating dataset...")
        
        train_dir = os.path.join(self.training_dataset_dir, "train")
        classes = ['combination', 'dry', 'normal', 'oily', 'sensitive']
        
        dataset_stats = {}
        total_images = 0
        
        for class_name in classes:
            class_dir = os.path.join(train_dir, class_name)
            if os.path.exists(class_dir):
                image_files = []
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(class_dir, ext)))
                    image_files.extend(glob.glob(os.path.join(class_dir, ext.upper())))
                
                count = len(image_files)
                dataset_stats[class_name] = count
                total_images += count
                print(f"  📊 {class_name}: {count} images")
            else:
                dataset_stats[class_name] = 0
                print(f"  ❌ {class_name}: Directory not found")
        
        print(f"  📈 Total images: {total_images}")
        
        # Check for minimum requirements
        min_images_per_class = 10
        insufficient_classes = [cls for cls, count in dataset_stats.items() if count < min_images_per_class]
        
        if insufficient_classes:
            print(f"  ⚠️  Warning: These classes have < {min_images_per_class} images: {insufficient_classes}")
            print("  Consider adding more images for better training results")
        
        if total_images < 50:
            print("  ⚠️  Warning: Very small dataset. Consider collecting more images.")
            return False
        
        print("  ✅ Dataset validation passed")
        return True, dataset_stats
    
    def run_training(self, use_enhanced=True):
        """Run the training process"""
        print("🚀 Starting model training...")
        
        try:
            if use_enhanced:
                print("  Using Enhanced Training Pipeline...")
                result = subprocess.run([
                    "python3", "enhanced_production_trainer.py"
                ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            else:
                print("  Using Standard Training Pipeline...")
                result = subprocess.run([
                    "python3", "train_models_for_production.py"
                ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                print("  ✅ Training completed successfully!")
                print("  📊 Training output:")
                # Show last few lines of output
                output_lines = result.stdout.split('\n')
                for line in output_lines[-10:]:
                    if line.strip():
                        print(f"    {line}")
                return True
            else:
                print("  ❌ Training failed!")
                print("  Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("  ⏰ Training timed out!")
            return False
        except Exception as e:
            print(f"  ❌ Training error: {e}")
            return False
    
    def validate_trained_models(self):
        """Validate that models were created and are functional"""
        print("🔍 Validating trained models...")
        
        # Check for required model files
        required_files = [
            'ensemble_skin_classifier.pkl',
            'feature_scaler.pkl',
            'label_encoder.pkl',
            'feature_selector.pkl'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"  ❌ Missing model files: {missing_files}")
            return False
        
        # Test model loading
        try:
            import joblib
            import numpy as np
            
            # Load and test ensemble model
            model = joblib.load('ensemble_skin_classifier.pkl')
            scaler = joblib.load('feature_scaler.pkl')
            label_encoder = joblib.load('label_encoder.pkl')
            feature_selector = joblib.load('feature_selector.pkl')
            
            # Test with dummy data
            dummy_features = np.random.rand(1, 79)
            features_scaled = scaler.transform(dummy_features)
            features_selected = feature_selector.transform(features_scaled)
            prediction = model.predict(features_selected)[0]
            probabilities = model.predict_proba(features_selected)[0]
            
            skin_type = label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            print(f"  ✅ Model test successful!")
            print(f"    Classes: {label_encoder.classes_}")
            print(f"    Test prediction: {skin_type} (confidence: {confidence:.3f})")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Model validation failed: {e}")
            return False
    
    def update_api_models(self):
        """Update API with new models"""
        print("🔄 Updating API models...")
        
        # Check if API files exist
        api_files = ['api_server.py', 'api_production.py']
        api_file = None
        
        for file in api_files:
            if os.path.exists(file):
                api_file = file
                break
        
        if not api_file:
            print("  ⚠️  No API file found. Manual deployment required.")
            return True
        
        print(f"  📝 API file detected: {api_file}")
        print("  ✅ Models are ready for API use")
        print("  💡 Restart your API server to use the new models")
        
        return True
    
    def generate_deployment_report(self, backup_folder, dataset_stats):
        """Generate a deployment report"""
        print("📋 Generating deployment report...")
        
        # Load training results if available
        training_results = {}
        if os.path.exists('training_report.json'):
            with open('training_report.json', 'r') as f:
                training_results = json.load(f)
        
        report = {
            "deployment_info": {
                "timestamp": datetime.now().isoformat(),
                "backup_location": backup_folder,
                "dataset_stats": dataset_stats,
                "total_images": sum(dataset_stats.values())
            },
            "training_results": training_results,
            "model_files": {
                "ensemble_model": "ensemble_skin_classifier.pkl",
                "feature_scaler": "feature_scaler.pkl", 
                "label_encoder": "label_encoder.pkl",
                "feature_selector": "feature_selector.pkl"
            },
            "next_steps": [
                "Restart API server to load new models",
                "Test with sample images to verify functionality",
                "Monitor model performance in production",
                "Consider collecting more data for underperforming classes"
            ]
        }
        
        report_file = f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  📄 Report saved: {report_file}")
        return report_file
    
    def run_complete_pipeline(self, use_enhanced=True):
        """Run the complete retraining and deployment pipeline"""
        print("🔄 Starting Complete Retraining Pipeline")
        print("=" * 50)
        
        try:
            # Step 1: Backup existing models
            backup_folder, backed_up_files = self.backup_existing_models()
            
            # Step 2: Organize new dataset
            if not self.organize_new_dataset():
                return False
            
            # Step 3: Validate dataset
            valid, dataset_stats = self.validate_dataset()
            if not valid:
                return False
            
            # Step 4: Run training
            if not self.run_training(use_enhanced):
                print("❌ Training failed. Restoring backup...")
                # Restore backup if training fails
                for file in backed_up_files:
                    backup_file = os.path.join(backup_folder, file)
                    if os.path.exists(backup_file):
                        shutil.copy2(backup_file, file)
                return False
            
            # Step 5: Validate trained models
            if not self.validate_trained_models():
                print("❌ Model validation failed. Restoring backup...")
                # Restore backup
                for file in backed_up_files:
                    backup_file = os.path.join(backup_folder, file)
                    if os.path.exists(backup_file):
                        shutil.copy2(backup_file, file)
                return False
            
            # Step 6: Update API
            self.update_api_models()
            
            # Step 7: Generate report
            report_file = self.generate_deployment_report(backup_folder, dataset_stats)
            
            print("\n🎉 Retraining and Deployment Complete!")
            print("=" * 50)
            print("✅ All steps completed successfully")
            print(f"📁 Backup location: {backup_folder}")
            print(f"📄 Deployment report: {report_file}")
            print("\n🚀 Next Steps:")
            print("1. Restart your API server")
            print("2. Test with sample images")
            print("3. Monitor performance")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function with user interaction"""
    print("🔄 Skin Classification Model Retraining Pipeline")
    print("=" * 55)
    
    # Get user inputs
    new_data_dir = input("Enter path to new training data directory [new_training_data]: ").strip()
    if not new_data_dir:
        new_data_dir = "new_training_data"
    
    use_enhanced = input("Use enhanced training pipeline? (y/n) [y]: ").strip().lower()
    use_enhanced = use_enhanced != 'n'
    
    print(f"\n📂 New data directory: {new_data_dir}")
    print(f"🧠 Training method: {'Enhanced' if use_enhanced else 'Standard'}")
    
    confirm = input("\nProceed with retraining? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Retraining cancelled")
        return
    
    # Run pipeline
    pipeline = RetrainingPipeline(new_data_dir)
    success = pipeline.run_complete_pipeline(use_enhanced)
    
    if success:
        print("\n🎯 Retraining completed successfully!")
    else:
        print("\n❌ Retraining failed. Check logs above.")

if __name__ == "__main__":
    main()
