#!/usr/bin/env python3
"""
Dataset Preparation Helper
Helps organize and validate new images for training
"""

import os
import shutil
import glob
from PIL import Image
import json
from datetime import datetime

class DatasetPreparator:
    """Helper class to prepare and validate training datasets"""
    
    def __init__(self, source_dir="new_images", target_dir="new_training_data"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.skin_types = ['combination', 'dry', 'normal', 'oily', 'sensitive']
        
    def create_dataset_structure(self):
        """Create the required directory structure"""
        print("📁 Creating dataset structure...")
        
        # Create main directory
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Create subdirectories for each skin type
        for skin_type in self.skin_types:
            dir_path = os.path.join(self.target_dir, skin_type)
            os.makedirs(dir_path, exist_ok=True)
            print(f"  ✅ Created {dir_path}")
        
        print(f"📂 Dataset structure created in {self.target_dir}")
    
    def interactive_image_sorting(self):
        """Interactive tool to help sort images by skin type"""
        print("🖼️  Interactive Image Sorting")
        print("=" * 40)
        
        if not os.path.exists(self.source_dir):
            print(f"❌ Source directory '{self.source_dir}' not found!")
            print("Please place your images in the 'new_images' folder")
            return
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.source_dir, ext)))
            image_files.extend(glob.glob(os.path.join(self.source_dir, ext.upper())))
        
        if not image_files:
            print(f"❌ No images found in {self.source_dir}")
            return
        
        print(f"📊 Found {len(image_files)} images to sort")
        print("\\nSkin type options:")
        for i, skin_type in enumerate(self.skin_types, 1):
            print(f"  {i}. {skin_type}")
        print("  s. Skip this image")
        print("  q. Quit sorting")
        
        sorted_count = 0
        skipped_count = 0
        
        for i, image_path in enumerate(image_files, 1):
            try:
                # Display image info
                img = Image.open(image_path)
                filename = os.path.basename(image_path)
                
                print(f"\\n📸 Image {i}/{len(image_files)}: {filename}")
                print(f"   Size: {img.size}, Mode: {img.mode}")
                
                # Get user input
                while True:
                    choice = input(f"Enter skin type (1-{len(self.skin_types)}, s=skip, q=quit): ").strip().lower()
                    
                    if choice == 'q':
                        print("👋 Sorting stopped by user")
                        break
                    elif choice == 's':
                        print("⏭️  Skipped")
                        skipped_count += 1
                        break
                    elif choice.isdigit() and 1 <= int(choice) <= len(self.skin_types):
                        # Copy image to appropriate folder
                        skin_type = self.skin_types[int(choice) - 1]
                        target_path = os.path.join(self.target_dir, skin_type, filename)
                        
                        # Handle duplicate filenames
                        counter = 1
                        while os.path.exists(target_path):
                            name, ext = os.path.splitext(filename)
                            target_path = os.path.join(self.target_dir, skin_type, f"{name}_{counter}{ext}")
                            counter += 1
                        
                        shutil.copy2(image_path, target_path)
                        print(f"✅ Copied to {skin_type}/")
                        sorted_count += 1
                        break
                    else:
                        print("❌ Invalid choice. Please try again.")
                
                if choice == 'q':
                    break
                    
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                continue
        
        print(f"\\n📊 Sorting Summary:")
        print(f"  ✅ Sorted: {sorted_count} images")
        print(f"  ⏭️  Skipped: {skipped_count} images")
        print(f"  📁 Organized dataset in: {self.target_dir}")
    
    def batch_organize_by_filename(self):
        """Organize images based on filename patterns"""
        print("🏷️  Batch organizing by filename patterns...")
        
        if not os.path.exists(self.source_dir):
            print(f"❌ Source directory '{self.source_dir}' not found!")
            return
        
        # Define filename patterns for each skin type
        patterns = {
            'combination': ['combination', 'combo', 'mixed'],
            'dry': ['dry', 'dehydrated', 'flaky'],
            'normal': ['normal', 'balanced', 'healthy', 'clear'],
            'oily': ['oily', 'greasy', 'shiny', 'acne'],
            'sensitive': ['sensitive', 'irritated', 'reactive', 'red']
        }
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.source_dir, ext)))
            image_files.extend(glob.glob(os.path.join(self.source_dir, ext.upper())))
        
        organized_count = {skin_type: 0 for skin_type in self.skin_types}
        unmatched_files = []
        
        for image_path in image_files:
            filename = os.path.basename(image_path).lower()
            matched = False
            
            # Try to match filename to skin type patterns
            for skin_type, keywords in patterns.items():
                if any(keyword in filename for keyword in keywords):
                    # Copy to appropriate folder
                    target_path = os.path.join(self.target_dir, skin_type, os.path.basename(image_path))
                    
                    # Handle duplicates
                    counter = 1
                    while os.path.exists(target_path):
                        name, ext = os.path.splitext(os.path.basename(image_path))
                        target_path = os.path.join(self.target_dir, skin_type, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.copy2(image_path, target_path)
                    organized_count[skin_type] += 1
                    matched = True
                    break
            
            if not matched:
                unmatched_files.append(image_path)
        
        # Report results
        print("📊 Batch Organization Results:")
        total_organized = sum(organized_count.values())
        for skin_type, count in organized_count.items():
            print(f"  {skin_type}: {count} images")
        
        print(f"\\n✅ Total organized: {total_organized}")
        print(f"❓ Unmatched files: {len(unmatched_files)}")
        
        if unmatched_files:
            print("\\n📝 Unmatched files (need manual sorting):")
            for file in unmatched_files[:10]:  # Show first 10
                print(f"  - {os.path.basename(file)}")
            if len(unmatched_files) > 10:
                print(f"  ... and {len(unmatched_files) - 10} more")
    
    def validate_dataset(self):
        """Validate the organized dataset"""
        print("🔍 Validating organized dataset...")
        
        if not os.path.exists(self.target_dir):
            print(f"❌ Target directory '{self.target_dir}' not found!")
            return False
        
        validation_results = {}
        total_images = 0
        issues = []
        
        for skin_type in self.skin_types:
            skin_dir = os.path.join(self.target_dir, skin_type)
            
            if not os.path.exists(skin_dir):
                issues.append(f"Missing directory: {skin_type}")
                validation_results[skin_type] = 0
                continue
            
            # Count images
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(skin_dir, ext)))
                image_files.extend(glob.glob(os.path.join(skin_dir, ext.upper())))
            
            count = len(image_files)
            validation_results[skin_type] = count
            total_images += count
            
            # Check for minimum images
            if count < 10:
                issues.append(f"{skin_type}: Only {count} images (recommended: 10+)")
            
            # Validate image files
            corrupted_files = []
            for img_file in image_files[:5]:  # Check first 5 images
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                except Exception:
                    corrupted_files.append(os.path.basename(img_file))
            
            if corrupted_files:
                issues.append(f"{skin_type}: Corrupted files: {corrupted_files}")
        
        # Display results
        print("📊 Dataset Validation Results:")
        for skin_type, count in validation_results.items():
            status = "✅" if count >= 10 else "⚠️ "
            print(f"  {status} {skin_type}: {count} images")
        
        print(f"\\n📈 Total images: {total_images}")
        
        if issues:
            print("\\n⚠️  Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\\n✅ Dataset validation passed!")
        
        # Save validation report
        report = {
            "validation_date": datetime.now().isoformat(),
            "dataset_stats": validation_results,
            "total_images": total_images,
            "issues": issues,
            "ready_for_training": len(issues) == 0 and total_images >= 50
        }
        
        report_file = os.path.join(self.target_dir, "validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Validation report saved: {report_file}")
        return len(issues) == 0
    
    def generate_dataset_summary(self):
        """Generate a summary of the prepared dataset"""
        summary_file = os.path.join(self.target_dir, "dataset_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("# Dataset Summary\\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## Directory Structure\\n")
            f.write(f"{self.target_dir}/\\n")
            
            total_images = 0
            for skin_type in self.skin_types:
                skin_dir = os.path.join(self.target_dir, skin_type)
                if os.path.exists(skin_dir):
                    image_files = glob.glob(os.path.join(skin_dir, "*.*"))
                    count = len(image_files)
                    total_images += count
                    f.write(f"├── {skin_type}/ ({count} images)\\n")
                else:
                    f.write(f"├── {skin_type}/ (0 images) ❌\\n")
            
            f.write(f"\\nTotal Images: {total_images}\\n")
            
            f.write("\\n## Next Steps\\n")
            f.write("1. Review the organized images\\n")
            f.write("2. Add more images if any class has < 10 images\\n")
            f.write("3. Run: python3 retrain_and_deploy.py\\n")
            f.write("4. Follow the retraining guide\\n")
        
        print(f"📋 Dataset summary saved: {summary_file}")

def main():
    """Main interactive menu"""
    print("🛠️  Dataset Preparation Tool")
    print("=" * 35)
    
    preparator = DatasetPreparator()
    
    while True:
        print("\\nChoose an option:")
        print("1. Create dataset structure")
        print("2. Interactive image sorting")
        print("3. Batch organize by filename")
        print("4. Validate dataset")
        print("5. Generate dataset summary")
        print("6. Exit")
        
        choice = input("\\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            preparator.create_dataset_structure()
        elif choice == '2':
            preparator.create_dataset_structure()
            preparator.interactive_image_sorting()
        elif choice == '3':
            preparator.create_dataset_structure()
            preparator.batch_organize_by_filename()
        elif choice == '4':
            preparator.validate_dataset()
        elif choice == '5':
            preparator.generate_dataset_summary()
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
