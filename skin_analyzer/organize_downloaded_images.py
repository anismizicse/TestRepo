#!/usr/bin/env python3
"""
Manual Image Organization Script
Run this after manually downloading images to organize them properly
"""

import os
import shutil
from PIL import Image
import hashlib

class ImageOrganizer:
    def __init__(self, downloads_dir="downloads", target_dir="training_dataset"):
        self.downloads_dir = downloads_dir
        self.target_dir = target_dir
        
    def organize_images(self):
        """Organize downloaded images into proper structure"""
        
        if not os.path.exists(self.downloads_dir):
            print(f"❌ Downloads directory {self.downloads_dir} not found")
            print("📝 Please create the directory and add your downloaded images")
            return
        
        # Skin type keywords for auto-classification
        keywords = {
            'dry': ['dry', 'flaky', 'rough', 'dehydrated'],
            'oily': ['oily', 'shiny', 'greasy', 'acne'],
            'sensitive': ['sensitive', 'red', 'irritated', 'reactive'],
            'combination': ['combination', 'mixed', 'tzone'],
            'normal': ['normal', 'balanced', 'healthy']
        }
        
        processed = 0
        
        for filename in os.listdir(self.downloads_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Try to auto-classify based on filename
                skin_type = self.classify_by_filename(filename, keywords)
                
                if skin_type:
                    self.move_image(filename, skin_type)
                    processed += 1
                else:
                    print(f"❓ Could not classify: {filename}")
                    print("   Please manually move to appropriate folder")
        
        print(f"✅ Processed {processed} images")
    
    def classify_by_filename(self, filename, keywords):
        """Classify image based on filename keywords"""
        filename_lower = filename.lower()
        
        for skin_type, terms in keywords.items():
            for term in terms:
                if term in filename_lower:
                    return skin_type
        return None
    
    def move_image(self, filename, skin_type):
        """Move image to appropriate training directory"""
        source = os.path.join(self.downloads_dir, filename)
        
        # Check image quality
        try:
            with Image.open(source) as img:
                width, height = img.size
                if width < 224 or height < 224:
                    print(f"⚠️  Low resolution: {filename} ({width}x{height})")
                    return
        except:
            print(f"❌ Invalid image: {filename}")
            return
        
        # Move to train directory (you can manually redistribute later)
        target_dir = os.path.join(self.target_dir, 'train', skin_type)
        os.makedirs(target_dir, exist_ok=True)
        
        # Create unique filename to avoid conflicts
        name, ext = os.path.splitext(filename)
        counter = 1
        target_path = os.path.join(target_dir, filename)
        
        while os.path.exists(target_path):
            new_name = f"{name}_{counter}{ext}"
            target_path = os.path.join(target_dir, new_name)
            counter += 1
        
        shutil.move(source, target_path)
        print(f"✅ Moved {filename} → {skin_type}/")

if __name__ == "__main__":
    organizer = ImageOrganizer()
    organizer.organize_images()
