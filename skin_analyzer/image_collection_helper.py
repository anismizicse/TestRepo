#!/usr/bin/env python3
"""
Image Collection Helper for Skin Type Training Data
Helps collect and organize training images from various sources
"""

import os
import requests
import urllib.parse
from PIL import Image
import shutil
from pathlib import Path
import json

class ImageCollectionHelper:
    """
    Helper class for collecting and organizing skin type training images
    """
    
    def __init__(self, base_dir="training_dataset"):
        self.base_dir = base_dir
        self.skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        self.setup_directories()
        
    def setup_directories(self):
        """Create organized directory structure for training data"""
        
        print("📁 Setting up training dataset directories...")
        
        # Create main directory structure
        for split in ['train', 'validation', 'test']:
            for skin_type in self.skin_types:
                dir_path = os.path.join(self.base_dir, split, skin_type)
                os.makedirs(dir_path, exist_ok=True)
        
        # Create metadata directory
        os.makedirs(os.path.join(self.base_dir, 'metadata'), exist_ok=True)
        
        print("✅ Directory structure created:")
        print(f"   📂 {self.base_dir}/")
        for split in ['train', 'validation', 'test']:
            print(f"     📂 {split}/")
            for skin_type in self.skin_types:
                print(f"       📂 {skin_type}/")
    
    def create_collection_guide(self):
        """Create a comprehensive guide for collecting images"""
        
        collection_sources = {
            "FREE_STOCK_PHOTO_SITES": {
                "Freepik": {
                    "url": "https://www.freepik.com/",
                    "search_terms": [
                        "dry skin",
                        "oily skin", 
                        "sensitive skin",
                        "normal skin",
                        "combination skin",
                        "skin texture",
                        "facial skin",
                        "dermatology"
                    ],
                    "instructions": [
                        "1. Go to freepik.com",
                        "2. Search for skin type terms",
                        "3. Filter by 'Free' photos",
                        "4. Download high-resolution images",
                        "5. Check licensing (attribution required)"
                    ],
                    "pros": "High quality, professional photos",
                    "cons": "May require attribution, limited medical accuracy"
                },
                
                "Unsplash": {
                    "url": "https://unsplash.com/",
                    "search_terms": [
                        "skincare",
                        "facial skin",
                        "beauty",
                        "dermatology",
                        "skin close up"
                    ],
                    "instructions": [
                        "1. Go to unsplash.com",
                        "2. Search for skin-related terms",
                        "3. Download high-resolution images",
                        "4. No attribution required"
                    ],
                    "pros": "Free, high quality, no attribution required",
                    "cons": "Limited medical/clinical images"
                },
                
                "Pexels": {
                    "url": "https://www.pexels.com/",
                    "search_terms": [
                        "skin texture",
                        "facial care",
                        "skincare routine",
                        "beauty face"
                    ],
                    "instructions": [
                        "1. Go to pexels.com",
                        "2. Search for skin images",
                        "3. Download original size",
                        "4. Free to use"
                    ],
                    "pros": "Completely free, good variety",
                    "cons": "May have fewer clinical images"
                }
            },
            
            "MEDICAL_DATABASES": {
                "DermNet": {
                    "url": "https://dermnetnz.org/",
                    "description": "Medical dermatology database",
                    "instructions": [
                        "Educational use images available",
                        "High medical accuracy",
                        "Contact for permission"
                    ],
                    "pros": "Medical accuracy, expert labeled",
                    "cons": "Requires permission, limited access"
                },
                
                "Open Access Journals": {
                    "description": "Medical research papers with skin images",
                    "sources": [
                        "PubMed Central",
                        "PLOS ONE",
                        "BMJ Open"
                    ],
                    "pros": "Scientific accuracy, peer-reviewed",
                    "cons": "Complex licensing, small dataset"
                }
            },
            
            "DATA_COLLECTION_METHODS": {
                "Personal_Photography": {
                    "setup": "Controlled photography sessions",
                    "equipment": [
                        "High-resolution camera or smartphone",
                        "Natural lighting or LED panels",
                        "Neutral background",
                        "Consistent distance (30-50cm)"
                    ],
                    "subjects": [
                        "Volunteers with different skin types",
                        "Multiple demographics and ages",
                        "Various skin conditions"
                    ],
                    "legal": [
                        "Obtain written consent",
                        "Privacy protection",
                        "Clear usage rights"
                    ]
                },
                
                "Crowdsourcing": {
                    "platforms": [
                        "Amazon Mechanical Turk",
                        "Appen",
                        "Custom mobile app"
                    ],
                    "quality_control": [
                        "Multiple reviewers per image",
                        "Clear submission guidelines",
                        "Expert validation"
                    ]
                }
            }
        }
        
        # Save collection guide
        with open(f"{self.base_dir}/metadata/collection_guide.json", 'w') as f:
            json.dump(collection_sources, f, indent=2)
        
        return collection_sources
    
    def create_manual_download_script(self):
        """Create a script to help with manual image organization"""
        
        script_content = '''#!/usr/bin/env python3
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
'''
        
        with open("organize_downloaded_images.py", 'w') as f:
            f.write(script_content)
        
        return script_content
    
    def create_download_instructions(self):
        """Create step-by-step download instructions"""
        
        instructions = """
# 📥 IMAGE DOWNLOAD INSTRUCTIONS

## 🎯 For Freepik (Dry Skin Images)

### Step 1: Access Freepik
1. Go to: https://www.freepik.com/free-photos-vectors/dry-skin
2. Create a free account if needed

### Step 2: Search and Filter
1. Use search terms:
   - "dry skin texture"
   - "dehydrated skin"
   - "flaky skin"
   - "rough skin texture"
   - "skin dermatology"

2. Apply filters:
   - ✅ Photos (not vectors)
   - ✅ Free content
   - ✅ High resolution
   - ✅ People/faces

### Step 3: Download Images
1. Click on promising images
2. Download the highest resolution available
3. Save to a "downloads" folder
4. Name files descriptively: "dry_skin_01.jpg", "dry_skin_texture_02.jpg"

### Step 4: Quality Check
- ✅ Resolution: Minimum 512x512 pixels
- ✅ Clear focus on skin
- ✅ Good lighting
- ✅ Visible skin texture details
- ❌ Avoid heavily filtered/edited images

## 🎯 Target Collection Goals

### Dry Skin (Priority)
- **Target:** 100+ images
- **Characteristics to look for:**
  - Visible flaking or scaling
  - Rough, uneven texture
  - Dull appearance
  - Fine lines more prominent
  - Tight-looking skin

### Other Skin Types
- **Oily Skin:** Shiny, large pores, acne-prone
- **Sensitive Skin:** Red, irritated, reactive
- **Normal Skin:** Balanced, smooth, healthy
- **Combination Skin:** Mixed characteristics

## 🔧 After Downloading

1. **Create downloads folder:**
   ```bash
   mkdir downloads
   # Place all downloaded images here
   ```

2. **Run organization script:**
   ```bash
   python organize_downloaded_images.py
   ```

3. **Manual review:**
   - Check auto-classification results
   - Move misclassified images
   - Remove poor quality images

4. **Train improved model:**
   ```bash
   python enhanced_training_pipeline.py
   ```

## 📊 Expected Results

With 100+ images per skin type:
- **Accuracy improvement:** 60% → 80%+
- **Confidence reliability:** Much better
- **Real-world performance:** Significantly improved

## ⚖️ Legal Considerations

- ✅ Check image licenses
- ✅ Provide attribution if required
- ✅ Use only for research/educational purposes
- ❌ Don't use copyrighted medical images without permission

## 💡 Pro Tips

1. **Quality over quantity** - 50 high-quality images > 200 poor ones
2. **Diverse lighting** - Various conditions improve robustness
3. **Different angles** - Front, 45-degree, close-ups
4. **Multiple demographics** - Various ages, skin tones
5. **Expert validation** - Have dermatologist review when possible
"""
        
        with open(f"{self.base_dir}/DOWNLOAD_INSTRUCTIONS.md", 'w') as f:
            f.write(instructions)
        
        return instructions
    
    def validate_dataset(self):
        """Validate the collected dataset"""
        
        print("\n📊 DATASET VALIDATION")
        print("="*30)
        
        stats = {}
        total_images = 0
        
        for split in ['train', 'validation', 'test']:
            split_path = os.path.join(self.base_dir, split)
            if not os.path.exists(split_path):
                continue
                
            stats[split] = {}
            
            for skin_type in self.skin_types:
                type_path = os.path.join(split_path, skin_type)
                if os.path.exists(type_path):
                    images = [f for f in os.listdir(type_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    stats[split][skin_type] = len(images)
                    total_images += len(images)
                else:
                    stats[split][skin_type] = 0
        
        # Print statistics
        for split, split_stats in stats.items():
            print(f"\n{split.upper()}:")
            for skin_type, count in split_stats.items():
                print(f"  {skin_type:12}: {count:3} images")
        
        print(f"\nTOTAL IMAGES: {total_images}")
        
        # Recommendations
        if total_images < 100:
            print("\n⚠️  RECOMMENDATION: Collect more images (target: 500+ per type)")
        elif total_images < 500:
            print("\n👍 GOOD: Decent dataset size, continue collecting")
        else:
            print("\n🎉 EXCELLENT: Large dataset, ready for professional training!")
        
        return stats

def main():
    """Main function to set up image collection"""
    
    print("📥 SKIN TYPE IMAGE COLLECTION HELPER")
    print("="*40)
    
    # Initialize helper
    helper = ImageCollectionHelper()
    
    # Create collection guide
    print("\n1. Creating collection guide...")
    helper.create_collection_guide()
    
    # Create manual download script
    print("2. Creating image organization script...")
    helper.create_manual_download_script()
    
    # Create download instructions
    print("3. Creating download instructions...")
    helper.create_download_instructions()
    
    # Validate existing dataset
    print("4. Validating current dataset...")
    helper.validate_dataset()
    
    print("\n✅ SETUP COMPLETE!")
    print("="*20)
    print("📂 Created training dataset structure")
    print("📋 Created download instructions: training_dataset/DOWNLOAD_INSTRUCTIONS.md")
    print("🐍 Created organization script: organize_downloaded_images.py")
    print("📊 Created collection guide: training_dataset/metadata/collection_guide.json")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Read DOWNLOAD_INSTRUCTIONS.md")
    print("2. Download images from Freepik and other sources")
    print("3. Place images in 'downloads' folder")
    print("4. Run: python organize_downloaded_images.py")
    print("5. Manually review and correct classifications")
    print("6. Run enhanced training when you have 100+ images per type")

if __name__ == "__main__":
    main()
