#!/usr/bin/env python3
"""
Freepik Collection Helper - Simplified approach for reliable collection
Provides direct links and semi-automated download assistance
"""

import webbrowser
import time
import os
import requests
from urllib.parse import quote

class FreepikCollectionHelper:
    """
    Helper for collecting images from Freepik with manual assistance
    """
    
    def __init__(self):
        self.base_urls = {
            'freepik': 'https://www.freepik.com/search',
            'unsplash': 'https://unsplash.com/s/photos',
            'pexels': 'https://www.pexels.com/search'
        }
        
        self.skin_type_searches = {
            'dry': [
                'dry skin texture',
                'flaky skin dermatology',
                'dehydrated skin close up',
                'rough skin surface',
                'skin dryness medical'
            ],
            'oily': [
                'oily skin texture',
                'shiny skin face',
                'acne prone skin',
                'greasy skin close up',
                'large pores skin'
            ],
            'sensitive': [
                'sensitive skin irritated',
                'red skin inflammation',
                'reactive skin dermatology',
                'sensitive facial skin',
                'skin irritation close up'
            ],
            'normal': [
                'normal skin texture',
                'healthy skin face',
                'balanced skin dermatology',
                'clear skin close up',
                'smooth skin texture'
            ],
            'combination': [
                'combination skin type',
                'mixed skin texture',
                't-zone oily skin',
                'combination facial skin',
                'partial oily skin'
            ]
        }
    
    def open_freepik_collection_session(self, skin_type='dry', max_tabs=5):
        """
        Open multiple Freepik search tabs for efficient manual collection
        """
        print(f"🎨 OPENING FREEPIK COLLECTION SESSION: {skin_type.upper()} SKIN")
        print("="*60)
        
        if skin_type not in self.skin_type_searches:
            print(f"❌ Unknown skin type: {skin_type}")
            return
        
        search_terms = self.skin_type_searches[skin_type][:max_tabs]
        
        for i, search_term in enumerate(search_terms):
            # Create Freepik search URL
            freepik_url = f"https://www.freepik.com/search?format=search&query={quote(search_term)}&type=photo"
            
            print(f"🔍 Opening tab {i+1}: {search_term}")
            print(f"   URL: {freepik_url}")
            
            try:
                webbrowser.open(freepik_url)
                time.sleep(1)  # Small delay between tab opens
            except Exception as e:
                print(f"❌ Error opening {search_term}: {e}")
        
        print(f"\n📋 MANUAL COLLECTION INSTRUCTIONS:")
        print("="*40)
        print("1. 🔍 In each browser tab, look for high-quality skin images")
        print("2. 📸 Right-click images and 'Save Image As...'")
        print("3. 💾 Save to a 'downloads' folder with descriptive names:")
        print(f"   Example: {skin_type}_skin_01.jpg, {skin_type}_texture_02.jpg")
        print("4. ✅ Aim for 20-50 images per skin type")
        print("5. 🔄 After downloading, run: python organize_downloaded_images.py")
        
        print(f"\n🎯 WHAT TO LOOK FOR ({skin_type.upper()} SKIN):")
        if skin_type == 'dry':
            print("   - Visible flaking or scaling")
            print("   - Rough, uneven texture")
            print("   - Dull appearance")
            print("   - Fine lines more prominent")
        elif skin_type == 'oily':
            print("   - Shiny, greasy appearance")
            print("   - Large, visible pores")
            print("   - Acne or blackheads")
            print("   - Thick skin texture")
        elif skin_type == 'sensitive':
            print("   - Redness or irritation")
            print("   - Reactive appearance")
            print("   - Thin skin appearance")
            print("   - Signs of inflammation")
        elif skin_type == 'normal':
            print("   - Balanced appearance")
            print("   - Even skin tone")
            print("   - Small pores")
            print("   - Smooth texture")
        elif skin_type == 'combination':
            print("   - Oily T-zone (forehead, nose, chin)")
            print("   - Normal/dry cheeks")
            print("   - Mixed pore sizes")
            print("   - Different textures in different areas")
    
    def create_download_folder_structure(self):
        """Create organized folder structure for downloads"""
        
        print("📁 Creating download folder structure...")
        
        # Create main downloads directory
        os.makedirs('downloads', exist_ok=True)
        
        # Create subdirectories for each skin type
        for skin_type in self.skin_type_searches.keys():
            skin_dir = os.path.join('downloads', skin_type)
            os.makedirs(skin_dir, exist_ok=True)
        
        print("✅ Created folder structure:")
        print("   📂 downloads/")
        for skin_type in self.skin_type_searches.keys():
            print(f"     📂 {skin_type}/")
        
        print("\n💡 TIP: Save images directly to the appropriate subfolder")
        print("   Example: downloads/dry/dry_skin_texture_01.jpg")
    
    def run_comprehensive_collection(self):
        """
        Run comprehensive collection for all skin types
        """
        print("🚀 COMPREHENSIVE FREEPIK COLLECTION")
        print("="*40)
        
        # Create folder structure
        self.create_download_folder_structure()
        
        print(f"\n🎯 COLLECTION PLAN:")
        print("We'll open browser tabs for each skin type systematically")
        print("This gives you the best manual collection experience")
        
        for skin_type in self.skin_type_searches.keys():
            input(f"\nPress Enter to open {skin_type.upper()} skin collection tabs...")
            self.open_freepik_collection_session(skin_type, max_tabs=3)
            
            print(f"\n⏱️  Take your time to download {skin_type} images...")
            print("⏯️  When ready for the next skin type, continue...")
        
        print(f"\n🎉 ALL SKIN TYPE TABS OPENED!")
        print("="*30)
        print("📋 FINAL CHECKLIST:")
        print("□ Downloaded dry skin images")
        print("□ Downloaded oily skin images") 
        print("□ Downloaded sensitive skin images")
        print("□ Downloaded normal skin images")
        print("□ Downloaded combination skin images")
        print("□ Organized images in downloads/ folders")
        print("□ Run: python organize_downloaded_images.py")
        print("□ Train improved model")
    
    def quick_dry_skin_collection(self):
        """Quick collection focused on dry skin only"""
        
        print("🏥 QUICK DRY SKIN COLLECTION")
        print("="*30)
        
        self.create_download_folder_structure()
        self.open_freepik_collection_session('dry', max_tabs=5)
        
        print(f"\n🎯 COLLECTION TARGET:")
        print("Aim to download 30-50 high-quality dry skin images")
        print("Focus on images showing flaky, rough, or dehydrated skin")
        
        return True
    
    def create_download_guide(self):
        """Create a downloadable guide for manual collection"""
        
        guide_content = """
# 📸 FREEPIK MANUAL COLLECTION GUIDE

## 🎯 Quick Start for Dry Skin

### Direct Links (Open these in browser)
1. **Dry Skin Texture:** https://www.freepik.com/search?format=search&query=dry%20skin%20texture&type=photo
2. **Flaky Skin:** https://www.freepik.com/search?format=search&query=flaky%20skin&type=photo
3. **Dehydrated Skin:** https://www.freepik.com/search?format=search&query=dehydrated%20skin&type=photo
4. **Rough Skin:** https://www.freepik.com/search?format=search&query=rough%20skin&type=photo

### Collection Process
1. **Filter Settings:**
   - ✅ Photos (not vectors)
   - ✅ Free content
   - ✅ High resolution

2. **Download Process:**
   - Right-click on high-quality images
   - Select "Save Image As..."
   - Save to `downloads/dry/` folder
   - Use descriptive names: `dry_skin_01.jpg`

3. **Quality Criteria:**
   - ✅ Resolution: 512x512+ pixels
   - ✅ Clear skin texture visible
   - ✅ Good lighting
   - ✅ Realistic appearance
   - ❌ Avoid heavily filtered images

### What to Look For (Dry Skin)
- Visible flaking or scaling
- Rough, uneven texture
- Dull appearance
- Fine lines more prominent
- Tight-looking skin

## 📊 Collection Targets

| Skin Type | Target Images | Priority |
|-----------|---------------|----------|
| Dry | 50+ | 🔥 HIGH |
| Oily | 30+ | Medium |
| Sensitive | 30+ | Medium |
| Normal | 30+ | Medium |
| Combination | 20+ | Low |

## 🔄 After Collection

```bash
# Organize downloaded images
python organize_downloaded_images.py

# Train improved model
python enhanced_training_pipeline.py
```

## 📈 Expected Results

- **Current accuracy:** ~60%
- **After 50+ dry skin images:** ~80%
- **After all skin types:** ~90%+

---
**Generated by Freepik Collection Helper**
"""
        
        with open('FREEPIK_COLLECTION_GUIDE.md', 'w') as f:
            f.write(guide_content)
        
        print("✅ Created: FREEPIK_COLLECTION_GUIDE.md")
        return True

def main():
    """Main function"""
    
    print("🎨 FREEPIK COLLECTION HELPER")
    print("="*30)
    
    helper = FreepikCollectionHelper()
    
    print("📋 COLLECTION OPTIONS:")
    print("1. 🏥 Quick dry skin collection (recommended)")
    print("2. 🔄 Comprehensive all skin types")
    print("3. 📖 Create collection guide only")
    print("4. 🎯 Custom skin type")
    
    choice = input("\nEnter choice (1-4) or press Enter for option 1: ").strip()
    
    if choice == "2":
        helper.run_comprehensive_collection()
    elif choice == "3":
        helper.create_download_guide()
        print("📖 Guide created! Read FREEPIK_COLLECTION_GUIDE.md")
    elif choice == "4":
        skin_type = input("Enter skin type (dry/oily/sensitive/normal/combination): ").lower().strip()
        if skin_type in helper.skin_type_searches:
            helper.open_freepik_collection_session(skin_type)
        else:
            print("❌ Invalid skin type")
    else:
        # Default: Quick dry skin collection
        helper.quick_dry_skin_collection()
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Download images from opened browser tabs")
    print("2. Save to downloads/ folders with descriptive names")
    print("3. Run: python organize_downloaded_images.py")
    print("4. Train improved model when you have 50+ images per type")

if __name__ == "__main__":
    main()
