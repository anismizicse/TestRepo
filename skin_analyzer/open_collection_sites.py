#!/usr/bin/env python3
"""
Quick Link Opener - Opens image collection sites directly in your default browser
"""

import webbrowser
import time

def open_collection_sites():
    """Open key image collection sites in browser"""
    
    sites = {
        "🎨 Freepik - Dry Skin": "https://www.freepik.com/free-photos-vectors/dry-skin",
        "📸 Unsplash - Dry Skin": "https://unsplash.com/s/photos/dry-skin", 
        "🌟 Pexels - Skin Texture": "https://www.pexels.com/search/skin%20texture/",
        "🎭 Pixabay - Skin Care": "https://pixabay.com/images/search/skin%20care/"
    }
    
    print("🖼️ OPENING IMAGE COLLECTION SITES")
    print("="*40)
    
    for name, url in sites.items():
        print(f"Opening: {name}")
        print(f"URL: {url}")
        
        try:
            webbrowser.open(url)
            print("✅ Opened successfully")
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Manual link: {url}")
        
        print("-" * 40)
        time.sleep(1)  # Small delay between opens
    
    print("\n🎯 COLLECTION INSTRUCTIONS:")
    print("1. Look for high-quality skin images")
    print("2. Download to 'downloads' folder")
    print("3. Use descriptive filenames")
    print("4. Run: python organize_downloaded_images.py")

if __name__ == "__main__":
    open_collection_sites()
