#!/usr/bin/env python3
"""
Web Scraping Helper for Image Collection
Helps collect images from various sources while respecting terms of service
"""

import requests
import json
import time
import os
from urllib.parse import quote
import base64

class ImageSearchHelper:
    """
    Helper for finding and collecting skin type images from various APIs
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def search_unsplash(self, query, per_page=30):
        """
        Search Unsplash for images (requires API key)
        Visit: https://unsplash.com/developers
        """
        
        # Note: You need to register for a free API key at unsplash.com/developers
        api_key = "YOUR_UNSPLASH_ACCESS_KEY"  # Replace with your key
        
        if api_key == "YOUR_UNSPLASH_ACCESS_KEY":
            print("❌ Please register for a free Unsplash API key at:")
            print("   https://unsplash.com/developers")
            return []
        
        url = "https://api.unsplash.com/search/photos"
        params = {
            'query': query,
            'per_page': per_page,
            'orientation': 'portrait'
        }
        headers = {'Authorization': f'Client-ID {api_key}'}
        
        try:
            response = self.session.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return self.extract_unsplash_urls(data)
            else:
                print(f"❌ Unsplash API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error searching Unsplash: {e}")
            return []
    
    def extract_unsplash_urls(self, data):
        """Extract download URLs from Unsplash API response"""
        urls = []
        
        for result in data.get('results', []):
            urls.append({
                'url': result['urls']['regular'],
                'download_url': result['links']['download'],
                'description': result.get('description', 'Skin image'),
                'photographer': result['user']['name'],
                'source': 'unsplash'
            })
        
        return urls
    
    def download_image(self, image_info, download_dir="downloads"):
        """Download a single image"""
        
        os.makedirs(download_dir, exist_ok=True)
        
        try:
            # Get image
            response = self.session.get(image_info['url'], timeout=30)
            if response.status_code == 200:
                # Generate filename
                filename = self.generate_filename(image_info)
                filepath = os.path.join(download_dir, filename)
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Downloaded: {filename}")
                return filepath
            else:
                print(f"❌ Failed to download: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    def generate_filename(self, image_info):
        """Generate appropriate filename for downloaded image"""
        
        # Clean description for filename
        desc = image_info.get('description', 'skin_image')
        desc = ''.join(c for c in desc if c.isalnum() or c in (' ', '-', '_')).strip()
        desc = desc.replace(' ', '_').lower()[:50]  # Limit length
        
        # Add source and extension
        source = image_info.get('source', 'unknown')
        timestamp = int(time.time())
        
        return f"{desc}_{source}_{timestamp}.jpg"
    
    def get_free_stock_sites(self):
        """Return list of free stock photo sites for manual collection"""
        
        sites = {
            "Unsplash": {
                "url": "https://unsplash.com/",
                "search_urls": {
                    "dry_skin": "https://unsplash.com/s/photos/dry-skin",
                    "oily_skin": "https://unsplash.com/s/photos/oily-skin",
                    "skincare": "https://unsplash.com/s/photos/skincare",
                    "facial_skin": "https://unsplash.com/s/photos/facial-skin"
                },
                "license": "Free to use",
                "attribution": "Not required but appreciated"
            },
            
            "Pexels": {
                "url": "https://www.pexels.com/",
                "search_urls": {
                    "skin_texture": "https://www.pexels.com/search/skin%20texture/",
                    "skincare": "https://www.pexels.com/search/skincare/",
                    "facial_care": "https://www.pexels.com/search/facial%20care/"
                },
                "license": "Free to use",
                "attribution": "Not required"
            },
            
            "Freepik": {
                "url": "https://www.freepik.com/",
                "search_urls": {
                    "dry_skin": "https://www.freepik.com/free-photos-vectors/dry-skin",
                    "oily_skin": "https://www.freepik.com/free-photos-vectors/oily-skin",
                    "sensitive_skin": "https://www.freepik.com/free-photos-vectors/sensitive-skin"
                },
                "license": "Free with attribution",
                "attribution": "Required for free accounts"
            },
            
            "Pixabay": {
                "url": "https://pixabay.com/",
                "search_urls": {
                    "skin_care": "https://pixabay.com/images/search/skin%20care/",
                    "dermatology": "https://pixabay.com/images/search/dermatology/"
                },
                "license": "Free to use",
                "attribution": "Not required"
            }
        }
        
        return sites
    
    def create_collection_bookmarks(self):
        """Create HTML bookmark file for easy access to image sources"""
        
        sites = self.get_free_stock_sites()
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Skin Type Image Collection Bookmarks</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
        .site { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .site h2 { color: #2196F3; margin-top: 0; }
        .search-links { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .search-link { padding: 10px; background: #f5f5f5; border-radius: 4px; text-align: center; }
        .search-link a { text-decoration: none; color: #333; font-weight: bold; }
        .search-link a:hover { color: #2196F3; }
        .license { background: #e8f5e8; padding: 10px; border-radius: 4px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🖼️ Skin Type Image Collection Bookmarks</h1>
    <p>Quick access to free stock photo sites for collecting training images.</p>
"""
        
        for site_name, site_info in sites.items():
            html_content += f"""
    <div class="site">
        <h2>{site_name}</h2>
        <p><strong>Website:</strong> <a href="{site_info['url']}" target="_blank">{site_info['url']}</a></p>
        
        <h3>Quick Search Links:</h3>
        <div class="search-links">
"""
            
            for search_name, search_url in site_info['search_urls'].items():
                display_name = search_name.replace('_', ' ').title()
                html_content += f"""
            <div class="search-link">
                <a href="{search_url}" target="_blank">{display_name}</a>
            </div>
"""
            
            html_content += f"""
        </div>
        
        <div class="license">
            <strong>License:</strong> {site_info['license']}<br>
            <strong>Attribution:</strong> {site_info['attribution']}
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open("image_collection_bookmarks.html", 'w') as f:
            f.write(html_content)
        
        return html_content

def create_manual_collection_guide():
    """Create step-by-step manual collection guide"""
    
    guide = """
# 📸 MANUAL IMAGE COLLECTION GUIDE

## 🎯 Step-by-Step Process for Freepik Dry Skin Images

### 1. Access Freepik
- Go to: https://www.freepik.com/free-photos-vectors/dry-skin
- Create free account if needed

### 2. Search Strategy
Use these search terms:
- "dry skin texture"
- "dehydrated skin close up"
- "flaky skin dermatology"
- "rough skin surface"
- "skin dryness medical"

### 3. Filtering Images
✅ **Select these filters:**
- Photos (not illustrations)
- Free content only
- High resolution
- People category

❌ **Avoid these:**
- Heavily edited/filtered images
- Stock model photos with makeup
- Low resolution images
- Cartoon/illustration style

### 4. Download Process
1. Click on each relevant image
2. Click "Free Download"
3. Choose highest resolution available
4. Save to your "downloads" folder
5. Use descriptive names: "dry_skin_texture_01.jpg"

### 5. Quality Checklist
For each image, verify:
- ✅ Resolution: 512x512 pixels minimum
- ✅ Clear skin texture visible
- ✅ Good lighting (not too dark/bright)
- ✅ Focus on skin area
- ✅ Realistic skin appearance

### 6. Quantity Targets
- **Immediate goal:** 50 dry skin images
- **Medium-term goal:** 100+ per skin type
- **Professional goal:** 500+ per skin type

## 🔄 After Downloading

### Organize Images
```bash
# Create downloads folder
mkdir downloads

# Download images to this folder
# Then run organization script:
python organize_downloaded_images.py
```

### Train Improved Model
```bash
# After collecting 50+ images per type:
python enhanced_training_pipeline.py
```

## 📊 Expected Improvements

| Images per Type | Expected Accuracy | Model Quality |
|----------------|-------------------|---------------|
| 10-20 | 65-70% | Basic |
| 50-100 | 75-85% | Good |
| 200-500 | 85-95% | Professional |
| 500+ | 90-95%+ | Medical-grade |

## 💡 Pro Tips

1. **Diverse lighting:** Collect images in different lighting conditions
2. **Multiple angles:** Front-facing, 45-degree, close-ups
3. **Various demographics:** Different ages, skin tones, ethnicities
4. **Real vs. stock:** Prefer realistic photos over styled stock images
5. **Medical accuracy:** Look for medically accurate representations

## ⚖️ Legal Compliance

- ✅ Only download free/licensed images
- ✅ Provide attribution when required
- ✅ Use for educational/research purposes only
- ❌ Don't use copyrighted medical images
- ❌ Respect website terms of service
"""
    
    with open("MANUAL_COLLECTION_GUIDE.md", 'w') as f:
        f.write(guide)
    
    return guide

def main():
    """Main function for image collection assistance"""
    
    print("🖼️ SKIN TYPE IMAGE COLLECTION ASSISTANT")
    print("="*45)
    
    # Initialize helper
    helper = ImageSearchHelper()
    
    # Create bookmark file
    print("1. Creating collection bookmarks...")
    helper.create_collection_bookmarks()
    
    # Create manual guide
    print("2. Creating manual collection guide...")
    create_manual_collection_guide()
    
    # Display quick access info
    sites = helper.get_free_stock_sites()
    
    print("\n📚 COLLECTION RESOURCES CREATED:")
    print("✅ image_collection_bookmarks.html - Quick access to all sites")
    print("✅ MANUAL_COLLECTION_GUIDE.md - Step-by-step instructions")
    
    print("\n🎯 QUICK ACCESS LINKS FOR DRY SKIN:")
    print("=" * 40)
    
    for site_name, site_info in sites.items():
        if 'dry_skin' in site_info['search_urls']:
            print(f"{site_name}: {site_info['search_urls']['dry_skin']}")
    
    print("\n📋 COLLECTION CHECKLIST:")
    print("□ 1. Open image_collection_bookmarks.html in browser")
    print("□ 2. Visit Freepik dry skin link")
    print("□ 3. Download 20-50 dry skin images")
    print("□ 4. Save to 'downloads' folder")
    print("□ 5. Run: python organize_downloaded_images.py")
    print("□ 6. Manually review classifications")
    print("□ 7. Repeat for other skin types")
    print("□ 8. Train improved model")
    
    print(f"\n🚀 START HERE:")
    print("Open: image_collection_bookmarks.html")
    print("Read: MANUAL_COLLECTION_GUIDE.md")

if __name__ == "__main__":
    main()
