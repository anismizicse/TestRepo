#!/usr/bin/env python3
"""
Enhanced Freepik Image Downloader with API Access
Downloads skin type images automatically from Freepik using API
"""

import os
import requests
import json
import time
import random
from urllib.parse import urlparse
from PIL import Image
import hashlib
from tqdm import tqdm
import threading

class FreepikImageDownloader:
    """
    Automated image downloader for Freepik using their API
    """
    
    def __init__(self, api_key, base_dir="training_dataset", downloads_dir="downloads"):
        self.api_key = api_key
        self.base_dir = base_dir
        self.downloads_dir = downloads_dir
        
        # Freepik API configuration
        self.base_url = "https://api.freepik.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'X-Freepik-API-Key': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'SkinTypeAnalyzer/1.0'
        })
        
        # Rate limiting
        self.request_delay = 1.0  # Seconds between requests
        self.last_request_time = 0
        self.lock = threading.Lock()
        
        # Setup directories
        self.setup_directories()
        
        print("🎨 Freepik Image Downloader Initialized")
        print(f"🔑 API Key: {self.api_key[:10]}...")
        print(f"📁 Downloads: {self.downloads_dir}")
        print(f"📁 Training Data: {self.base_dir}")
    
    def setup_directories(self):
        """Create necessary directory structure"""
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        for skin_type in skin_types:
            for split in ['train', 'validation', 'test']:
                os.makedirs(os.path.join(self.base_dir, split, skin_type), exist_ok=True)
    
    def rate_limit(self):
        """Implement rate limiting to respect API limits"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.request_delay:
                sleep_time = self.request_delay - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def search_freepik_images(self, query, limit=50, image_type="photo"):
        """
        Search for images on Freepik using their API
        """
        print(f"🔍 Searching Freepik: {query}")
        
        url = f"{self.base_url}/search"
        
        params = {
            'locale': 'en-US',
            'term': query,
            'limit': min(limit, 50),  # Max 50 per request for free tier
            'order': 'latest',
            'filters': {
                'license': 'free',
                'content_type': image_type,
                'orientation': 'portrait'
            }
        }
        
        try:
            self.rate_limit()
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self.process_freepik_results(data, query)
            elif response.status_code == 401:
                print("❌ API Authentication failed - check your API key")
                return []
            elif response.status_code == 429:
                print("⚠️ Rate limit exceeded - waiting...")
                time.sleep(10)
                return self.search_freepik_images(query, limit, image_type)
            else:
                print(f"❌ Freepik API error: {response.status_code}")
                print(f"Response: {response.text}")
                return []
                
        except Exception as e:
            print(f"❌ Freepik search error: {e}")
            return []
    
    def process_freepik_results(self, data, query):
        """Process Freepik API results and extract download info"""
        results = []
        
        for item in data.get('data', []):
            try:
                # Extract image information
                image_info = {
                    'id': item.get('id'),
                    'title': item.get('title', query),
                    'description': item.get('description', ''),
                    'preview_url': item.get('image', {}).get('source', {}).get('url'),
                    'download_url': None,  # Will be fetched separately
                    'tags': item.get('tags', []),
                    'source': 'freepik',
                    'query': query,
                    'license': item.get('license', 'free'),
                    'content_type': item.get('content_type')
                }
                
                # Only process photos with people (skin images)
                if (image_info['content_type'] == 'photo' and 
                    image_info['preview_url'] and
                    any(tag.lower() in ['skin', 'face', 'beauty', 'skincare', 'dermatology'] 
                        for tag in image_info['tags'])):
                    results.append(image_info)
                    
            except Exception as e:
                print(f"❌ Error processing Freepik result: {e}")
        
        print(f"✅ Found {len(results)} relevant skin images")
        return results
    
    def get_download_url(self, resource_id):
        """
        Get the download URL for a specific Freepik resource
        """
        url = f"{self.base_url}/resources/{resource_id}/download"
        
        try:
            self.rate_limit()
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('url')
            else:
                print(f"❌ Failed to get download URL for {resource_id}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting download URL: {e}")
            return None
    
    def download_single_image(self, image_info):
        """Download a single image from Freepik"""
        
        try:
            # Get download URL first
            if image_info['id']:
                download_url = self.get_download_url(image_info['id'])
                if not download_url:
                    # Fallback to preview URL
                    download_url = image_info['preview_url']
            else:
                download_url = image_info['preview_url']
            
            if not download_url:
                return None
            
            # Download image
            self.rate_limit()
            response = self.session.get(download_url, timeout=30)
            
            if response.status_code == 200:
                # Generate filename
                filename = self.generate_filename(image_info)
                filepath = os.path.join(self.downloads_dir, filename)
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Validate image
                if self.validate_image(filepath):
                    return filepath
                else:
                    os.remove(filepath)
                    return None
            else:
                print(f"❌ Download failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    def generate_filename(self, image_info):
        """Generate unique filename for downloaded image"""
        
        # Clean title/query for filename
        title = image_info.get('title', image_info.get('query', 'skin')).replace(' ', '_').lower()
        title = ''.join(c for c in title if c.isalnum() or c in ('_', '-'))[:30]
        
        # Add metadata
        source = image_info.get('source', 'freepik')
        image_id = image_info.get('id', str(int(time.time())))
        
        # Create hash for uniqueness
        content = f"{title}_{source}_{image_id}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return f"{title}_{source}_{hash_suffix}.jpg"
    
    def validate_image(self, filepath):
        """Validate downloaded image quality"""
        
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                
                # Check minimum resolution
                if width < 300 or height < 300:
                    return False
                
                # Check file size (not too small)
                file_size = os.path.getsize(filepath)
                if file_size < 15000:  # 15KB minimum
                    return False
                
                return True
                
        except Exception:
            return False
    
    def classify_and_organize(self, filepath, target_skin_type=None):
        """Classify and move image to appropriate skin type folder"""
        
        filename = os.path.basename(filepath)
        
        # Auto-classify based on filename if no target specified
        if not target_skin_type:
            target_skin_type = self.auto_classify_image(filename)
        
        if target_skin_type:
            target_dir = os.path.join(self.base_dir, 'train', target_skin_type)
            target_path = os.path.join(target_dir, filename)
            
            # Ensure unique filename
            counter = 1
            while os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{counter}{ext}"
                target_path = os.path.join(target_dir, new_filename)
                counter += 1
            
            # Move file
            os.rename(filepath, target_path)
            return target_path
        
        return filepath
    
    def auto_classify_image(self, filename):
        """Auto-classify image based on filename keywords"""
        
        filename_lower = filename.lower()
        
        keywords = {
            'dry': ['dry', 'flaky', 'rough', 'dehydrated', 'scaling', 'cracked'],
            'oily': ['oily', 'shiny', 'greasy', 'acne', 'sebum', 'pores'],
            'sensitive': ['sensitive', 'red', 'irritated', 'reactive', 'inflamed', 'rash'],
            'combination': ['combination', 'mixed', 'tzone', 'combo'],
            'normal': ['normal', 'balanced', 'healthy', 'clear', 'smooth']
        }
        
        for skin_type, terms in keywords.items():
            for term in terms:
                if term in filename_lower:
                    return skin_type
        
        return 'normal'  # Default classification
    
    def download_skin_type_images(self, skin_type, count=50):
        """Download images for a specific skin type from Freepik"""
        
        print(f"\n🎯 COLLECTING {skin_type.upper()} SKIN IMAGES FROM FREEPIK")
        print("="*60)
        
        # Define search queries for each skin type
        queries = {
            'dry': [
                'dry skin texture close up',
                'dehydrated skin face',
                'flaky skin dermatology',
                'rough skin surface',
                'cracked skin dry'
            ],
            'oily': [
                'oily skin face shine',
                'greasy skin texture',
                'acne prone skin',
                'sebaceous skin pores',
                'shiny skin surface'
            ],
            'sensitive': [
                'sensitive skin redness',
                'irritated skin face',
                'reactive skin inflammation',
                'red skin dermatology',
                'sensitive facial skin'
            ],
            'normal': [
                'normal healthy skin',
                'balanced skin texture',
                'clear skin face',
                'smooth skin surface',
                'healthy facial skin'
            ],
            'combination': [
                'combination skin face',
                'mixed skin type',
                'oily t-zone dry cheeks',
                'combination facial skin',
                'mixed skin texture'
            ]
        }
        
        skin_queries = queries.get(skin_type, [f'{skin_type} skin'])
        all_results = []
        
        # Search for images
        for query in skin_queries:
            print(f"🔍 Searching: {query}")
            results = self.search_freepik_images(query, limit=count//len(skin_queries))
            all_results.extend(results)
            
            # Rate limiting between queries
            time.sleep(2)
        
        print(f"📊 Found {len(all_results)} potential images")
        
        # Download images with progress bar
        downloaded = []
        print(f"⬇️ Downloading {skin_type} skin images...")
        
        for image_info in tqdm(all_results, desc=f"Downloading {skin_type}"):
            filepath = self.download_single_image(image_info)
            if filepath:
                # Organize into skin type folder
                organized_path = self.classify_and_organize(filepath, skin_type)
                downloaded.append(organized_path)
                
                # Limit total downloads
                if len(downloaded) >= count:
                    break
            
            # Small delay between downloads
            time.sleep(0.5)
        
        print(f"✅ Successfully collected {len(downloaded)} {skin_type} skin images")
        return downloaded
    
    def download_all_skin_types(self, images_per_type=30):
        """Download images for all skin types from Freepik"""
        
        print("🎨 FREEPIK AUTOMATED SKIN TYPE IMAGE COLLECTION")
        print("="*55)
        
        skin_types = ['dry', 'oily', 'normal', 'sensitive', 'combination']
        total_downloaded = {}
        
        for skin_type in skin_types:
            try:
                downloaded = self.download_skin_type_images(skin_type, images_per_type)
                total_downloaded[skin_type] = len(downloaded)
                
                # Progress update
                print(f"📊 Progress: {skin_type} = {len(downloaded)} images")
                
                # Longer delay between skin types
                time.sleep(3)
                
            except Exception as e:
                print(f"❌ Error collecting {skin_type} images: {e}")
                total_downloaded[skin_type] = 0
        
        # Summary
        print(f"\n🎉 FREEPIK COLLECTION COMPLETE!")
        print("="*35)
        
        total_images = sum(total_downloaded.values())
        for skin_type, count in total_downloaded.items():
            print(f"{skin_type:12}: {count:3} images")
        
        print(f"{'TOTAL':12}: {total_images:3} images")
        
        if total_images >= 100:
            print("\n🚀 Ready for enhanced training!")
            print("Run: python enhanced_training_pipeline.py")
        else:
            print(f"\n📈 Collect {100 - total_images} more images for best results")
        
        return total_downloaded

def main():
    """Main function for Freepik collection"""
    
    # Your Freepik API key
    FREEPIK_API_KEY = "FPSX70e60abcb422221aedd4c3497246cf77"
    
    print("🎨 FREEPIK SKIN TYPE IMAGE COLLECTOR")
    print("="*40)
    
    # Initialize downloader
    downloader = FreepikImageDownloader(FREEPIK_API_KEY)
    
    # Start collection
    print(f"🔑 Using API Key: {FREEPIK_API_KEY[:10]}...")
    print("🚀 Starting Freepik image collection...")
    
    # Download images for all skin types
    total_downloaded = downloader.download_all_skin_types(images_per_type=40)
    
    # Final summary
    total_images = sum(total_downloaded.values())
    print(f"\n🎉 COLLECTION SUMMARY:")
    print(f"Total images from Freepik: {total_images}")
    
    if total_images >= 150:
        print("🏆 Excellent! Ready for professional-level training")
        print("🔥 Combined with existing data, you now have a powerful dataset!")
    elif total_images >= 50:
        print("👍 Good collection! Ready for enhanced training")
    else:
        print("📈 Consider running again or trying different search terms")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Review collected images in training_dataset/")
    print("2. Run: python enhanced_training_pipeline.py")
    print("3. Test improved model accuracy!")

if __name__ == "__main__":
    main()
