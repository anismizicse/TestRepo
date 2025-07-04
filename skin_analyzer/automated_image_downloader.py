#!/usr/bin/env python3
"""
Automated Image Downloader for Skin Type Classification
Downloads images from various free APIs and organizes them automatically
Enhanced with Freepik scraping capabilities
"""

import os
import requests
import json
import time
import random
from urllib.parse import urlparse, quote
from PIL import Image
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# New imports for Freepik scraping
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

class AutomatedImageDownloader:
    """
    Automated image downloader that respects rate limits and terms of service
    Enhanced with Freepik scraping
    """
    
    def __init__(self, base_dir="training_dataset", downloads_dir="downloads"):
        self.base_dir = base_dir
        self.downloads_dir = downloads_dir
        self.session = requests.Session()
        
        # Use rotating user agents
        self.ua = UserAgent()
        self.session.headers.update({
            'User-Agent': self.ua.random
        })
        
        # Rate limiting
        self.request_delay = 2.0  # Increased delay for Freepik
        self.last_request_time = 0
        self.lock = threading.Lock()
        
        # Setup directories
        self.setup_directories()
        
        # Download statistics
        self.stats = {
            'total_downloaded': 0,
            'freepik_downloaded': 0,
            'unsplash_downloaded': 0,
            'pexels_downloaded': 0,
            'failed_downloads': 0,
            'by_skin_type': {
                'dry': 0, 'oily': 0, 'normal': 0, 
                'sensitive': 0, 'combination': 0
            }
        }
        
        # API keys (users need to register for free keys)
        self.api_keys = {
            'unsplash': os.getenv('UNSPLASH_ACCESS_KEY'),
            'pexels': os.getenv('PEXELS_API_KEY'),
        }
        
        print("🤖 Enhanced Automated Image Downloader Initialized")
        print(f"📁 Downloads: {self.downloads_dir}")
        print(f"📁 Training Data: {self.base_dir}")
        if SELENIUM_AVAILABLE:
            print("✅ Selenium available - Freepik scraping enabled")
        else:
            print("⚠️  Selenium not available - install with 'pip install selenium'")

    def setup_directories(self):
        """Create necessary directory structure"""
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        for skin_type in skin_types:
            os.makedirs(os.path.join(self.base_dir, 'train', skin_type), exist_ok=True)
    
    def rate_limit(self):
        """Implement rate limiting to respect API limits"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.request_delay:
                sleep_time = self.request_delay - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def download_from_unsplash(self, query, count=30):
        """
        Download images from Unsplash API
        Register for free at: https://unsplash.com/developers
        """
        
        if not self.api_keys['unsplash']:
            print("❌ Unsplash API key not configured")
            print("📝 Register for free at: https://unsplash.com/developers")
            print("💡 Add your key to the script or config file")
            return []
        
        print(f"📸 Downloading from Unsplash: {query}")
        
        url = "https://api.unsplash.com/search/photos"
        headers = {'Authorization': f"Client-ID {self.api_keys['unsplash']}"}
        
        params = {
            'query': query,
            'per_page': min(count, 30),  # Max 30 per request
            'orientation': 'portrait',
            'content_filter': 'high'
        }
        
        try:
            self.rate_limit()
            response = self.session.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self.process_unsplash_results(data, query)
            else:
                print(f"❌ Unsplash API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Unsplash error: {e}")
            return []
    
    def process_unsplash_results(self, data, query):
        """Process Unsplash API results and download images"""
        downloaded = []
        
        for result in data.get('results', []):
            try:
                image_info = {
                    'url': result['urls']['regular'],
                    'download_url': result['links']['download'],
                    'description': result.get('description', query),
                    'photographer': result['user']['name'],
                    'source': 'unsplash',
                    'query': query,
                    'id': result['id']
                }
                
                filepath = self.download_single_image(image_info)
                if filepath:
                    downloaded.append(filepath)
                    
            except Exception as e:
                print(f"❌ Error processing Unsplash result: {e}")
        
        return downloaded
    
    def download_from_pexels(self, query, count=30):
        """
        Download images from Pexels API
        Register for free at: https://www.pexels.com/api/
        """
        
        if not self.api_keys['pexels']:
            print("❌ Pexels API key not configured")
            print("📝 Register for free at: https://www.pexels.com/api/")
            return []
        
        print(f"🌟 Downloading from Pexels: {query}")
        
        url = "https://api.pexels.com/v1/search"
        headers = {'Authorization': self.api_keys['pexels']}
        
        params = {
            'query': query,
            'per_page': min(count, 80),  # Max 80 per request
            'orientation': 'portrait'
        }
        
        try:
            self.rate_limit()
            response = self.session.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self.process_pexels_results(data, query)
            else:
                print(f"❌ Pexels API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Pexels error: {e}")
            return []
    
    def process_pexels_results(self, data, query):
        """Process Pexels API results and download images"""
        downloaded = []
        
        for photo in data.get('photos', []):
            try:
                image_info = {
                    'url': photo['src']['large'],
                    'description': photo.get('alt', query),
                    'photographer': photo['photographer'],
                    'source': 'pexels',
                    'query': query,
                    'id': photo['id']
                }
                
                filepath = self.download_single_image(image_info)
                if filepath:
                    downloaded.append(filepath)
                    
            except Exception as e:
                print(f"❌ Error processing Pexels result: {e}")
        
        return downloaded
    
    def download_single_image(self, image_info):
        """Download a single image with error handling"""
        
        try:
            self.rate_limit()
            
            # Download image
            response = self.session.get(image_info['url'], timeout=30)
            if response.status_code != 200:
                return None
            
            # Generate unique filename
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
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    def generate_filename(self, image_info):
        """Generate unique filename for downloaded image"""
        
        # Clean query for filename
        query = image_info.get('query', 'skin').replace(' ', '_').lower()
        source = image_info.get('source', 'unknown')
        image_id = image_info.get('id', str(int(time.time())))
        
        # Create hash for uniqueness
        content = f"{query}_{source}_{image_id}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return f"{query}_{source}_{hash_suffix}.jpg"
    
    def validate_image(self, filepath):
        """Validate downloaded image quality"""
        
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                
                # Check minimum resolution
                if width < 224 or height < 224:
                    return False
                
                # Check file size (not too small)
                file_size = os.path.getsize(filepath)
                if file_size < 10000:  # 10KB minimum
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
            'dry': ['dry', 'flaky', 'rough', 'dehydrated', 'scaling'],
            'oily': ['oily', 'shiny', 'greasy', 'acne', 'sebum'],
            'sensitive': ['sensitive', 'red', 'irritated', 'reactive', 'inflamed'],
            'combination': ['combination', 'mixed', 'tzone', 'combo'],
            'normal': ['normal', 'balanced', 'healthy', 'clear']
        }
        
        for skin_type, terms in keywords.items():
            for term in terms:
                if term in filename_lower:
                    return skin_type
        
        return 'normal'  # Default classification
    
    def download_skin_type_images(self, skin_type, count_per_source=20):
        """Download images for a specific skin type from multiple sources"""
        
        print(f"\n🎯 COLLECTING {skin_type.upper()} SKIN IMAGES")
        print("="*50)
        
        # Define search queries for each skin type
        queries = {
            'dry': ['dry skin', 'dehydrated skin', 'flaky skin', 'rough skin texture'],
            'oily': ['oily skin', 'shiny skin', 'greasy skin', 'acne prone skin'],
            'sensitive': ['sensitive skin', 'irritated skin', 'red skin', 'reactive skin'],
            'normal': ['normal skin', 'healthy skin', 'balanced skin', 'clear skin'],
            'combination': ['combination skin', 'mixed skin type', 'oily t-zone']
        }
        
        skin_queries = queries.get(skin_type, [f'{skin_type} skin'])
        all_downloaded = []
        
        # Download from available APIs
        for query in skin_queries:
            print(f"\n🔍 Searching: {query}")
            
            # Try Unsplash
            if self.api_keys['unsplash']:
                downloaded = self.download_from_unsplash(query, count_per_source)
                all_downloaded.extend(downloaded)
            
            # Try Pexels
            if self.api_keys['pexels']:
                downloaded = self.download_from_pexels(query, count_per_source)
                all_downloaded.extend(downloaded)
            
            # Rate limiting between queries
            time.sleep(2)
        
        # Organize downloaded images
        organized = []
        print(f"\n📁 Organizing {len(all_downloaded)} images...")
        
        for filepath in all_downloaded:
            organized_path = self.classify_and_organize(filepath, skin_type)
            organized.append(organized_path)
        
        print(f"✅ Collected {len(organized)} {skin_type} skin images")
        return organized
    
    def download_all_skin_types(self, images_per_type=50):
        """Download images for all skin types"""
        
        print("🚀 AUTOMATED SKIN TYPE IMAGE COLLECTION")
        print("="*50)
        
        if not any(self.api_keys.values()):
            print("❌ No API keys configured!")
            self.setup_api_keys()
            return
        
        skin_types = ['dry', 'oily', 'normal', 'sensitive', 'combination']
        total_downloaded = {}
        
        for skin_type in skin_types:
            try:
                downloaded = self.download_skin_type_images(skin_type, images_per_type // 4)
                total_downloaded[skin_type] = len(downloaded)
                
                # Progress update
                print(f"📊 Progress: {skin_type} = {len(downloaded)} images")
                
                # Longer delay between skin types
                time.sleep(5)
                
            except Exception as e:
                print(f"❌ Error collecting {skin_type} images: {e}")
                total_downloaded[skin_type] = 0
        
        # Summary
        print(f"\n🎉 COLLECTION COMPLETE!")
        print("="*30)
        
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
    
    def setup_api_keys(self):
        """Interactive setup for API keys"""
        
        print("\n🔑 API KEY SETUP")
        print("="*20)
        print("To download images automatically, you need free API keys:")
        print()
        
        print("1. 📸 UNSPLASH (Recommended)")
        print("   • Go to: https://unsplash.com/developers")
        print("   • Create free account")
        print("   • Create new application")
        print("   • Copy 'Access Key'")
        print()
        
        print("2. 🌟 PEXELS (Optional)")
        print("   • Go to: https://www.pexels.com/api/")
        print("   • Create free account")
        print("   • Get API key")
        print()
        
        # Option to input keys
        unsplash_key = input("Enter Unsplash Access Key (or press Enter to skip): ").strip()
        if unsplash_key:
            self.api_keys['unsplash'] = unsplash_key
            print("✅ Unsplash key configured")
        
        pexels_key = input("Enter Pexels API Key (or press Enter to skip): ").strip()
        if pexels_key:
            self.api_keys['pexels'] = pexels_key
            print("✅ Pexels key configured")
        
        # Save to config file
        if any(self.api_keys.values()):
            self.save_api_keys()
    
    def save_api_keys(self):
        """Save API keys to config file"""
        
        config = {
            'api_keys': {k: v for k, v in self.api_keys.items() if v}
        }
        
        with open('downloader_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("💾 API keys saved to downloader_config.json")
    
    def load_api_keys(self):
        """Load API keys from config file"""
        
        try:
            with open('downloader_config.json', 'r') as f:
                config = json.load(f)
                self.api_keys.update(config.get('api_keys', {}))
            print("✅ Loaded API keys from config")
        except FileNotFoundError:
            pass  # No config file exists yet
    
    def download_from_freepik(self, query, count=30):
        """
        Scrape images from Freepik website
        Requires Selenium and Chrome WebDriver
        """
        
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium is not available")
            print("📝 Install with: pip install selenium")
            return []
        
        print(f"🌐 Scraping from Freepik: {query}")
        
        options = Options()
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Path to your ChromeDriver
        webdriver_service = None
        if os.name == 'posix':
            # Linux or Mac
            webdriver_service = webdriver.chrome.service.Service('/usr/bin/chromedriver')
        else:
            # Windows
            webdriver_service = webdriver.chrome.service.Service('chromedriver.exe')
        
        driver = webdriver.Chrome(service=webdriver_service, options=options)
        driver.implicitly_wait(10)  # Wait for elements to load
        
        downloaded = []
        
        try:
            # Build search URL
            search_url = f"https://www.freepik.com/search?format=search&query={quote(query)}"
            driver.get(search_url)
            
            # Wait for results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".gallery__image"))
            )
            
            # Parse results
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            image_elements = soup.select(".gallery__image img")
            
            for img in image_elements[:count]:
                try:
                    img_url = img['src']
                    img_url = img_url.replace('small', 'large')  # Get higher resolution
                    
                    # Download image
                    image_info = {
                        'url': img_url,
                        'description': query,
                        'source': 'freepik',
                        'query': query,
                        'id': str(int(time.time() * 1000))  # Unique ID based on timestamp
                    }
                    
                    filepath = self.download_single_image(image_info)
                    if filepath:
                        downloaded.append(filepath)
                        self.stats['freepik_downloaded'] += 1
                    
                except Exception as e:
                    print(f"❌ Error processing Freepik image: {e}")
        
        except Exception as e:
            print(f"❌ Freepik scraping error: {e}")
        
        finally:
            driver.quit()
        
        return downloaded
    
    def download_all_images(self, skin_type, count_per_source=20):
        """Download images for a specific skin type from all sources"""
        
        print(f"\n🎯 COLLECTING {skin_type.upper()} SKIN IMAGES FROM ALL SOURCES")
        print("="*50)
        
        # Define search queries for each skin type
        queries = {
            'dry': ['dry skin', 'dehydrated skin', 'flaky skin', 'rough skin texture'],
            'oily': ['oily skin', 'shiny skin', 'greasy skin', 'acne prone skin'],
            'sensitive': ['sensitive skin', 'irritated skin', 'red skin', 'reactive skin'],
            'normal': ['normal skin', 'healthy skin', 'balanced skin', 'clear skin'],
            'combination': ['combination skin', 'mixed skin type', 'oily t-zone']
        }
        
        skin_queries = queries.get(skin_type, [f'{skin_type} skin'])
        all_downloaded = []
        
        # Download from Unsplash, Pexels, and Freepik
        for query in skin_queries:
            print(f"\n🔍 Searching: {query}")
            
            # Unsplash
            if self.api_keys['unsplash']:
                downloaded = self.download_from_unsplash(query, count_per_source)
                all_downloaded.extend(downloaded)
            
            # Pexels
            if self.api_keys['pexels']:
                downloaded = self.download_from_pexels(query, count_per_source)
                all_downloaded.extend(downloaded)
            
            # Freepik (scraping)
            downloaded = self.download_from_freepik(query, count_per_source)
            all_downloaded.extend(downloaded)
            
            # Rate limiting between queries
            time.sleep(2)
        
        # Organize downloaded images
        organized = []
        print(f"\n📁 Organizing {len(all_downloaded)} images...")
        
        for filepath in all_downloaded:
            organized_path = self.classify_and_organize(filepath, skin_type)
            organized.append(organized_path)
        
        print(f"✅ Collected {len(organized)} {skin_type} skin images from all sources")
        return organized
    
    def download_all_skin_types(self, images_per_type=50):
        """Download images for all skin types"""
        
        print("🚀 AUTOMATED SKIN TYPE IMAGE COLLECTION")
        print("="*50)
        
        if not any(self.api_keys.values()):
            print("❌ No API keys configured!")
            self.setup_api_keys()
            return
        
        skin_types = ['dry', 'oily', 'normal', 'sensitive', 'combination']
        total_downloaded = {}
        
        for skin_type in skin_types:
            try:
                downloaded = self.download_all_images(skin_type, images_per_type // 4)
                total_downloaded[skin_type] = len(downloaded)
                
                # Progress update
                print(f"📊 Progress: {skin_type} = {len(downloaded)} images")
                
                # Longer delay between skin types
                time.sleep(5)
                
            except Exception as e:
                print(f"❌ Error collecting {skin_type} images: {e}")
                total_downloaded[skin_type] = 0
        
        # Summary
        print(f"\n🎉 COLLECTION COMPLETE!")
        print("="*30)
        
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
    
    def setup_api_keys(self):
        """Interactive setup for API keys"""
        
        print("\n🔑 API KEY SETUP")
        print("="*20)
        print("To download images automatically, you need free API keys:")
        print()
        
        print("1. 📸 UNSPLASH (Recommended)")
        print("   • Go to: https://unsplash.com/developers")
        print("   • Create free account")
        print("   • Create new application")
        print("   • Copy 'Access Key'")
        print()
        
        print("2. 🌟 PEXELS (Optional)")
        print("   • Go to: https://www.pexels.com/api/")
        print("   • Create free account")
        print("   • Get API key")
        print()
        
        # Option to input keys
        unsplash_key = input("Enter Unsplash Access Key (or press Enter to skip): ").strip()
        if unsplash_key:
            self.api_keys['unsplash'] = unsplash_key
            print("✅ Unsplash key configured")
        
        pexels_key = input("Enter Pexels API Key (or press Enter to skip): ").strip()
        if pexels_key:
            self.api_keys['pexels'] = pexels_key
            print("✅ Pexels key configured")
        
        # Save to config file
        if any(self.api_keys.values()):
            self.save_api_keys()
    
    def save_api_keys(self):
        """Save API keys to config file"""
        
        config = {
            'api_keys': {k: v for k, v in self.api_keys.items() if v}
        }
        
        with open('downloader_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("💾 API keys saved to downloader_config.json")
    
    def load_api_keys(self):
        """Load API keys from config file"""
        
        try:
            with open('downloader_config.json', 'r') as f:
                config = json.load(f)
                self.api_keys.update(config.get('api_keys', {}))
            print("✅ Loaded API keys from config")
        except FileNotFoundError:
            pass  # No config file exists yet

    def scrape_freepik_dry_skin(self, max_images=50):
        """
        Scrape dry skin images from Freepik using Selenium
        """
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium not available. Install with: pip install selenium")
            return []
        
        print(f"🎨 Starting Freepik collection for dry skin images...")
        
        # Setup Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(f'--user-agent={self.ua.random}')
        
        downloaded_files = []
        
        try:
            # Initialize Chrome driver
            driver = webdriver.Chrome(options=chrome_options)
            
            # Search terms for dry skin
            search_terms = [
                "dry skin texture",
                "flaky skin",
                "dehydrated skin", 
                "rough skin",
                "dry facial skin",
                "skin dryness"
            ]
            
            for search_term in search_terms:
                print(f"  🔍 Searching for: {search_term}")
                
                # Navigate to Freepik search
                search_url = f"https://www.freepik.com/search?format=search&query={quote(search_term)}&type=photo"
                driver.get(search_url)
                
                # Wait for page load
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='search-result']"))
                    )
                except:
                    print(f"    ⚠️  Page load timeout for: {search_term}")
                    continue
                
                # Scroll to load more images
                self._scroll_freepik_page(driver, max_images_per_term=10)
                
                # Find and download images
                image_elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid='search-result'] img")
                
                downloaded_count = 0
                for img_element in image_elements[:10]:  # Limit per search term
                    if len(downloaded_files) >= max_images:
                        break
                    
                    try:
                        # Get image source URL
                        img_src = img_element.get_attribute('src')
                        if not img_src or 'placeholder' in img_src.lower():
                            continue
                        
                        # Create image info
                        image_info = {
                            'url': img_src,
                            'source': 'freepik',
                            'query': search_term,
                            'skin_type': 'dry',
                            'id': f"{int(time.time())}_{random.randint(1000,9999)}"
                        }
                        
                        # Generate filename
                        filename = self.generate_filename(image_info)
                        
                        # Download image
                        filepath = self.download_single_image(img_src, filename)
                        
                        if filepath:
                            downloaded_files.append({
                                'filepath': filepath,
                                'info': image_info
                            })
                            downloaded_count += 1
                            self.stats['freepik_downloaded'] += 1
                            self.stats['total_downloaded'] += 1
                            print(f"    ✅ Downloaded: {filename}")
                        
                        # Rate limiting
                        time.sleep(self.request_delay)
                        
                    except Exception as e:
                        print(f"    ❌ Error downloading image: {e}")
                        self.stats['failed_downloads'] += 1
                
                print(f"    📊 Downloaded {downloaded_count} images for '{search_term}'")
                
                # Delay between search terms
                time.sleep(3)
            
            driver.quit()
            
        except Exception as e:
            print(f"❌ Freepik scraping error: {e}")
            if 'driver' in locals():
                driver.quit()
        
        return downloaded_files
    
    def _scroll_freepik_page(self, driver, max_images_per_term=10):
        """Scroll Freepik page to load more images"""
        scrolls = 0
        max_scrolls = 3  # Limit scrolling
        
        while scrolls < max_scrolls:
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait for new content
            time.sleep(2)
            
            # Check number of loaded images
            images = driver.find_elements(By.CSS_SELECTOR, "[data-testid='search-result'] img")
            if len(images) >= max_images_per_term:
                break
            
            scrolls += 1
    
    def scrape_all_freepik_skin_types(self, images_per_type=20):
        """
        Scrape images for all skin types from Freepik
        """
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium not available for Freepik scraping")
            return []
        
        print("🎨 COMPREHENSIVE FREEPIK COLLECTION")
        print("="*40)
        
        skin_type_searches = {
            'dry': [
                'dry skin texture',
                'flaky skin',
                'dehydrated skin',
                'rough skin'
            ],
            'oily': [
                'oily skin',
                'shiny skin',
                'acne prone skin',
                'greasy skin'
            ],
            'sensitive': [
                'sensitive skin',
                'irritated skin',
                'red skin',
                'reactive skin'
            ],
            'normal': [
                'normal skin',
                'healthy skin',
                'balanced skin',
                'clear skin'
            ],
            'combination': [
                'combination skin',
                'mixed skin type',
                't-zone oily skin'
            ]
        }
        
        all_downloaded = []
        
        for skin_type, search_terms in skin_type_searches.items():
            print(f"\n📸 Collecting {skin_type} skin images...")
            
            type_downloaded = self._scrape_freepik_skin_type(
                skin_type, search_terms, images_per_type
            )
            all_downloaded.extend(type_downloaded)
            
            self.stats['by_skin_type'][skin_type] += len(type_downloaded)
            
            # Delay between skin types
            time.sleep(5)
        
        return all_downloaded
    
    def _scrape_freepik_skin_type(self, skin_type, search_terms, max_images):
        """Scrape specific skin type from Freepik"""
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f'--user-agent={self.ua.random}')
        
        downloaded_files = []
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            
            images_per_term = max_images // len(search_terms)
            
            for search_term in search_terms:
                if len(downloaded_files) >= max_images:
                    break
                
                print(f"  🔍 Searching: {search_term}")
                
                # Navigate to search
                search_url = f"https://www.freepik.com/search?format=search&query={quote(search_term)}&type=photo"
                driver.get(search_url)
                
                # Wait and scroll
                time.sleep(3)
                self._scroll_freepik_page(driver, images_per_term)
                
                # Download images
                image_elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid='search-result'] img")
                
                for img_element in image_elements[:images_per_term]:
                    if len(downloaded_files) >= max_images:
                        break
                    
                    try:
                        img_src = img_element.get_attribute('src')
                        if not img_src or 'placeholder' in img_src.lower():
                            continue
                        
                        image_info = {
                            'url': img_src,
                            'source': 'freepik',
                            'query': search_term,
                            'skin_type': skin_type,
                            'id': f"{int(time.time())}_{random.randint(1000,9999)}"
                        }
                        
                        filename = self.generate_filename(image_info)
                        filepath = self.download_single_image(img_src, filename)
                        
                        if filepath:
                            downloaded_files.append({
                                'filepath': filepath,
                                'info': image_info
                            })
                            print(f"    ✅ {skin_type}: {os.path.basename(filepath)}")
                        
                        time.sleep(self.request_delay)
                        
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                
                time.sleep(2)
            
            driver.quit()
            
        except Exception as e:
            print(f"❌ Error scraping {skin_type}: {e}")
            if 'driver' in locals():
                driver.quit()
        
        return downloaded_files

def create_manual_downloader():
    """Create a simpler manual download helper for when APIs aren't available"""
    
    manual_script = '''#!/usr/bin/env python3
"""
Manual Download Helper - For when APIs aren't available
Helps organize manually downloaded images
"""

import os
import shutil
from PIL import Image
import hashlib

class ManualDownloadHelper:
    """Helper for organizing manually downloaded images"""
    
    def __init__(self):
        self.downloads_dir = "downloads"
        self.training_dir = "training_dataset"
        self.skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        
        # Setup directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create directory structure"""
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        for skin_type in self.skin_types:
            os.makedirs(f"{self.training_dir}/train/{skin_type}", exist_ok=True)
    
    def organize_downloads(self):
        """Organize images from downloads folder"""
        
        if not os.path.exists(self.downloads_dir):
            print(f"❌ Downloads directory not found: {self.downloads_dir}")
            print("📝 Create it and add your downloaded images")
            return
        
        print("📁 ORGANIZING DOWNLOADED IMAGES")
        print("="*35)
        
        organized = {skin_type: 0 for skin_type in self.skin_types}
        
        for filename in os.listdir(self.downloads_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(self.downloads_dir, filename)
                
                # Validate image
                if self.validate_image(filepath):
                    # Classify and move
                    skin_type = self.classify_image(filename)
                    target_path = self.move_to_training(filepath, skin_type)
                    
                    if target_path:
                        organized[skin_type] += 1
                        print(f"✅ {filename} → {skin_type}/")
                else:
                    print(f"❌ Invalid image: {filename}")
        
        # Summary
        print(f"\\n📊 ORGANIZATION COMPLETE:")
        total = sum(organized.values())
        for skin_type, count in organized.items():
            print(f"  {skin_type:12}: {count} images")
        print(f"  {'TOTAL':12}: {total} images")
        
        return organized
    
    def validate_image(self, filepath):
        """Validate image quality"""
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                return width >= 224 and height >= 224
        except:
            return False
    
    def classify_image(self, filename):
        """Classify image based on filename"""
        filename_lower = filename.lower()
        
        keywords = {
            'dry': ['dry', 'flaky', 'rough', 'dehydrated'],
            'oily': ['oily', 'shiny', 'greasy', 'acne'],
            'sensitive': ['sensitive', 'red', 'irritated'],
            'combination': ['combination', 'mixed', 'combo'],
            'normal': ['normal', 'balanced', 'healthy']
        }
        
        for skin_type, terms in keywords.items():
            for term in terms:
                if term in filename_lower:
                    return skin_type
        
        return 'normal'  # Default
    
    def move_to_training(self, source_path, skin_type):
        """Move image to training directory"""
        filename = os.path.basename(source_path)
        target_dir = f"{self.training_dir}/train/{skin_type}"
        target_path = os.path.join(target_dir, filename)
        
        # Handle duplicates
        counter = 1
        while os.path.exists(target_path):
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{counter}{ext}"
            target_path = os.path.join(target_dir, new_filename)
            counter += 1
        
        try:
            shutil.move(source_path, target_path)
            return target_path
        except Exception as e:
            print(f"❌ Error moving {filename}: {e}")
            return None

def main():
    """Main function for manual organization"""
    helper = ManualDownloadHelper()
    organized = helper.organize_downloads()
    
    total_images = sum(organized.values())
    if total_images >= 100:
        print("\\n🚀 Ready for training! Run:")
        print("python enhanced_training_pipeline.py")
    else:
        print(f"\\n📈 Download {100 - total_images} more images for best results")

if __name__ == "__main__":
    main()
'''
    
    with open('manual_download_helper.py', 'w') as f:
        f.write(manual_script)
    
    return manual_script

def main():
    """Main function for automated downloading"""
    
    print("🤖 AUTOMATED SKIN TYPE IMAGE DOWNLOADER")
    print("="*45)
    
    # Initialize downloader
    downloader = AutomatedImageDownloader()
    
    # Load existing API keys
    downloader.load_api_keys()
    
    # Check if API keys are available
    if not any(downloader.api_keys.values()):
        print("⚠️  No API keys found for automated downloading")
        print("🔧 Setting up API keys...")
        downloader.setup_api_keys()
        
        if not any(downloader.api_keys.values()):
            print("❌ No API keys configured")
            print("📝 Creating manual download helper instead...")
            create_manual_downloader()
            print("✅ Created: manual_download_helper.py")
            print("💡 Use this for organizing manually downloaded images")
            return
    
    # Start automated collection
    print("\n🚀 Starting automated image collection...")
    total_downloaded = downloader.download_all_skin_types(images_per_type=100)
    
    # Final summary
    total_images = sum(total_downloaded.values())
    print(f"\n🎉 COLLECTION SUMMARY:")
    print(f"Total images collected: {total_images}")
    
    if total_images >= 250:
        print("🏆 Excellent! Ready for professional-level training")
    elif total_images >= 100:
        print("👍 Good! Ready for enhanced training")
    else:
        print("📈 Consider collecting more images for better accuracy")

def run_freepik_collection():
    """Enhanced main function focused on Freepik collection"""
    
    print("🎨 FREEPIK AUTOMATED COLLECTION")
    print("="*40)
    
    # Check Chrome installation
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        test_driver = webdriver.Chrome(options=chrome_options)
        test_driver.quit()
        print("✅ Chrome WebDriver available")
    except Exception as e:
        print("❌ Chrome WebDriver not found")
        print("📋 Please install Chrome browser and ChromeDriver")
        print("   macOS: brew install chromedriver")
        print("   Or download from: https://chromedriver.chromium.org/")
        return
    
    # Initialize downloader
    downloader = AutomatedImageDownloader()
    
    print("\n🎯 COLLECTION OPTIONS:")
    print("1. 🏥 Dry skin images only (quick start)")
    print("2. 🔄 All skin types (comprehensive)")
    print("3. 📱 Custom collection")
    
    choice = input("\nEnter choice (1-3) or press Enter for option 1: ").strip()
    
    if choice == "2":
        # Comprehensive collection
        print("\n🚀 Starting comprehensive Freepik collection...")
        all_files = downloader.scrape_all_freepik_skin_types(images_per_type=50)
        
        # Organize files
        organized_count = 0
        for file_data in all_files:
            skin_type = file_data['info']['skin_type']
            success = downloader.classify_and_organize(
                file_data['filepath'], 
                target_skin_type=skin_type
            )
            if success:
                organized_count += 1
        
        print(f"\n📊 COMPREHENSIVE COLLECTION SUMMARY:")
        print(f"Total downloaded: {len(all_files)}")
        print(f"Successfully organized: {organized_count}")
        
    elif choice == "3":
        # Custom collection
        search_term = input("\nEnter custom search term (e.g., 'dry skin'): ").strip()
        max_images = int(input("Enter max images to collect (e.g., 30): ") or "30")
        
        print(f"\n🔍 Searching Freepik for: {search_term}")
        # Custom implementation would go here
        
    else:
        # Default: Dry skin collection
        print("\n🏥 Starting dry skin collection from Freepik...")
        
        dry_files = downloader.scrape_freepik_dry_skin(max_images=50)
        
        # Organize dry skin images
        organized_count = 0
        for file_data in dry_files:
            success = downloader.classify_and_organize(
                file_data['filepath'], 
                target_skin_type='dry'
            )
            if success:
                organized_count += 1
        
        print(f"\n📊 DRY SKIN COLLECTION SUMMARY:")
        print(f"Downloaded: {len(dry_files)} images")
        print(f"Organized: {organized_count} images")
        print(f"Location: training_dataset/train/dry/")
    
    # Final statistics
    downloader.print_final_statistics()
    
    # Next steps
    print(f"\n🚀 NEXT STEPS:")
    print("1. Review collected images for quality")
    print("2. Run enhanced training:")
    print("   python enhanced_training_pipeline.py")
    print("3. Test improved model in web app")

def print_final_statistics(self):
    """Print collection statistics"""
    print(f"\n📈 COLLECTION STATISTICS:")
    print("="*30)
    print(f"🎨 Freepik images: {self.stats['freepik_downloaded']}")
    print(f"📸 Unsplash images: {self.stats['unsplash_downloaded']}")
    print(f"🌟 Pexels images: {self.stats['pexels_downloaded']}")
    print(f"❌ Failed downloads: {self.stats['failed_downloads']}")
    print(f"📊 Total successful: {self.stats['total_downloaded']}")
    
    print(f"\n📁 By Skin Type:")
    for skin_type, count in self.stats['by_skin_type'].items():
        print(f"  {skin_type:12}: {count:3} images")

# Add method to class
AutomatedImageDownloader.print_final_statistics = print_final_statistics

if __name__ == "__main__":
    # Check if user wants Freepik collection or general collection
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "freepik":
        run_freepik_collection()
    else:
        print("🤖 AUTOMATED IMAGE COLLECTION")
        print("="*30)
        print("Options:")
        print("  python automated_image_downloader.py freepik  # Freepik collection")
        print("  python automated_image_downloader.py          # General collection")
        print()
        
        choice = input("Run Freepik collection? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '']:
            run_freepik_collection()
        else:
            main()
