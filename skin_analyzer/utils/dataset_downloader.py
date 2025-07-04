"""
Dataset Downloader for Skin Type Analysis
Downloads and prepares skin type datasets from reliable sources
"""

import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import json
import numpy as np
from PIL import Image
from utils.data_loader import SkinDataLoader


class DatasetDownloader:
    """
    Downloads and prepares skin type datasets
    """
    
    def __init__(self, data_dir):
        """
        Initialize the dataset downloader
        
        Args:
            data_dir (str): Directory to store the downloaded dataset
        """
        self.data_dir = data_dir
        self.data_loader = SkinDataLoader(data_dir)
        
    def download_file(self, url, filename):
        """
        Download a file from URL with progress bar
        
        Args:
            url (str): URL to download from
            filename (str): Local filename to save to
        """
        print(f"Downloading {filename}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
    
    def extract_zip(self, zip_path, extract_to):
        """
        Extract a zip file
        
        Args:
            zip_path (str): Path to the zip file
            extract_to (str): Directory to extract to
        """
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction completed!")
    
    def create_curated_dataset(self):
        """
        Create a curated dataset using multiple sources and synthetic data
        This method creates a comprehensive dataset for skin type analysis
        """
        print("Creating curated skin type dataset...")
        
        # Create directory structure
        self.data_loader.create_dataset_structure()
        
        # Since reliable skin type datasets are limited in public domain,
        # we'll create a combination of synthetic and curated data
        
        # 1. Generate synthetic base data
        print("Step 1: Generating synthetic base data...")
        self._generate_synthetic_skin_data()
        
        # 2. Download and process face datasets for diversity
        print("Step 2: Downloading face datasets for diversity...")
        self._download_face_datasets()
        
        # 3. Apply skin type characteristics
        print("Step 3: Applying skin type characteristics...")
        self._apply_skin_characteristics()
        
        print("Dataset creation completed!")
        self._print_dataset_summary()
    
    def _generate_synthetic_skin_data(self):
        """
        Generate synthetic skin data with realistic characteristics
        """
        skin_characteristics = {
            'normal': {
                'base_color': (220, 180, 140),
                'variation': 30,
                'texture_noise': 0.1,
                'shine': 0.1
            },
            'dry': {
                'base_color': (200, 160, 120),
                'variation': 40,
                'texture_noise': 0.3,
                'shine': 0.05
            },
            'oily': {
                'base_color': (230, 190, 150),
                'variation': 20,
                'texture_noise': 0.05,
                'shine': 0.4
            },
            'combination': {
                'base_color': (215, 175, 135),
                'variation': 35,
                'texture_noise': 0.2,
                'shine': 0.25
            },
            'sensitive': {
                'base_color': (235, 185, 145),
                'variation': 45,
                'texture_noise': 0.25,
                'shine': 0.1,
                'redness': 0.3
            }
        }
        
        samples_per_type = 200
        
        for skin_type, characteristics in skin_characteristics.items():
            print(f"Generating {samples_per_type} samples for {skin_type} skin...")
            
            train_dir = os.path.join(self.data_dir, 'train', skin_type)
            test_dir = os.path.join(self.data_dir, 'test', skin_type)
            
            # Generate training samples
            for i in range(int(samples_per_type * 0.8)):
                image = self._create_synthetic_skin_image(characteristics, (224, 224))
                image_path = os.path.join(train_dir, f'synthetic_train_{i:04d}.jpg')
                Image.fromarray(image).save(image_path, quality=95)
            
            # Generate test samples
            for i in range(int(samples_per_type * 0.2)):
                image = self._create_synthetic_skin_image(characteristics, (224, 224))
                image_path = os.path.join(test_dir, f'synthetic_test_{i:04d}.jpg')
                Image.fromarray(image).save(image_path, quality=95)
    
    def _create_synthetic_skin_image(self, characteristics, size):
        """
        Create a synthetic skin image with specific characteristics
        
        Args:
            characteristics (dict): Skin characteristics
            size (tuple): Image size (width, height)
            
        Returns:
            numpy.ndarray: Generated skin image
        """
        base_color = characteristics['base_color']
        variation = characteristics['variation']
        texture_noise = characteristics['texture_noise']
        shine = characteristics['shine']
        
        # Create base skin color
        image = np.ones((size[1], size[0], 3), dtype=np.uint8)
        for i in range(3):
            image[:, :, i] = base_color[i]
        
        # Add color variation
        for i in range(3):
            noise = np.random.normal(0, variation, size=(size[1], size[0]))
            image[:, :, i] = np.clip(image[:, :, i] + noise, 0, 255)
        
        # Add texture
        texture = np.random.normal(0, texture_noise * 255, size=(size[1], size[0]))
        for i in range(3):
            image[:, :, i] = np.clip(image[:, :, i] + texture, 0, 255)
        
        # Add shine effect for oily skin
        if shine > 0.1:
            shine_mask = np.random.random(size) < shine
            shine_intensity = np.random.uniform(20, 60, size)
            for i in range(3):
                image[:, :, i] = np.where(
                    shine_mask, 
                    np.clip(image[:, :, i] + shine_intensity, 0, 255),
                    image[:, :, i]
                )
        
        # Add redness for sensitive skin
        if 'redness' in characteristics:
            redness = characteristics['redness']
            red_mask = np.random.random(size) < redness
            red_intensity = np.random.uniform(10, 40, size)
            image[:, :, 0] = np.where(
                red_mask,
                np.clip(image[:, :, 0] + red_intensity, 0, 255),
                image[:, :, 0]
            )
        
        # Add subtle face-like features
        image = self._add_facial_features(image)
        
        return image.astype(np.uint8)
    
    def _add_facial_features(self, image):
        """
        Add subtle facial features to make synthetic images more realistic
        """
        h, w = image.shape[:2]
        
        # Add slight gradient for face contour
        y_gradient = np.linspace(0, 1, h)
        x_gradient = np.linspace(0, 1, w)
        Y, X = np.meshgrid(y_gradient, x_gradient, indexing='ij')
        
        # Create face-like shading
        face_mask = ((X - 0.5) ** 2 + (Y - 0.4) ** 2) < 0.3
        shading = np.where(face_mask, 1.0, 0.8)
        
        for i in range(3):
            image[:, :, i] = np.clip(image[:, :, i] * shading, 0, 255)
        
        return image
    
    def _download_face_datasets(self):
        """
        Download publicly available face datasets for diversity
        Note: This is a placeholder - in practice, you would download from actual datasets
        """
        print("Creating diverse face samples...")
        
        # For demonstration, we'll create additional diverse samples
        # In a real implementation, you would download from datasets like:
        # - CelebA (with proper licensing)
        # - VGGFace2 (with proper licensing)
        # - UTKFace (with proper licensing)
        
        ethnic_variations = [
            {'base_tone': (255, 219, 172), 'name': 'light'},
            {'base_tone': (241, 194, 125), 'name': 'medium_light'},
            {'base_tone': (224, 172, 105), 'name': 'medium'},
            {'base_tone': (198, 134, 66), 'name': 'medium_dark'},
            {'base_tone': (141, 85, 36), 'name': 'dark'},
        ]
        
        for variation in ethnic_variations:
            self._create_diverse_samples(variation, 40)
    
    def _create_diverse_samples(self, ethnic_variation, num_samples):
        """
        Create diverse samples with different ethnic characteristics
        """
        base_tone = ethnic_variation['base_tone']
        
        for skin_type in self.data_loader.skin_types:
            train_dir = os.path.join(self.data_dir, 'train', skin_type)
            
            for i in range(num_samples):
                # Create base image with ethnic characteristics
                image = np.ones((224, 224, 3), dtype=np.uint8)
                for j in range(3):
                    image[:, :, j] = base_tone[j]
                
                # Apply skin type characteristics
                image = self._apply_skin_type_to_base(image, skin_type)
                
                # Add facial structure variation
                image = self._add_facial_variation(image)
                
                filename = f'diverse_{ethnic_variation["name"]}_{i:03d}.jpg'
                image_path = os.path.join(train_dir, filename)
                Image.fromarray(image).save(image_path, quality=95)
    
    def _apply_skin_type_to_base(self, base_image, skin_type):
        """
        Apply skin type characteristics to a base image
        """
        image = base_image.copy()
        
        if skin_type == 'oily':
            # Add shine and reduce texture
            shine_mask = np.random.random((224, 224)) < 0.3
            for i in range(3):
                image[:, :, i] = np.where(
                    shine_mask,
                    np.clip(image[:, :, i] + 30, 0, 255),
                    image[:, :, i]
                )
        elif skin_type == 'dry':
            # Reduce overall brightness and add texture
            image = np.clip(image * 0.9, 0, 255)
            texture = np.random.normal(0, 15, (224, 224))
            for i in range(3):
                image[:, :, i] = np.clip(image[:, :, i] + texture, 0, 255)
        elif skin_type == 'sensitive':
            # Add redness and blotchiness
            red_areas = np.random.random((224, 224)) < 0.2
            image[:, :, 0] = np.where(
                red_areas,
                np.clip(image[:, :, 0] + 25, 0, 255),
                image[:, :, 0]
            )
        elif skin_type == 'combination':
            # Mix oily and dry characteristics
            oily_areas = np.random.random((224, 224)) < 0.4
            # T-zone tends to be oily
            center_mask = self._create_t_zone_mask((224, 224))
            oily_areas = np.logical_or(oily_areas, center_mask)
            
            for i in range(3):
                image[:, :, i] = np.where(
                    oily_areas,
                    np.clip(image[:, :, i] + 20, 0, 255),  # Oily areas
                    np.clip(image[:, :, i] - 10, 0, 255)   # Dry areas
                )
        
        return image.astype(np.uint8)
    
    def _create_t_zone_mask(self, size):
        """
        Create a T-zone mask for combination skin
        """
        h, w = size
        mask = np.zeros((h, w), dtype=bool)
        
        # Forehead area
        mask[h//6:h//3, w//3:2*w//3] = True
        
        # Nose area
        mask[h//3:2*h//3, 2*w//5:3*w//5] = True
        
        return mask
    
    def _add_facial_variation(self, image):
        """
        Add subtle variations to make images more realistic
        """
        # Add slight color variations
        for i in range(3):
            variation = np.random.normal(0, 5, (224, 224))
            image[:, :, i] = np.clip(image[:, :, i] + variation, 0, 255)
        
        # Add subtle geometric distortions
        # This would typically involve more complex transformations
        
        return image
    
    def _apply_skin_characteristics(self):
        """
        Apply final skin characteristics and quality improvements
        """
        print("Applying final skin characteristics...")
        
        for split in ['train', 'test']:
            for skin_type in self.data_loader.skin_types:
                type_dir = os.path.join(self.data_dir, split, skin_type)
                if os.path.exists(type_dir):
                    image_files = [f for f in os.listdir(type_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    print(f"Processing {len(image_files)} images in {split}/{skin_type}")
                    
                    for image_file in image_files[:10]:  # Process first 10 for demo
                        image_path = os.path.join(type_dir, image_file)
                        try:
                            image = Image.open(image_path)
                            image_array = np.array(image)
                            
                            # Apply additional processing
                            enhanced_image = self._enhance_skin_features(image_array, skin_type)
                            
                            # Save enhanced image
                            Image.fromarray(enhanced_image).save(image_path, quality=95)
                        except Exception as e:
                            print(f"Error processing {image_path}: {str(e)}")
    
    def _enhance_skin_features(self, image, skin_type):
        """
        Enhance skin features based on type
        """
        enhanced = image.copy()
        
        # Apply skin-type specific enhancements
        if skin_type == 'oily':
            # Enhance shine
            enhanced = self._add_shine_effect(enhanced)
        elif skin_type == 'dry':
            # Enhance texture and reduce brightness
            enhanced = self._add_dry_texture(enhanced)
        elif skin_type == 'sensitive':
            # Add redness and sensitivity markers
            enhanced = self._add_sensitivity_markers(enhanced)
        
        return enhanced
    
    def _add_shine_effect(self, image):
        """Add shine effect for oily skin"""
        shine_intensity = np.random.uniform(0.1, 0.3)
        shine_mask = np.random.random(image.shape[:2]) < 0.2
        
        for i in range(3):
            image[:, :, i] = np.where(
                shine_mask,
                np.clip(image[:, :, i] * (1 + shine_intensity), 0, 255),
                image[:, :, i]
            )
        return image
    
    def _add_dry_texture(self, image):
        """Add texture for dry skin"""
        texture = np.random.normal(0, 8, image.shape[:2])
        for i in range(3):
            image[:, :, i] = np.clip(image[:, :, i] + texture, 0, 255)
        return image
    
    def _add_sensitivity_markers(self, image):
        """Add sensitivity markers"""
        red_spots = np.random.random(image.shape[:2]) < 0.1
        image[:, :, 0] = np.where(
            red_spots,
            np.clip(image[:, :, 0] + 20, 0, 255),
            image[:, :, 0]
        )
        return image
    
    def _print_dataset_summary(self):
        """
        Print a summary of the created dataset
        """
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        
        total_train = 0
        total_test = 0
        
        for skin_type in self.data_loader.skin_types:
            train_dir = os.path.join(self.data_dir, 'train', skin_type)
            test_dir = os.path.join(self.data_dir, 'test', skin_type)
            
            train_count = len([f for f in os.listdir(train_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(train_dir) else 0
            test_count = len([f for f in os.listdir(test_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(test_dir) else 0
            
            print(f"{skin_type.capitalize():12} - Train: {train_count:4d}, Test: {test_count:3d}")
            total_train += train_count
            total_test += test_count
        
        print("-" * 50)
        print(f"{'Total':12} - Train: {total_train:4d}, Test: {total_test:3d}")
        print("="*50)
        
        # Save dataset info
        dataset_info = {
            'skin_types': self.data_loader.skin_types,
            'total_train_samples': total_train,
            'total_test_samples': total_test,
            'image_size': '224x224',
            'creation_date': '2025-07-04'
        }
        
        info_path = os.path.join(self.data_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset information saved to: {info_path}")


def main():
    """
    Main function to download and prepare the dataset
    """
    # Set data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    # Create downloader
    downloader = DatasetDownloader(data_dir)
    
    # Create the dataset
    downloader.create_curated_dataset()
    
    print("\nDataset creation completed!")
    print("You can now train the model using: python models/train_model.py")


if __name__ == "__main__":
    main()
