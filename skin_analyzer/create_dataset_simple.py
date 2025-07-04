"""
Simplified Dataset Creator - Works without TensorFlow
Creates a synthetic dataset for skin type classification
"""

import os
import numpy as np
from PIL import Image
import json
import random
from tqdm import tqdm

class SimplifiedDatasetCreator:
    """
    Creates a synthetic dataset for skin type classification without TensorFlow dependencies
    """
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        
    def create_dataset_structure(self):
        """Create the directory structure for the dataset"""
        print("Creating dataset directory structure...")
        
        # Create main directories
        for split in ['train', 'test']:
            for skin_type in self.skin_types:
                dir_path = os.path.join(self.data_dir, split, skin_type)
                os.makedirs(dir_path, exist_ok=True)
        
        print("✅ Directory structure created")
    
    def create_synthetic_skin_image(self, skin_type, image_size=(224, 224)):
        """
        Create a synthetic skin image with specific characteristics
        """
        # Base skin colors for different types
        skin_characteristics = {
            'normal': {
                'base_color': (220, 180, 140),
                'variation': 25,
                'shine_prob': 0.1,
                'texture_strength': 0.15
            },
            'dry': {
                'base_color': (200, 160, 120),
                'variation': 35,
                'shine_prob': 0.05,
                'texture_strength': 0.3
            },
            'oily': {
                'base_color': (230, 190, 150),
                'variation': 20,
                'shine_prob': 0.4,
                'texture_strength': 0.1
            },
            'combination': {
                'base_color': (215, 175, 135),
                'variation': 30,
                'shine_prob': 0.25,
                'texture_strength': 0.2
            },
            'sensitive': {
                'base_color': (235, 185, 145),
                'variation': 40,
                'shine_prob': 0.1,
                'texture_strength': 0.25,
                'redness_prob': 0.3
            }
        }
        
        char = skin_characteristics[skin_type]
        
        # Create base image
        image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Apply base skin color
        for i in range(3):
            image[:, :, i] = char['base_color'][i]
        
        # Add color variation
        for i in range(3):
            noise = np.random.normal(0, char['variation'], image_size)
            image[:, :, i] = np.clip(image[:, :, i] + noise, 0, 255)
        
        # Add texture
        texture = np.random.normal(0, char['texture_strength'] * 50, image_size)
        for i in range(3):
            image[:, :, i] = np.clip(image[:, :, i] + texture, 0, 255)
        
        # Add shine (for oily skin)
        if random.random() < char['shine_prob']:
            shine_mask = np.random.random(image_size) < 0.3
            shine_intensity = np.random.uniform(20, 50, image_size)
            for i in range(3):
                image[:, :, i] = np.where(
                    shine_mask,
                    np.clip(image[:, :, i] + shine_intensity, 0, 255),
                    image[:, :, i]
                )
        
        # Add redness (for sensitive skin)
        if skin_type == 'sensitive' and random.random() < char.get('redness_prob', 0):
            red_mask = np.random.random(image_size) < 0.2
            red_intensity = np.random.uniform(15, 40, image_size)
            image[:, :, 0] = np.where(
                red_mask,
                np.clip(image[:, :, 0] + red_intensity, 0, 255),
                image[:, :, 0]
            )
        
        # Create face-like oval shape
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        for y in range(image_size[1]):
            for x in range(image_size[0]):
                # Create oval mask
                distance = ((x - center_x) / (image_size[0] * 0.4))**2 + ((y - center_y) / (image_size[1] * 0.45))**2
                if distance > 1:
                    # Outside face - make background
                    image[y, x] = [50, 50, 50]
        
        # Add basic facial features
        self._add_facial_features(image, center_x, center_y)
        
        return image.astype(np.uint8)
    
    def _add_facial_features(self, image, center_x, center_y):
        """Add basic facial features to make the image more realistic"""
        h, w = image.shape[:2]
        
        # Eyes
        eye_y = center_y - 30
        for eye_x in [center_x - 40, center_x + 40]:
            if 0 <= eye_y < h and 0 <= eye_x < w:
                for dy in range(-8, 8):
                    for dx in range(-12, 12):
                        if (0 <= eye_y + dy < h and 0 <= eye_x + dx < w and 
                            dx*dx + dy*dy <= 64):  # Circle
                            image[eye_y + dy, eye_x + dx] = [80, 80, 80]
        
        # Nose
        nose_y = center_y
        for dy in range(-15, 15):
            for dx in range(-4, 4):
                if 0 <= nose_y + dy < h and 0 <= center_x + dx < w:
                    image[nose_y + dy, center_x + dx] = np.clip(
                        image[nose_y + dy, center_x + dx] - 15, 0, 255
                    )
        
        # Mouth
        mouth_y = center_y + 35
        for dy in range(-4, 4):
            for dx in range(-20, 20):
                if (0 <= mouth_y + dy < h and 0 <= center_x + dx < w and
                    abs(dx) > 5):  # Skip center for mouth opening
                    image[mouth_y + dy, center_x + dx] = [120, 80, 80]
    
    def create_dataset(self, samples_per_type=200):
        """
        Create the complete synthetic dataset
        """
        print(f"Creating synthetic dataset with {samples_per_type} samples per skin type...")
        
        # Create directory structure
        self.create_dataset_structure()
        
        dataset_info = {
            'skin_types': self.skin_types,
            'samples_per_type': samples_per_type,
            'image_size': [224, 224, 3],
            'train_split': 0.8,
            'test_split': 0.2
        }
        
        total_created = 0
        
        for skin_type in self.skin_types:
            print(f"\nCreating {skin_type} skin samples...")
            
            # Calculate train/test split
            train_samples = int(samples_per_type * 0.8)
            test_samples = samples_per_type - train_samples
            
            # Create training samples
            train_dir = os.path.join(self.data_dir, 'train', skin_type)
            for i in tqdm(range(train_samples), desc=f"Train {skin_type}"):
                image = self.create_synthetic_skin_image(skin_type)
                image_path = os.path.join(train_dir, f'{skin_type}_train_{i:04d}.jpg')
                Image.fromarray(image).save(image_path, quality=90)
                total_created += 1
            
            # Create test samples
            test_dir = os.path.join(self.data_dir, 'test', skin_type)
            for i in tqdm(range(test_samples), desc=f"Test {skin_type}"):
                image = self.create_synthetic_skin_image(skin_type)
                image_path = os.path.join(test_dir, f'{skin_type}_test_{i:04d}.jpg')
                Image.fromarray(image).save(image_path, quality=90)
                total_created += 1
        
        # Save dataset info
        info_path = os.path.join(self.data_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✅ Dataset creation completed!")
        print(f"Total images created: {total_created}")
        print(f"Dataset info saved to: {info_path}")
        
        self._print_dataset_summary()
        
        return dataset_info
    
    def _print_dataset_summary(self):
        """Print a summary of the created dataset"""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        total_train = 0
        total_test = 0
        
        for skin_type in self.skin_types:
            train_dir = os.path.join(self.data_dir, 'train', skin_type)
            test_dir = os.path.join(self.data_dir, 'test', skin_type)
            
            train_count = len([f for f in os.listdir(train_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(train_dir) else 0
            test_count = len([f for f in os.listdir(test_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(test_dir) else 0
            
            print(f"{skin_type.capitalize():12} - Train: {train_count:4d}, Test: {test_count:3d}")
            total_train += train_count
            total_test += test_count
        
        print("-" * 60)
        print(f"{'Total':12} - Train: {total_train:4d}, Test: {total_test:3d}")
        print("="*60)


def main():
    """Create the dataset"""
    print("🔬 SKIN ANALYZER - DATASET CREATION")
    print("="*60)
    
    # Create dataset
    data_dir = 'data'
    creator = SimplifiedDatasetCreator(data_dir)
    
    # Create smaller dataset for testing (50 samples per type)
    dataset_info = creator.create_dataset(samples_per_type=50)
    
    print("\n🎉 Dataset creation completed successfully!")
    print("\nNext steps:")
    print("1. Install TensorFlow when available for Python 3.13")
    print("2. Train models: python models/train_model.py")
    print("3. Make predictions: python predict.py --image your_image.jpg")
    

if __name__ == "__main__":
    main()
