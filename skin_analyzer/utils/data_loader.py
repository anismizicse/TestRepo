"""
Data Loading Utilities for Skin Type Analysis
Handles dataset loading, splitting, and batch generation
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from utils.image_processor import ImageProcessor
import json


class SkinDataLoader:
    """
    Handles loading and preparing skin type datasets
    """
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader
        
        Args:
            data_dir (str): Path to the data directory
            img_size (tuple): Target image size
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.image_processor = ImageProcessor(target_size=img_size)
        self.label_encoder = LabelEncoder()
        
        # Skin type classes
        self.skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        self.num_classes = len(self.skin_types)
        
    def create_dataset_structure(self):
        """
        Create the required dataset directory structure
        """
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        for skin_type in self.skin_types:
            train_type_dir = os.path.join(train_dir, skin_type)
            test_type_dir = os.path.join(test_dir, skin_type)
            
            os.makedirs(train_type_dir, exist_ok=True)
            os.makedirs(test_type_dir, exist_ok=True)
            
        print(f"Dataset structure created in {self.data_dir}")
        print("Please organize your images into the following directories:")
        for skin_type in self.skin_types:
            print(f"  - {os.path.join(train_dir, skin_type)}")
            print(f"  - {os.path.join(test_dir, skin_type)}")
    
    def load_dataset_from_directory(self, split='train', validation_split=0.2):
        """
        Load dataset from directory structure
        
        Args:
            split (str): 'train' or 'test'
            validation_split (float): Fraction of training data to use for validation
            
        Returns:
            tuple: (images, labels) or (train_images, train_labels, val_images, val_labels)
        """
        data_split_dir = os.path.join(self.data_dir, split)
        
        if not os.path.exists(data_split_dir):
            raise ValueError(f"Data directory {data_split_dir} does not exist")
        
        images = []
        labels = []
        file_paths = []
        
        # Load images from each skin type directory
        for skin_type in self.skin_types:
            skin_type_dir = os.path.join(data_split_dir, skin_type)
            
            if not os.path.exists(skin_type_dir):
                print(f"Warning: Directory {skin_type_dir} does not exist")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(skin_type_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            print(f"Found {len(image_files)} images for {skin_type} skin type")
            
            for image_file in image_files:
                image_path = os.path.join(skin_type_dir, image_file)
                file_paths.append(image_path)
                labels.append(skin_type)
        
        if not file_paths:
            raise ValueError(f"No images found in {data_split_dir}")
        
        # Process images
        print(f"Processing {len(file_paths)} images...")
        processed_images = []
        valid_labels = []
        
        for i, (image_path, label) in enumerate(zip(file_paths, labels)):
            if i % 50 == 0:
                print(f"Processed {i}/{len(file_paths)} images")
            
            processed_image, face_detected = self.image_processor.process_image(
                image_path, 
                augment=(split == 'train')
            )
            
            if processed_image is not None:
                processed_images.append(processed_image)
                valid_labels.append(label)
            else:
                print(f"Failed to process image: {image_path}")
        
        if not processed_images:
            raise ValueError("No images could be processed successfully")
        
        # Convert to numpy arrays
        images = np.array(processed_images)
        labels = np.array(valid_labels)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        categorical_labels = to_categorical(encoded_labels, num_classes=self.num_classes)
        
        print(f"Successfully loaded {len(images)} images")
        print(f"Image shape: {images.shape}")
        print(f"Label distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for skin_type, count in zip(unique, counts):
            print(f"  {skin_type}: {count}")
        
        if split == 'train' and validation_split > 0:
            # Split training data into train and validation
            train_images, val_images, train_labels, val_labels = train_test_split(
                images, categorical_labels, 
                test_size=validation_split, 
                random_state=42, 
                stratify=encoded_labels
            )
            return train_images, train_labels, val_images, val_labels
        else:
            return images, categorical_labels
    
    def create_tf_dataset(self, images, labels, shuffle=True, augment=False):
        """
        Create TensorFlow dataset from images and labels
        
        Args:
            images (numpy.ndarray): Array of images
            labels (numpy.ndarray): Array of labels
            shuffle (bool): Whether to shuffle the dataset
            augment (bool): Whether to apply data augmentation
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        
        if augment:
            dataset = dataset.map(
                self._augment_tf_image, 
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_tf_image(self, image, label):
        """
        Apply TensorFlow-based data augmentation
        
        Args:
            image (tf.Tensor): Image tensor
            label (tf.Tensor): Label tensor
            
        Returns:
            tuple: (augmented_image, label)
        """
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Random saturation
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        
        # Ensure values are in [0, 1] range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def generate_synthetic_data(self, num_samples_per_class=100):
        """
        Generate synthetic data for testing purposes
        
        Args:
            num_samples_per_class (int): Number of samples to generate per class
        """
        print("Generating synthetic data for testing...")
        
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        # Create directories
        self.create_dataset_structure()
        
        for skin_type in self.skin_types:
            # Generate training samples
            train_type_dir = os.path.join(train_dir, skin_type)
            for i in range(num_samples_per_class):
                # Generate random image (224x224x3)
                synthetic_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                
                # Add some patterns based on skin type
                if skin_type == 'oily':
                    # Add shine effect
                    synthetic_image[:, :, 1] = np.clip(synthetic_image[:, :, 1] + 30, 0, 255)
                elif skin_type == 'dry':
                    # Reduce brightness
                    synthetic_image = np.clip(synthetic_image - 20, 0, 255)
                elif skin_type == 'sensitive':
                    # Add redness
                    synthetic_image[:, :, 0] = np.clip(synthetic_image[:, :, 0] + 25, 0, 255)
                
                # Save image
                image_path = os.path.join(train_type_dir, f'synthetic_{i:04d}.png')
                from PIL import Image
                Image.fromarray(synthetic_image).save(image_path)
            
            # Generate test samples (fewer)
            test_type_dir = os.path.join(test_dir, skin_type)
            for i in range(num_samples_per_class // 4):
                synthetic_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                
                # Apply same patterns as training
                if skin_type == 'oily':
                    synthetic_image[:, :, 1] = np.clip(synthetic_image[:, :, 1] + 30, 0, 255)
                elif skin_type == 'dry':
                    synthetic_image = np.clip(synthetic_image - 20, 0, 255)
                elif skin_type == 'sensitive':
                    synthetic_image[:, :, 0] = np.clip(synthetic_image[:, :, 0] + 25, 0, 255)
                
                image_path = os.path.join(test_type_dir, f'synthetic_{i:04d}.png')
                from PIL import Image
                Image.fromarray(synthetic_image).save(image_path)
        
        print(f"Generated {num_samples_per_class} training and {num_samples_per_class // 4} test samples per class")
        print("Synthetic data generation completed!")
    
    def save_label_encoder(self, filepath):
        """
        Save the label encoder for later use
        
        Args:
            filepath (str): Path to save the label encoder
        """
        import joblib
        joblib.dump(self.label_encoder, filepath)
        
        # Also save class names
        class_names = {i: name for i, name in enumerate(self.skin_types)}
        with open(filepath.replace('.pkl', '_classes.json'), 'w') as f:
            json.dump(class_names, f)
    
    def load_label_encoder(self, filepath):
        """
        Load a previously saved label encoder
        
        Args:
            filepath (str): Path to the saved label encoder
        """
        import joblib
        self.label_encoder = joblib.load(filepath)
        
        # Load class names
        with open(filepath.replace('.pkl', '_classes.json'), 'r') as f:
            class_names = json.load(f)
        
        self.skin_types = [class_names[str(i)] for i in range(len(class_names))]
        self.num_classes = len(self.skin_types)
