"""
Image Preprocessing Utilities for Skin Type Analysis
Handles face detection, cropping, and image normalization
"""

import cv2
import numpy as np
from PIL import Image

# Optional TensorFlow import
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class ImageProcessor:
    """
    Handles all image preprocessing tasks for skin type analysis
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the image processor
        
        Args:
            target_size (tuple): Target size for processed images (width, height)
        """
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_face(self, image):
        """
        Detect faces in the image using Haar cascades
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (face_detected, face_coords) where face_coords is (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Return the largest face detected
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            return True, largest_face
        return False, None
    
    def crop_face(self, image, face_coords, margin=0.2):
        """
        Crop the face region with some margin
        
        Args:
            image (numpy.ndarray): Input image
            face_coords (tuple): Face coordinates (x, y, w, h)
            margin (float): Margin around face (0.2 = 20% margin)
            
        Returns:
            numpy.ndarray: Cropped face image
        """
        x, y, w, h = face_coords
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Calculate new coordinates with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        return image[y1:y2, x1:x2]
    
    def normalize_image(self, image):
        """
        Normalize image pixel values to [0, 1] range
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def resize_image(self, image, target_size=None):
        """
        Resize image to target size
        
        Args:
            image (numpy.ndarray): Input image
            target_size (tuple): Target size (width, height)
            
        Returns:
            numpy.ndarray: Resized image
        """
        if target_size is None:
            target_size = self.target_size
            
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def augment_image(self, image, augment_type='random'):
        """
        Apply data augmentation to the image
        
        Args:
            image (numpy.ndarray): Input image
            augment_type (str): Type of augmentation
            
        Returns:
            numpy.ndarray: Augmented image
        """
        if augment_type == 'random':
            # Random selection of augmentations
            augmentations = [
                self._rotate_image,
                self._flip_horizontal,
                self._adjust_brightness,
                self._adjust_contrast
            ]
            
            # Apply 1-2 random augmentations
            num_augmentations = np.random.randint(1, 3)
            selected_augmentations = np.random.choice(
                augmentations, 
                size=num_augmentations, 
                replace=False
            )
            
            for aug_func in selected_augmentations:
                image = aug_func(image)
                
        return image
    
    def _rotate_image(self, image, max_angle=15):
        """Rotate image by a random angle"""
        angle = np.random.uniform(-max_angle, max_angle)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    def _flip_horizontal(self, image):
        """Flip image horizontally with 50% probability"""
        if np.random.random() > 0.5:
            return cv2.flip(image, 1)
        return image
    
    def _adjust_brightness(self, image, max_delta=0.2):
        """Adjust image brightness"""
        delta = np.random.uniform(-max_delta, max_delta)
        return np.clip(image + delta * 255, 0, 255).astype(np.uint8)
    
    def _adjust_contrast(self, image, contrast_range=(0.8, 1.2)):
        """Adjust image contrast"""
        contrast = np.random.uniform(*contrast_range)
        return np.clip(image * contrast, 0, 255).astype(np.uint8)
    
    def process_image(self, image_path, augment=False):
        """
        Complete image processing pipeline
        
        Args:
            image_path (str): Path to the image file
            augment (bool): Whether to apply data augmentation
            
        Returns:
            numpy.ndarray: Processed image ready for model input
            bool: Whether face was detected successfully
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    return None, False
            else:
                image = image_path
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face
            face_detected, face_coords = self.detect_face(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            
            if face_detected:
                # Crop face
                face_image = self.crop_face(image, face_coords)
            else:
                # Use center crop if no face detected
                h, w = image.shape[:2]
                size = min(h, w)
                start_h = (h - size) // 2
                start_w = (w - size) // 2
                face_image = image[start_h:start_h + size, start_w:start_w + size]
            
            # Resize image
            processed_image = self.resize_image(face_image)
            
            # Apply augmentation if requested
            if augment:
                processed_image = self.augment_image(processed_image)
            
            # Normalize image
            processed_image = self.normalize_image(processed_image)
            
            return processed_image, face_detected
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, False
    
    def preprocess_batch(self, image_paths, augment=False):
        """
        Process a batch of images
        
        Args:
            image_paths (list): List of image paths
            augment (bool): Whether to apply data augmentation
            
        Returns:
            numpy.ndarray: Batch of processed images
            list: List of face detection results
        """
        processed_images = []
        face_detection_results = []
        
        for image_path in image_paths:
            processed_image, face_detected = self.process_image(image_path, augment)
            if processed_image is not None:
                processed_images.append(processed_image)
                face_detection_results.append(face_detected)
            else:
                print(f"Failed to process image: {image_path}")
        
        if processed_images:
            return np.array(processed_images), face_detection_results
        else:
            return np.array([]), []
