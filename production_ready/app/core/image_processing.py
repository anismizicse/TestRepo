"""
Image Processing Utilities
===========================

Image preprocessing and validation utilities.
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional, Tuple

def validate_image(image_data: bytes) -> bool:
    """Validate if the uploaded data is a valid image."""
    try:
        Image.open(io.BytesIO(image_data))
        return True
    except Exception:
        return False

def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """Decode base64 image string to OpenCV format."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
        
    except Exception as e:
        return None

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for analysis."""
    try:
        # Resize image while maintaining aspect ratio
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = target_size[0], int(w * target_size[0] / h)
        else:
            new_h, new_w = int(h * target_size[1] / w), target_size[1]
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create a square image with padding
        result = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        y_offset = (target_size[0] - new_h) // 2
        x_offset = (target_size[1] - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Apply noise reduction
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result
        
    except Exception:
        # If preprocessing fails, just resize
        return cv2.resize(image, target_size)

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better analysis."""
    try:
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
        
    except Exception:
        return image
