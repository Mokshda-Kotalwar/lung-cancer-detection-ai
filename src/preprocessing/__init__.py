"""
Data Preprocessing Module for Lung Cancer Detection
Handles DICOM processing, normalization, and image preparation
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
import logging
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


class DICOMProcessor:
    """Process DICOM medical images"""
    
    def __init__(self, window_center: int = 40, window_width: int = 400):
        """
        Initialize DICOM processor
        
        Args:
            window_center: Center of windowing (HU units)
            window_width: Width of windowing (HU units)
        """
        self.window_center = window_center
        self.window_width = window_width
        
        try:
            import pydicom
            self.pydicom = pydicom
        except ImportError:
            logger.warning("pydicom not installed. Install for DICOM support.")
            self.pydicom = None
    
    def load_dicom(self, filepath: str) -> Optional[np.ndarray]:
        """Load and convert DICOM to numpy array"""
        if self.pydicom is None:
            logger.error("pydicom required for DICOM loading")
            return None
        
        try:
            ds = self.pydicom.dcmread(filepath)
            image = ds.pixel_array.astype(np.float32)
            
            # Apply window/level transformation
            image = self.apply_window_level(image)
            return image
        except Exception as e:
            logger.error(f"Error loading DICOM {filepath}: {e}")
            return None
    
    def apply_window_level(self, image: np.ndarray) -> np.ndarray:
        """Apply window/level transformation (DICOM windowing)"""
        lower_bound = self.window_center - self.window_width / 2
        upper_bound = self.window_center + self.window_width / 2
        
        windowed = np.clip(image, lower_bound, upper_bound)
        windowed = ((windowed - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
        
        return windowed


class ImageNormalizer:
    """Normalize images using various methods"""
    
    @staticmethod
    def minmax_normalize(image: np.ndarray, min_val: float = 0, max_val: float = 1) -> np.ndarray:
        """Min-max normalization"""
        img_min = image.min()
        img_max = image.max()
        
        if img_max == img_min:
            return np.ones_like(image) * min_val
        
        normalized = (image - img_min) / (img_max - img_min)
        normalized = normalized * (max_val - min_val) + min_val
        return normalized.astype(np.float32)
    
    @staticmethod
    def zscore_normalize(image: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        mean = image.mean()
        std = image.std()
        
        if std == 0:
            return image.astype(np.float32)
        
        normalized = (image - mean) / std
        return normalized.astype(np.float32)
    
    @staticmethod
    def clahe_normalize(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(image)
        
        return enhanced.astype(np.float32) / 255.0


class LungSegmentation:
    """Lung segmentation utilities"""
    
    @staticmethod
    def segment_lungs(image: np.ndarray, threshold: int = -400) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple lung segmentation using threshold
        
        Args:
            image: Input image
            threshold: Threshold value for lung identification
        
        Returns:
            Segmented image and mask
        """
        # Simple threshold-based segmentation
        binary = image < threshold
        
        # Remove small components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Apply mask
        segmented = image * binary
        
        return segmented.astype(np.float32), binary.astype(np.float32)
    
    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply binary mask to image"""
        return (image * mask).astype(np.float32)


class ImageAugmentation:
    """Data augmentation for training"""
    
    def __init__(self, rotation_range: int = 20, brightness_range: List[float] = None):
        """Initialize augmentation"""
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range or [0.8, 1.2]
    
    def get_augmentation_transforms(self, target_size: int = 512):
        """Get PyTorch augmentation transforms"""
        return transforms.Compose([
            transforms.RandomRotation(self.rotation_range),
            transforms.ColorJitter(
                brightness=self.brightness_range[0],
                contrast=0.2,
                saturation=0.2
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image"""
        # Random rotation
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        image = np.clip(image * brightness_factor, 0, 1)
        
        return image.astype(np.float32)


class ImagePreprocessor:
    """Main image preprocessing pipeline"""
    
    def __init__(self, config):
        """Initialize preprocessor with configuration"""
        self.config = config
        self.dicom_processor = DICOMProcessor(
            window_center=config.preprocessing.window_center,
            window_width=config.preprocessing.window_width
        )
        self.normalizer = ImageNormalizer()
        self.lung_seg = LungSegmentation()
        self.augmentation = ImageAugmentation(
            rotation_range=config.preprocessing.rotation_range
        )
    
    def preprocess(self, image: np.ndarray, apply_aug: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image (numpy array or PIL Image)
            apply_aug: Whether to apply augmentation
        
        Returns:
            Preprocessed image
        """
        # Convert to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Resize
        if image.shape != (self.config.model.input_size, self.config.model.input_size):
            image = cv2.resize(image, (self.config.model.input_size, self.config.model.input_size))
        
        # Normalize
        if self.config.preprocessing.normalize_method == "minmax":
            image = self.normalizer.minmax_normalize(image)
        elif self.config.preprocessing.normalize_method == "zscore":
            image = self.normalizer.zscore_normalize(image)
        elif self.config.preprocessing.normalize_method == "clahe":
            image = self.normalizer.clahe_normalize(image)
        
        # Lung segmentation
        if self.config.preprocessing.use_lung_mask:
            image, mask = self.lung_seg.segment_lungs(image)
        
        # Augmentation
        if apply_aug and self.config.model.apply_augmentation:
            image = self.augmentation.augment_image(image)
        
        # Convert to torch tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.dim() == 2:
                image = image.unsqueeze(0)  # Add channel dimension
        
        return image
    
    def batch_preprocess(self, images: List[np.ndarray], apply_aug: bool = False) -> torch.Tensor:
        """Preprocess a batch of images"""
        processed = [self.preprocess(img, apply_aug) for img in images]
        return torch.stack(processed)


if __name__ == "__main__":
    from config import config
    preprocessor = ImagePreprocessor(config)
    print("Preprocessing module loaded successfully")
