"""
3D Volumetric Dataset and Loader for Lung Segmentation
Author: Senior AI Engineer & Medical Imaging Expert
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Union, Any
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


class Lung3DDataset(Dataset):
    """
    Custom Dataset for 3D Volumetric Lung CT Scan segmentation.
    Loads 3D volumes (NumPy .npy or simple raw shapes) and 3D binary masks,
    performs isotropic resampling, and returns PyTorch tensors.
    """

    def __init__(
        self,
        volume_paths: List[Union[str, Path]],
        mask_paths: List[Union[str, Path]],
        target_shape: Tuple[int, int, int] = (32, 128, 128),
        is_training: bool = False
    ):
        """
        Args:
            volume_paths: List of paths to 3D volume .npy files. Shape: (D, H, W)
            mask_paths: List of paths to corresponding 3D binary mask .npy files
            target_shape: Target 3D output dimension (Depth, Height, Width)
            is_training: If true, applies random spatial 3D augmentations
        """
        self.volume_paths = [Path(p) for p in volume_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.target_shape = target_shape
        self.is_training = is_training
        
        assert len(self.volume_paths) == len(self.mask_paths), "Volumes and masks mismatch."

    def __len__(self) -> int:
        return len(self.volume_paths)

    def _resize_volume_3d(self, array: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """
        Resample 3D array to target dimensions using interpolation.
        Uses trilinear for CT scan intensities, and nearest-neighbor for binary masks.
        """
        curr_shape = array.shape
        if curr_shape == self.target_shape:
            return array

        # Calculate scaling factors
        factors = [
            t / c for t, c in zip(self.target_shape, curr_shape)
        ]
        
        order = 0 if is_mask else 3  # Nearest for masks, cubic spline for images
        resized = zoom(array, factors, order=order, mode='constant', cval=0.0)
        return resized

    def _apply_augmentations_3d(self, volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply aligned 3D augmentations (flip/rotations)."""
        # Random axis flips
        for axis in [0, 1, 2]:
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis)
                mask = np.flip(mask, axis=axis)
                
        # Random 90 degree rotation in plane (last two dimensions)
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            volume = np.rot90(volume, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(1, 2))
            
        return volume.copy(), mask.copy()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_path = self.volume_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            # 1. Load 3D arrays
            volume = np.load(vol_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.float32)
            
            # 2. Resample to standard target shape
            volume = self._resize_volume_3d(volume, is_mask=False)
            mask = self._resize_volume_3d(mask, is_mask=True)
            
            # 3. Apply volumetric augmentations
            if self.is_training:
                volume, mask = self._apply_augmentations_3d(volume, mask)
                
            # 4. Standard windowing normalization to lung parenchyma [-1000 to 400 HU]
            # Clip and normalize
            volume = np.clip(volume, -1000.0, 400.0)
            volume = (volume - (-1000.0)) / (400.0 - (-1000.0))
            
            # Convert binary mask to [0.0, 1.0] float representation
            mask = (mask > 0.5).astype(np.float32)
            
            # Add channel dimension: (Channels, Depth, Height, Width)
            vol_tensor = torch.from_numpy(volume).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            
            return {
                "volume": vol_tensor,
                "mask": mask_tensor,
                "path": str(vol_path)
            }
        except Exception as e:
            logger.error(f"Failed to load volumetric sample at index {idx}: {e}")
            # Fallback to zero arrays
            d, h, w = self.target_shape
            return {
                "volume": torch.zeros((1, d, h, w), dtype=torch.float32),
                "mask": torch.zeros((1, d, h, w), dtype=torch.float32),
                "path": str(vol_path)
            }


def get_unet3d_dataloader(
    volume_paths: List[Union[str, Path]],
    mask_paths: List[Union[str, Path]],
    batch_size: int = 2,
    target_shape: Tuple[int, int, int] = (32, 128, 128),
    is_training: bool = False,
    shuffle: bool = True
) -> DataLoader:
    """Create a standard DataLoader for 3D volumes."""
    dataset = Lung3DDataset(
        volume_paths=volume_paths,
        mask_paths=mask_paths,
        target_shape=target_shape,
        is_training=is_training
    )
    
    # 3D loaders consume high VRAM. Keep batch_size small (1-4)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0, # Synchronous to avoid multiprocessing complexity with 3D arrays
        pin_memory=torch.cuda.is_available()
    )
