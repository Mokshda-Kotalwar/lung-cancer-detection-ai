"""
Medical Image Preprocessing & Dataset Pipeline for Lung Cancer Detection
Author: Senior AI Engineer & Medical Imaging Expert
"""

import os
import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union

logger = logging.getLogger(__name__)


def discover_medical_samples(
    data_root: Union[str, Path],
    supported_extensions: Optional[Tuple[str, ...]] = None,
) -> List[Dict[str, Any]]:
    """Discover CT/medical image samples from labeled folders, manifests, or Excel metadata."""
    root = Path(data_root)
    if not root.exists():
        return []

    if supported_extensions is None:
        supported_extensions = (".dcm", ".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

    if root.suffix.lower() in {".csv", ".xlsx", ".xls"}:
        try:
            import pandas as pd

            if root.suffix.lower() == ".csv":
                df = pd.read_csv(root)
            else:
                df = pd.read_excel(root)

            manifest: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                path_value = None
                for key in ["path", "image_path", "file_path", "image", "filepath"]:
                    if key in df.columns and pd.notna(row.get(key)):
                        path_value = row.get(key)
                        break

                if not path_value and "Series Instance UID" in df.columns:
                    path_value = row.get("Series Instance UID")

                if path_value is None:
                    continue

                label = row.get("label", row.get("class", row.get("Label", 2)))
                label_name = row.get("label_name", row.get("class_name", row.get("Label Name", "uncertain")))
                if isinstance(label_name, str):
                    label_name = label_name.lower()
                if label_name in {"benign", "healthy", "normal"}:
                    label = 0
                elif label_name in {"malignant", "cancer", "positive"}:
                    label = 1
                elif label_name in {"uncertain", "unknown", "ambiguous"}:
                    label = 2

                manifest.append({
                    "path": Path(str(path_value)),
                    "label": int(label),
                    "label_name": str(label_name),
                    "metadata": row.to_dict(),
                })
            return manifest
        except Exception as exc:
            logger.warning(f"Failed to read manifest file {root}: {exc}")
            return []

    label_mapping = {
        "benign": 0,
        "healthy": 0,
        "normal": 0,
        "malignant": 1,
        "cancer": 1,
        "positive": 1,
        "uncertain": 2,
        "unknown": 2,
        "ambiguous": 2,
    }

    manifest: List[Dict[str, Any]] = []
    if root.is_file():
        if root.suffix.lower() in supported_extensions:
            manifest.append({"path": root, "label": 2, "label_name": "uncertain"})
        return manifest

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in supported_extensions:
            continue

        label_name = None
        label_idx = None
        for parent in [file_path.parent] + list(file_path.parents):
            if parent == root or parent == parent.parent:
                break
            parent_name = parent.name.lower()
            if parent_name in label_mapping:
                label_name = parent_name
                label_idx = label_mapping[parent_name]
                break

        if label_name is None:
            continue

        manifest.append({
            "path": file_path,
            "label": label_idx,
            "label_name": label_name,
        })

    return manifest


def build_lidc_manifest(excel_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """Convert the LIDC-IDRI workbook metadata into a CSV manifest that the project can use."""
    import pandas as pd

    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Workbook not found: {excel_path}")

    df = pd.read_excel(excel_path)
    rows = []
    for _, row in df.iterrows():
        modality = str(row.get("Modality", "")).strip().upper()
        if modality != "CT":
            continue
        rows.append({
            "path": "",
            "label": 2,
            "label_name": "uncertain",
            "patient_id": row.get("Patient ID"),
            "series_uid": row.get("Series Instance UID"),
            "modality": modality,
            "study_uid": row.get("Study Instance UID"),
            "image_count": row.get("Image Count"),
            "series_description": row.get("Series Description"),
        })

    manifest_df = pd.DataFrame(rows)
    if output_path is None:
        output_path = excel_path.with_suffix(".csv")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False)
    logger.info(f"Wrote LIDC-IDRI manifest with {len(manifest_df)} CT entries to {output_path}")
    return output_path


# Try to import Albumentations and log if it fails
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning("Albumentations not installed. Please install it using `pip install albumentations`.")


class MedicalImagePreprocessor:
    """
    Advanced preprocessing pipeline for medical CT images.
    Handles DICOM windowing, noise removal, CLAHE, and Albumentations augmentations.
    """

    def __init__(self, config: Any, is_training: bool = False, has_bboxes: bool = False):
        """
        Initialize the preprocessor with central configurations.

        Args:
            config: Main Config instance
            is_training: Whether to construct the augmentation pipeline
            has_bboxes: Whether bounding boxes are provided
        """
        self.config = config
        self.is_training = is_training
        self.has_bboxes = has_bboxes
        
        # Load preprocessing specs
        self.target_size = getattr(config.model, 'input_size', 512)
        self.normalize_method = getattr(config.preprocessing, 'normalize_method', 'minmax')
        self.window_center = getattr(config.preprocessing, 'window_center', 40)
        self.window_width = getattr(config.preprocessing, 'window_width', 400)
        self.denoise = getattr(config.preprocessing, 'denoise', True)
        self.denoise_sigma = getattr(config.preprocessing, 'denoise_sigma', 1.0)
        self.use_lung_mask = getattr(config.preprocessing, 'use_lung_mask', False)
        
        # Setup Albumentations pipelines
        self.transform = self._build_transforms()

    def _build_transforms(self) -> Optional[Any]:
        """Build Albumentations transform pipelines."""
        if not ALBUMENTATIONS_AVAILABLE:
            return None

        transforms_list = []
        
        # Resize to standard input size
        transforms_list.append(A.Resize(self.target_size, self.target_size))

        # Add data augmentations for training
        if self.is_training and getattr(self.config.model, 'apply_augmentation', True):
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    A.GridDistortion(p=0.5),
                ], p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ])

        # Normalize and convert to PyTorch Tensor
        # Note: We divide by 255.0 manually or let ToTensorV2 handle it.
        # Since CT slices can be 16-bit or float32, we pass them as float32 to albumentations
        transforms_list.append(ToTensorV2())

        # Combine as an Albumentations Compose pipeline
        # Specify bounding box and mask parameters if needed
        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']) if (self.is_training and self.has_bboxes) else None
        )

    def apply_window_level(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Hounsfield Unit (HU) windowing to highlight soft tissues/nodules.
        
        Args:
            image: Raw CT slice in HU values (typically float32 or int16)
        Returns:
            8-bit windowed image [0-255]
        """
        try:
            lower_bound = self.window_center - (self.window_width / 2)
            upper_bound = self.window_center + (self.window_width / 2)
            
            # Clip HU values to specified range
            windowed = np.clip(image, lower_bound, upper_bound)
            
            # Normalize to 0-255 range
            windowed = ((windowed - lower_bound) / (upper_bound - lower_bound) * 255.0).astype(np.uint8)
            return windowed
        except Exception as e:
            logger.error(f"Error applying window/level: {e}")
            # Fallback min-max normalization to uint8
            img_min, img_max = image.min(), image.max()
            if img_max == img_min:
                return np.zeros_like(image, dtype=np.uint8)
            return ((image - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

    def apply_noise_removal(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Edge-preserving Bilateral Filter to remove quantum mottle.
        
        Args:
            image: 8-bit or float image
        Returns:
            Denoised image
        """
        try:
            if not self.denoise:
                return image
            
            # Bilateral filter preserves sharp edges (critical for nodules) while smoothing out noise
            if image.dtype == np.uint8:
                return cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
            else:
                # For floating-point images, standard Gaussian or fast bilateral
                return cv2.GaussianBlur(image, (3, 3), self.denoise_sigma)
        except Exception as e:
            logger.warning(f"Error during noise removal: {e}. Returning original.")
            return image

    def apply_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance local details using CLAHE.
        
        Args:
            image: uint8 grayscale image
        Returns:
            CLAHE-enhanced image
        """
        try:
            if image.dtype != np.uint8:
                image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255.0).astype(np.uint8)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        except Exception as e:
            logger.warning(f"CLAHE execution failed: {e}. Returning original.")
            return image

    def apply_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values according to config.
        
        Args:
            image: numpy array
        Returns:
            Normalized float32 numpy array
        """
        image_float = image.astype(np.float32)
        
        if self.normalize_method == "minmax":
            img_min, img_max = image_float.min(), image_float.max()
            if img_max == img_min:
                return np.zeros_like(image_float)
            return (image_float - img_min) / (img_max - img_min)
            
        elif self.normalize_method == "zscore":
            mean, std = image_float.mean(), image_float.std()
            if std == 0:
                return image_float
            return (image_float - mean) / std
            
        return image_float

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Execute core sequential image enhancement pipeline before augmentation.
        
        Args:
            image: raw input slice (HU values or standard image)
        Returns:
            Preprocessed 8-bit or normalized grayscale image
        """
        # If image is raw HU CT scan (values outside standard color ranges)
        if image.min() < -100 or image.max() > 300:
            image = self.apply_window_level(image)
        elif image.dtype != np.uint8:
            # Scale float images to uint8 if standard grayscale input
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255.0).astype(np.uint8)

        # Apply noise removal (Bilateral filter)
        image = self.apply_noise_removal(image)

        # Apply contrast enhancement (CLAHE)
        image = self.apply_contrast_enhancement(image)

        # Normalize pixel values
        image = self.apply_normalization(image)
        
        return image


class LungNoduleDataset(Dataset):
    """
    Highly robust, production-quality Dataset class for Lung Cancer Detection.
    Supports classification, object detection (bounding boxes), and segmentation (masks).
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        labels: Optional[List[Any]] = None,
        masks_paths: Optional[List[Union[str, Path]]] = None,
        bboxes: Optional[List[List[float]]] = None,
        config: Any = None,
        is_training: bool = False
    ):
        """
        Args:
            image_paths: List of paths to CT slices (DICOM, NPY, PNG, JPG)
            labels: List of classification label ids (0, 1, 2)
            masks_paths: List of paths to binary nodule masks (optional)
            bboxes: List of bounding boxes in format [x_center, y_center, w, h] normalized (optional)
            config: Main config instance
            is_training: If true, applies training augmentations
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.masks_paths = [Path(p) for p in masks_paths] if masks_paths else None
        self.bboxes = bboxes
        self.is_training = is_training
        
        # Instantiate the preprocessor
        self.preprocessor = MedicalImagePreprocessor(config, is_training=is_training, has_bboxes=(bboxes is not None))

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image safely depending on extension."""
        if not path.exists():
            raise FileNotFoundError(f"Medical scan file not found: {path}")

        ext = path.suffix.lower()
        
        try:
            if ext == '.dcm':
                # Try pydicom
                import pydicom
                ds = pydicom.dcmread(path)
                # Rescale slope and intercept if present to get true HU
                image = ds.pixel_array.astype(np.float32)
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    image = image * ds.RescaleSlope + ds.RescaleIntercept
                return image
            elif ext == '.npy':
                return np.load(path).astype(np.float32)
            elif ext in ['.png', '.jpg', '.jpeg']:
                # Read grayscale image
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise IOError(f"Unable to read image file: {path}")
                return img.astype(np.float32)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            # Return dummy zero slice to prevent crash during training loop
            target_size = self.preprocessor.target_size
            return np.zeros((target_size, target_size), dtype=np.float32)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load segmentation mask safely."""
        try:
            if not path.exists():
                raise FileNotFoundError(f"Mask file not found: {path}")
            
            if path.suffix.lower() == '.npy':
                mask = np.load(path)
            else:
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise IOError(f"Unable to read mask file: {path}")
            
            # Binarize mask (0 or 1)
            return (mask > 0).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error loading mask {path}: {e}")
            target_size = self.preprocessor.target_size
            return np.zeros((target_size, target_size), dtype=np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches a single sample from the dataset.
        Applies preprocessing and augmentations to image, bounding boxes, and masks simultaneously.
        """
        img_path = self.image_paths[idx]
        
        # 1. Load raw image/CT scan
        image = self._load_image(img_path)
        
        # 2. Run sequential enhancement (denoise, window, clahe)
        enhanced_image = self.preprocessor.preprocess_image(image)
        
        # Standardize grayscale image to be 3-channel for backbones like EfficientNet,
        # or keep 1-channel if desired. Timms models typically expect 3 channels.
        # We duplicate the grayscale channel to RGB
        enhanced_rgb = cv2.cvtColor(enhanced_rgb_src := (enhanced_image * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # Scale back to [0, 1] range for albumentations input
        enhanced_rgb = enhanced_rgb.astype(np.float32) / 255.0

        # Construct payload for albumentations
        payload = {"image": enhanced_rgb}
        
        # 3. Load associated targets
        # Segmentation Mask
        mask_present = self.masks_paths is not None and idx < len(self.masks_paths)
        if mask_present:
            mask = self._load_mask(self.masks_paths[idx])
            # Resize mask to input size before Albumentations to match dimensions
            if mask.shape != (self.preprocessor.target_size, self.preprocessor.target_size):
                mask = cv2.resize(mask, (self.preprocessor.target_size, self.preprocessor.target_size), interpolation=cv2.INTER_NEAREST)
            payload["mask"] = mask
            
        # Bounding Boxes (for Detection)
        boxes_present = self.bboxes is not None and idx < len(self.bboxes)
        if boxes_present and self.is_training:
            # Albumentations expects bounding boxes and list of their label ids
            payload["bboxes"] = [self.bboxes[idx]]
            payload["class_labels"] = [self.labels[idx] if self.labels else 0]

        # 4. Apply Albumentations (maintains alignment between image, mask, and bboxes)
        if self.preprocessor.transform:
            try:
                transformed = self.preprocessor.transform(**payload)
                transformed_image = transformed["image"]
                
                transformed_mask = transformed.get("mask", None)
                transformed_bboxes = transformed.get("bboxes", None)
            except Exception as e:
                logger.error(f"Albumentations transformation failed at index {idx}: {e}")
                # Fallback to simple tensor conversion
                transformed_image = torch.from_numpy(enhanced_rgb).permute(2, 0, 1).float()
                transformed_mask = torch.from_numpy(payload["mask"]).long() if "mask" in payload else None
                transformed_bboxes = payload.get("bboxes", None)
        else:
            # Fallback if albumentations is unavailable
            transformed_image = torch.from_numpy(enhanced_rgb).permute(2, 0, 1).float()
            transformed_mask = torch.from_numpy(payload["mask"]).long() if "mask" in payload else None
            transformed_bboxes = payload.get("bboxes", None)

        # Output preparation
        sample = {
            "image": transformed_image,
            "image_path": str(img_path)
        }
        
        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        if transformed_mask is not None:
            # Convert mask to torch tensor
            if not isinstance(transformed_mask, torch.Tensor):
                transformed_mask = torch.from_numpy(transformed_mask).long()
            sample["mask"] = transformed_mask
            
        if transformed_bboxes is not None:
            sample["bboxes"] = torch.tensor(transformed_bboxes, dtype=torch.float32)
            
        return sample


def collate_detection_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for object detection batching.
    Handles variable numbers of bounding boxes per image.
    """
    images = torch.stack([item["image"] for item in batch])
    paths = [item["image_path"] for item in batch]
    
    collated = {
        "images": images,
        "image_paths": paths
    }
    
    if "label" in batch[0]:
        collated["labels"] = torch.stack([item["label"] for item in batch])
        
    if "mask" in batch[0]:
        collated["masks"] = torch.stack([item["mask"] for item in batch])
        
    if "bboxes" in batch[0]:
        # Keep boxes as list of tensors since box counts can vary
        collated["bboxes"] = [item["bboxes"] for item in batch]
        
    return collated


def get_dataloader(
    image_paths: List[Union[str, Path]],
    labels: Optional[List[int]] = None,
    masks_paths: Optional[List[Union[str, Path]]] = None,
    bboxes: Optional[List[List[float]]] = None,
    config: Any = None,
    is_training: bool = False,
    shuffle: bool = True
) -> DataLoader:
    """
    Factory function to initialize a clean PyTorch DataLoader.
    """
    dataset = LungNoduleDataset(
        image_paths=image_paths,
        labels=labels,
        masks_paths=masks_paths,
        bboxes=bboxes,
        config=config,
        is_training=is_training
    )
    
    batch_size = getattr(config.model, 'batch_size', 16)
    num_workers = getattr(config.model, 'num_workers', 4)
    
    # Set persistent_workers if num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_fn,
        persistent_workers=(num_workers > 0)
    )
    return loader
