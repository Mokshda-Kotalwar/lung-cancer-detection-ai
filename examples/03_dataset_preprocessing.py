"""
Example: Medical Image Preprocessing, Data Augmentation, and DataLoader Pipeline
Demonstrates loading, preprocessing with CLAHE/Denoising, Albumentations augmentations, and PyTorch dataloading.
"""

import sys
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional


# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, DATA_DIR, OUTPUTS_DIR
from src.preprocessing import MedicalImagePreprocessor, LungNoduleDataset, get_dataloader
from src.utils import setup_logging, set_seed

# Create output folder for visualizations
VIS_DIR = OUTPUTS_DIR / "preprocessing_visualizations"

VIS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_ct_scan(path: Path, has_nodule: bool = True) -> Tuple[np.ndarray, Optional[List[float]]]:
    """
    Generate synthetic CT-like slice (with lungs and a synthetic nodule) and save it.
    This enables running the example instantly without downloading LIDC-IDRI first.
    """
    # 1. Base grayscale slice (representing a chest CT)
    # Background is black (-1000 HU equivalent, standard scaled)
    size = 512
    slice_img = np.zeros((size, size), dtype=np.float32)
    
    # 2. Draw Chest wall boundary (oval, gray HU ~ -100 to 200)
    cv2.ellipse(slice_img, (256, 256), (200, 240), 0, 0, 360, 100, -1)
    
    # 3. Draw Lungs (two darker ellipses inside chest, HU ~ -700 to -800)
    cv2.ellipse(slice_img, (170, 240), (70, 120), 10, 0, 360, -750, -1)
    cv2.ellipse(slice_img, (342, 240), (70, 120), -10, 0, 360, -750, -1)
    
    # 4. Add quantum noise (quantum mottle)
    noise = np.random.normal(0, 45, (size, size)).astype(np.float32)
    slice_img += noise
    
    # 5. Add a nodule (dense sphere inside lung cavity, HU ~ 50)
    bbox = None
    if has_nodule:
        # Place nodule inside left lung
        nodule_center = (180, 220)
        radius = 12
        cv2.circle(slice_img, nodule_center, radius, 30, -1)
        
        # Calculate YOLO bbox format [x_center, y_center, width, height] normalized
        x_center = nodule_center[0] / size
        y_center = nodule_center[1] / size
        w = (radius * 2) / size
        h = (radius * 2) / size
        bbox = [x_center, y_center, w, h]

    # Save as .npy (NumPy file containing HU scale values)
    np.save(path, slice_img)
    return slice_img, bbox


def save_visualization(original: np.ndarray, preprocessed: torch.Tensor, filename: str):
    """Save visualization comparing original and preprocessed images."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image (grayscale)
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original HU CT Scan (Raw)")
    axes[0].axis('off')
    
    # Preprocessed and augmented channel (using first channel of PyTorch Tensor)
    prep_np = preprocessed[0].numpy()
    axes[1].imshow(prep_np, cmap='gray')
    axes[1].set_title("Enhanced + Augmented (PyTorch Tensor)")
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = VIS_DIR / filename
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to: {save_path}")


def main():
    setup_logging(level="INFO")
    set_seed(config.system.random_seed)
    
    print("\n" + "=" * 80)
    print("LUNG CANCER DETECTION - PREPROCESSING & DATA AUGMENTATION PIPELINE")
    print("=" * 80)
    
    # 1. Setup temporary directory for synthetic CT data
    temp_data_dir = DATA_DIR / "temp_example"
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/5] Generating synthetic CT datasets in {temp_data_dir}...")
    image_paths = []
    bboxes = []
    labels = []
    
    for i in range(10):
        img_path = temp_data_dir / f"slice_{i:03d}.npy"
        has_nodule = (i % 2 == 0) # half have nodules
        raw_img, bbox = generate_synthetic_ct_scan(img_path, has_nodule=has_nodule)
        
        image_paths.append(img_path)
        if has_nodule:
            bboxes.append(bbox)
            labels.append(1) # Malignant/Positive class
        else:
            bboxes.append([0.0, 0.0, 0.0, 0.0]) # Empty/Background bbox
            labels.append(0) # Benign/Normal class

    # 2. Initialize MedicalImagePreprocessor
    print("\n[2/5] Initializing MedicalImagePreprocessor...")
    # Temporarily set apply_augmentation config flags
    config.model.apply_augmentation = True
    config.preprocessing.normalize_method = "minmax"
    config.preprocessing.denoise = True
    
    preprocessor = MedicalImagePreprocessor(config, is_training=True)
    
    # 3. Process a single sample manually and visualize
    print("\n[3/5] Visualizing enhancement steps on a single CT slice...")
    raw_sample = np.load(image_paths[0])
    
    # Perform step-by-step visualizations
    windowed = preprocessor.apply_window_level(raw_sample)
    denoised = preprocessor.apply_noise_removal(windowed)
    clahe_enhanced = preprocessor.apply_contrast_enhancement(denoised)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(raw_sample, cmap='gray')
    plt.title("1. Raw (HU CT)")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(windowed, cmap='gray')
    plt.title("2. Windowed (-160 to 240)")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(denoised, cmap='gray')
    plt.title("3. Denoised (Bilateral)")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(clahe_enhanced, cmap='gray')
    plt.title("4. Contrast Enhanced (CLAHE)")
    plt.axis('off')
    
    step_vis_path = VIS_DIR / "enhancement_steps.png"
    plt.savefig(step_vis_path, dpi=150)
    plt.close()
    print(f"Saved step-by-step preprocessing stages to: {step_vis_path}")

    # 4. Create PyTorch Dataset & DataLoader
    print("\n[4/5] Loading datasets via PyTorch Dataset & DataLoader...")
    
    # Set model config parameters
    config.model.batch_size = 4
    config.model.num_workers = 0 # 0 for debugging/synchronous testing
    
    dataset = LungNoduleDataset(
        image_paths=image_paths,
        labels=labels,
        bboxes=bboxes,
        config=config,
        is_training=True
    )
    
    dataloader = get_dataloader(
        image_paths=image_paths,
        labels=labels,
        bboxes=bboxes,
        config=config,
        is_training=True,
        shuffle=True
    )
    
    # Fetch a single batch from dataloader
    batch = next(iter(dataloader))
    print(f"Dataloader batch loaded successfully!")
    print(f"  Images batch shape: {batch['images'].shape}")
    print(f"  Labels batch shape: {batch['labels'].shape}")
    if "bboxes" in batch:
        print(f"  Bounding Boxes count: {len(batch['bboxes'])}")
    
    # 5. Save visualizations of transformed/augmented images
    print("\n[5/5] Visualizing Albumentations augmentations...")
    for idx in range(min(4, len(dataset))):
        sample = dataset[idx]
        original_raw = np.load(image_paths[idx])
        original_windowed = preprocessor.apply_window_level(original_raw)
        
        save_visualization(
            original=original_windowed,
            preprocessed=sample["image"],
            filename=f"augmented_sample_{idx}.png"
        )
        
    print("\n[SUCCESS] Pipeline executed and visualized successfully!")
    print("=" * 80 + "\n")



if __name__ == "__main__":
    main()
