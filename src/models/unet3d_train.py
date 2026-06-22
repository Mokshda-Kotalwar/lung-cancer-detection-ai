"""
3D U-Net Segmentation Training & Evaluation Module
Author: Senior AI Engineer & Medical Imaging Expert
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import config, DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from src.models.unet3d import UNet3D
from src.preprocessing.unet3d_dataset import get_unet3d_dataloader

logger = logging.getLogger(__name__)


class DiceBCELoss(nn.Module):
    """Combined Binary Cross Entropy and Dice Loss for segmentation."""
    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        
        # Flatten inputs and targets
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (inputs_flat.sum() + targets_flat.sum() + 1e-6)
        
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


def calculate_segmentation_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate Dice Coefficient and Intersection over Union (IoU) on binary inputs.
    
    Args:
        preds: Sigmoid model output tensor (0.0 - 1.0)
        targets: Target binary tensor
    Returns:
        dice: Dice Score
        iou: IoU Score
    """
    # Threshold predictions
    preds_bin = (preds > 0.5).float()
    
    preds_flat = preds_bin.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum().item()
    union = preds_flat.sum().item() + targets_flat.sum().item() - intersection
    total_pixels = preds_flat.sum().item() + targets_flat.sum().item()
    
    dice = (2.0 * intersection + 1e-6) / (total_pixels + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return dice, iou


def visualize_3d_slice(volume: torch.Tensor, target_mask: torch.Tensor, 
                       pred_mask: torch.Tensor, epoch: int, idx: int):
    """
    Saves axial slice comparisons (original, true mask, predicted mask) 
    from the middle of the 3D volume.
    """
    # Exclude batch/channel dimensions
    vol_np = volume.cpu().squeeze().numpy() # Shape: (D, H, W)
    target_np = target_mask.cpu().squeeze().numpy()
    pred_np = pred_mask.cpu().squeeze().numpy()
    
    # Grab the middle axial slice
    mid_idx = vol_np.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#1e1e1e')
    
    # Slice raw
    axes[0].imshow(vol_np[mid_idx], cmap='gray')
    axes[0].set_title("CT Axial Middle Slice", color='white', fontsize=12)
    axes[0].axis('off')
    
    # Ground Truth Mask overlay
    axes[1].imshow(vol_np[mid_idx], cmap='gray')
    axes[1].imshow(target_np[mid_idx], cmap='jet', alpha=0.3)
    axes[1].set_title("Ground Truth Mask", color='white', fontsize=12)
    axes[1].axis('off')
    
    # Prediction overlay
    axes[2].imshow(vol_np[mid_idx], cmap='gray')
    axes[2].imshow(pred_np[mid_idx] > 0.5, cmap='jet', alpha=0.3)
    axes[2].set_title(f"3D U-Net Prediction", color='white', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    out_dir = OUTPUTS_DIR / "unet3d_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = out_dir / f"epoch_{epoch}_sample_{idx}.png"
    plt.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    logger.info(f"Saved axial slice prediction output visualization: {save_path}")


def train_unet3d(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cuda"
) -> nn.Module:
    """Train 3D U-Net model and save checkpoints."""
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_dice = 0.0
    
    checkpoint_dir = MODELS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("STARTING 3D U-NET LUNG SEGMENTATION MODEL TRAINING")
    print("=" * 80)
    
    for epoch in range(1, epochs + 1):
        # 1. Training Loop
        model.train()
        train_loss = 0.0
        train_dices, train_ious = [], []
        
        for batch_idx, batch in enumerate(train_loader):
            volumes = batch["volume"].to(device)
            masks = batch["mask"].to(device)
            
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            dice, iou = calculate_segmentation_metrics(outputs, masks)
            train_dices.append(dice)
            train_ious.append(iou)
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = np.mean(train_dices)
        avg_train_iou = np.mean(train_ious)
        
        # 2. Validation Loop
        model.eval()
        val_loss = 0.0
        val_dices, val_ious = [], []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                volumes = batch["volume"].to(device)
                masks = batch["mask"].to(device)
                
                outputs = model(volumes)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                dice, iou = calculate_segmentation_metrics(outputs, masks)
                val_dices.append(dice)
                val_ious.append(iou)
                
                # Visualize the first batch's first sample
                if batch_idx == 0:
                    visualize_3d_slice(volumes[0], masks[0], outputs[0], epoch, batch_idx)
                    
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = np.mean(val_dices)
        avg_val_iou = np.mean(val_ious)
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.3f} | Train IoU: {avg_train_iou:.3f} || "
              f"Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.3f} | Val IoU: {avg_val_iou:.3f}")
        
        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            best_model_path = checkpoint_dir / "best_unet3d.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best 3D U-Net checkpoint with Dice {best_dice:.3f}: {best_model_path}")
            
    print("\n[SUCCESS] 3D U-Net Training Complete!")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 1. Setup local synthetic volumetric dataset
    temp_3d_dir = DATA_DIR / "temp_3d"
    temp_3d_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating synthetic 3D volumes for verification...")
    vol_paths = []
    mask_paths = []
    
    # Draw simple spherical lung structures inside 3D array
    for i in range(4):
        vol_path = temp_3d_dir / f"volume_{i}.npy"
        mask_path = temp_3d_dir / f"mask_{i}.npy"
        
        # 3D Shape: (Depth=16, Height=64, Width=64)
        vol_arr = np.random.normal(-1000, 20, (16, 64, 64)).astype(np.float32)
        mask_arr = np.zeros((16, 64, 64), dtype=np.float32)
        
        # Place synthetic circular lungs
        # z-center, y-center, x-center
        zc, yc, xc = 8, 32, 32
        radius = 16
        
        # Create coordinates grid
        z, y, x = np.ogrid[:16, :64, :64]
        dist_sq = (z - zc)**2 + (y - yc)**2 + (x - xc)**2
        
        # Fill lung area with -750 HU, mask with 1.0
        lung_pixels = dist_sq <= radius**2
        vol_arr[lung_pixels] = -750.0
        mask_arr[lung_pixels] = 1.0
        
        np.save(vol_path, vol_arr)
        np.save(mask_path, mask_arr)
        vol_paths.append(vol_path)
        mask_paths.append(mask_path)

    # 2. Build 3D Dataloaders
    # target_shape: (Depth=16, Height=64, Width=64)
    train_loader = get_unet3d_dataloader(
        volume_paths=vol_paths[:2],
        mask_paths=mask_paths[:2],
        batch_size=1,
        target_shape=(16, 64, 64),
        is_training=True
    )
    val_loader = get_unet3d_dataloader(
        volume_paths=vol_paths[2:],
        mask_paths=mask_paths[2:],
        batch_size=1,
        target_shape=(16, 64, 64),
        is_training=False
    )
    
    # 3. Instantiate and train model (small epoch count for testing verification)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Compact model size for fast verification
    model = UNet3D(in_channels=1, out_channels=1, features=[8, 16, 32, 64])
    
    train_unet3d(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=1e-3,
        device=device
    )
