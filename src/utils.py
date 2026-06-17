"""
Utility functions and helpers for Lung Cancer Detection System
"""

import logging
import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> None:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        handlers=handlers
    )


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Get appropriate device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU device (slow)")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_checkpoint_dir(base_dir: str) -> Path:
    """Create timestamped checkpoint directory"""
    checkpoint_dir = Path(base_dir) / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}TB"


def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """
    Calculate medical evaluation metrics
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        tn: True negatives
    
    Returns:
        Dictionary with metrics
    """
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    
    # F1 score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Dice coefficient (for medical imaging)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'f1_score': f1,
        'dice_coefficient': dice
    }


def validate_image_path(image_path: str) -> bool:
    """Validate if image path exists and is readable"""
    path = Path(image_path)
    
    if not path.exists():
        logger.error(f"Image path does not exist: {image_path}")
        return False
    
    if not path.is_file():
        logger.error(f"Path is not a file: {image_path}")
        return False
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.dcm', '.nii', '.nii.gz'}
    if path.suffix.lower() not in supported_formats:
        logger.error(f"Unsupported image format: {path.suffix}")
        return False
    
    return True


def load_config_from_env(config_class):
    """Load configuration from environment variables"""
    env_vars = {}
    for key, value in os.environ.items():
        if key.startswith("LUNGCANCER_"):
            config_key = key.replace("LUNGCANCER_", "").lower()
            env_vars[config_key] = value
    
    return env_vars


class ProgressTracker:
    """Simple progress tracking utility"""
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.desc = desc
        self.current = 0
    
    def update(self, amount: int = 1) -> None:
        """Update progress"""
        self.current += amount
        percent = (self.current / self.total) * 100
        print(f"\r{self.desc}: {percent:.1f}% ({self.current}/{self.total})", end="")
    
    def finish(self) -> None:
        """Mark as finished"""
        print(f"\r{self.desc}: 100% ({self.total}/{self.total}) ✓")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide with default for zero division"""
    if denominator == 0:
        return default
    return numerator / denominator


if __name__ == "__main__":
    setup_logging(level="DEBUG")
    logger.info("Utility module loaded successfully")
    
    # Test utilities
    set_seed(42)
    device = get_device()
    print(f"Random seed set, device: {device}")
