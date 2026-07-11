"""
DenseNet121 Transfer Learning Classifier for Lung Cancer Detection
Author: Senior AI Engineer & Medical Imaging Expert
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class DenseNetClassifier(nn.Module):
    """
    DenseNet121 classification model for lung cancer classification.
    Uses transfer learning with a custom classification head.
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize DenseNet121 classifier.
        
        Args:
            num_classes: Number of target classes (default: 3 - Benign, Malignant, Uncertain)
            pretrained: Whether to load ImageNet pre-trained weights
            freeze_backbone: Whether to freeze backbone layers initially
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Load DenseNet121 backbone
        if pretrained:
            try:
                # Modern torchvision API (>= 0.13)
                weights = models.DenseNet121_Weights.DEFAULT
                self.backbone = models.densenet121(weights=weights)
                logger.info("Loaded pretrained DenseNet121 backbone using DenseNet121_Weights.DEFAULT")
            except Exception as e:
                # Legacy torchvision API or fallback
                try:
                    self.backbone = models.densenet121(pretrained=True)
                    logger.info("Loaded pretrained DenseNet121 backbone using legacy pretrained=True")
                except Exception as e2:
                    logger.warning(f"Failed to load pretrained weights ({e2}), initializing random weights: {e}")
                    self.backbone = models.densenet121(weights=None)
        else:
            self.backbone = models.densenet121(weights=None)
            logger.info("Initialized DenseNet121 backbone with random weights")
            
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Froze all backbone parameters for transfer learning")
            
        # Extract features input size for classifier head modification
        in_features = self.backbone.classifier.in_features
        
        # Custom head: Linear -> BatchNorm -> ReLU -> Dropout -> Linear
        # This provides a robust and regularized head for medical classification task
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        logger.info(f"Custom classifier head initialized for {num_classes} classes")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier."""
        return self.backbone(x)
        
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfroze backbone parameters for fine-tuning")

    def freeze_backbone(self):
        """Freeze all backbone parameters, keeping only the classification head active."""
        # Freeze everything in backbone features
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        logger.info("Froze backbone feature extraction layers")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = DenseNetClassifier(num_classes=3)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 3), "Output shape mismatch!"
    print("DenseNetClassifier verification successful!")
