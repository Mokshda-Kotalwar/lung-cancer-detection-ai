"""
Explainability Module - Grad-CAM Visualization
Provides interpretable AI explanations for lung cancer detection
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM)"""
    
    def __init__(self, model: torch.nn.Module, target_layer: str, device: str = "cuda"):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Name of layer to visualize (e.g., 'layer4', 'features.16')
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.target_layer_name = target_layer
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            # Save activations (detached clone)
            self.activations = output.detach().clone()
            
            # Register a hook on the activation tensor directly to capture gradients
            if output.requires_grad:
                def tensor_backward_hook(grad):
                    self.gradients = grad.detach().clone()
                    
                output.register_hook(tensor_backward_hook)
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name or self.target_layer_name in name:
                self.target_layer = module
                module.register_forward_hook(forward_hook)
                logger.info(f"Registered hooks for layer: {name}")
                break
        
        if self.target_layer is None:
            logger.warning(f"Target layer '{self.target_layer_name}' not found")
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM visualization
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class for visualization. If None, uses predicted class
        
        Returns:
            Grad-CAM heatmap (H, W)
        """
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            
            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=1)[0]
            
            # Backward pass
            target_score = output[0, target_class]
            self.model.zero_grad()
            target_score.backward()
        
        # Calculate Grad-CAM
        if self.gradients is None or self.activations is None:
            logger.error("Failed to capture gradients or activations")
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        
        # Compute weights
        weights = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations[0]).sum(dim=0)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input shape
        h, w = input_tensor.shape[-2:]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return cam
    
    def visualize(self, input_tensor: torch.Tensor, original_image: np.ndarray,
                  target_class: Optional[int] = None, colormap: str = "jet",
                  alpha: float = 0.4) -> np.ndarray:
        """
        Generate visualization overlay
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            original_image: Original image for overlay
            target_class: Target class
            colormap: Matplotlib colormap name
            alpha: Transparency of overlay
        
        Returns:
            Overlay image
        """
        # Generate Grad-CAM
        cam = self.generate(input_tensor, target_class)
        
        # Convert CAM to color
        cam_color = self._apply_colormap(cam, colormap)
        
        # Resize to match original image
        if original_image.shape[:2] != cam_color.shape[:2]:
            cam_color = cv2.resize(cam_color, 
                                   (original_image.shape[1], original_image.shape[0]))
        
        # Create overlay
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        overlay = cv2.addWeighted(original_image.astype(np.uint8), 1 - alpha,
                                  cam_color.astype(np.uint8), alpha, 0)
        
        return overlay
    
    @staticmethod
    def _apply_colormap(heatmap: np.ndarray, colormap: str = "jet") -> np.ndarray:
        """Apply colormap to heatmap"""
        # Normalize to 0-255
        heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply OpenCV colormap
        colormap_map = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "cool": cv2.COLORMAP_COOL,
            "spring": cv2.COLORMAP_SPRING,
            "summer": cv2.COLORMAP_SUMMER,
            "autumn": cv2.COLORMAP_AUTUMN,
            "winter": cv2.COLORMAP_WINTER,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "plasma": cv2.COLORMAP_PLASMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
        }
        
        cmap = colormap_map.get(colormap, cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(heatmap, cmap)
        
        return colored


class LayerCAM:
    """Layer-wise Relevance Propagation variant"""
    
    def __init__(self, model: torch.nn.Module, target_layer: str, device: str = "cuda"):
        """Initialize LayerCAM"""
        self.model = model
        self.device = device
        self.target_layer_name = target_layer
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        for name, module in self.model.named_modules():
            if name == self.target_layer_name or self.target_layer_name in name:
                module.register_forward_hook(forward_hook)
                break
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate LayerCAM visualization"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1)[0]
        
        if self.activations is None:
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        
        # Simple averaging of activations
        cam = self.activations[0].mean(dim=0)
        cam = F.relu(cam)
        
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        h, w = input_tensor.shape[-2:]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return cam


class ExplainabilityEngine:
    """Main explainability engine combining multiple methods"""
    
    def __init__(self, model: torch.nn.Module, config):
        """
        Initialize explainability engine
        
        Args:
            model: Classification model
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.gradcam = None
        self.layercam = None
        
        if config.explainability.use_gradcam:
            self.gradcam = GradCAM(
                model,
                target_layer=config.explainability.target_layer,
                device=config.system.device
            )
        
        if config.explainability.use_gradcam:
            self.layercam = LayerCAM(
                model,
                target_layer=config.explainability.target_layer,
                device=config.system.device
            )
    
    def explain(self, input_tensor: torch.Tensor, original_image: np.ndarray,
                target_class: Optional[int] = None, method: str = "gradcam") -> np.ndarray:
        """
        Generate explanation for prediction
        
        Args:
            input_tensor: Input tensor
            original_image: Original image for overlay
            target_class: Target class to explain
            method: Explanation method ('gradcam' or 'layercam')
        
        Returns:
            Explanation visualization
        """
        if method == "gradcam" and self.gradcam is not None:
            return self.gradcam.visualize(
                input_tensor,
                original_image,
                target_class=target_class,
                colormap=self.config.explainability.colormap,
                alpha=self.config.explainability.alpha
            )
        elif method == "layercam" and self.layercam is not None:
            cam = self.layercam.generate(input_tensor, target_class)
            return self._overlay_heatmap(original_image, cam)
        else:
            logger.error(f"Method {method} not available")
            return original_image
    
    @staticmethod
    def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                         alpha: float = 0.4) -> np.ndarray:
        """Overlay heatmap on image"""
        # Normalize heatmap to 0-255
        heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        overlay = cv2.addWeighted(image.astype(np.uint8), 1 - alpha,
                                  colored.astype(np.uint8), alpha, 0)
        return overlay


def save_explanation(image: np.ndarray, filepath: str):
    """Save explanation visualization"""
    try:
        cv2.imwrite(filepath, image)
        logger.info(f"Explanation saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving explanation: {e}")


if __name__ == "__main__":
    print("Explainability module loaded successfully")
