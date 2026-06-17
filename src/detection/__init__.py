"""
Model Inference Module for Lung Cancer Detection
Integrates YOLOv8 for detection and EfficientNet for classification
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import cv2

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result container"""
    boxes: np.ndarray  # [x1, y1, x2, y2, conf, class]
    confidences: np.ndarray
    class_ids: np.ndarray
    image_shape: Tuple[int, int]
    processing_time: float


@dataclass
class ClassificationResult:
    """Classification result container"""
    class_id: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float


class YOLODetector:
    """YOLOv8 based nodule detection"""
    
    def __init__(self, model_name: str = "yolov8m", device: str = "cuda"):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_name: Model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            device: Device to run on (cuda, cpu)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            # Download and load pretrained model
            model_path = f"{self.model_name}.pt"
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            logger.info(f"YOLOv8 ({self.model_name}) loaded successfully on {self.device}")
        except ImportError:
            logger.error("ultralytics package required. Install: pip install ultralytics")
        except Exception as e:
            logger.error(f"Error loading YOLOv8: {e}")
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.45, 
               iou_threshold: float = 0.5) -> DetectionResult:
        """
        Detect nodules in image
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        
        Returns:
            DetectionResult with bounding boxes
        """
        import time
        start_time = time.time()
        
        if self.model is None:
            logger.error("Model not loaded")
            return DetectionResult(np.array([]), np.array([]), np.array([]), 
                                  image.shape, 0)
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Run detection
        results = self.model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        result = results[0]
        
        # Extract detections
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            boxes=boxes,
            confidences=confidences,
            class_ids=class_ids,
            image_shape=image.shape,
            processing_time=processing_time
        )
    
    def draw_detections(self, image: np.ndarray, detections: DetectionResult,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes on image"""
        result_image = image.copy()
        
        if len(detections.boxes) == 0:
            return result_image
        
        for box, conf in zip(detections.boxes, detections.confidences):
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence
            text = f"Conf: {conf:.2f}"
            cv2.putText(result_image, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return result_image


class EfficientNetClassifier:
    """EfficientNet based classification"""
    
    def __init__(self, model_version: str = "b4", num_classes: int = 3, device: str = "cuda"):
        """
        Initialize EfficientNet classifier
        
        Args:
            model_version: Model version (b0-b7)
            num_classes: Number of output classes
            device: Device to run on
        """
        self.model_version = model_version
        self.num_classes = num_classes
        self.device = device
        self.model = None
        self.class_names = ["Benign", "Malignant", "Uncertain"]
        self._load_model()
    
    def _load_model(self):
        """Load EfficientNet model"""
        try:
            import timm
            from torch import nn
            
            # Load pretrained model
            model_name = f"efficientnet_{self.model_version}"
            self.model = timm.create_model(model_name, pretrained=True)
            
            # Modify final layer for classification
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, self.num_classes)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"EfficientNet-{self.model_version} loaded successfully")
        except ImportError:
            logger.error("timm package required. Install: pip install timm")
        except Exception as e:
            logger.error(f"Error loading EfficientNet: {e}")
    
    def classify(self, image: torch.Tensor) -> ClassificationResult:
        """
        Classify image
        
        Args:
            image: Input tensor (C, H, W) or (B, C, H, W)
        
        Returns:
            ClassificationResult
        """
        import time
        start_time = time.time()
        
        if self.model is None:
            logger.error("Model not loaded")
            return ClassificationResult(0, "Unknown", 0.0, {}, 0)
        
        # Ensure image is on correct device
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        if image.device != self.device:
            image = image.to(self.device)
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(image)
            probabilities = torch.softmax(logits, dim=1)
            confidence, class_id = torch.max(probabilities[0], 0)
        
        # Get probability distribution
        prob_dict = {
            self.class_names[i]: probabilities[0, i].item()
            for i in range(self.num_classes)
        }
        
        processing_time = time.time() - start_time
        
        return ClassificationResult(
            class_id=int(class_id),
            class_name=self.class_names[int(class_id)],
            confidence=float(confidence),
            probabilities=prob_dict,
            processing_time=processing_time
        )
    
    def batch_classify(self, images: torch.Tensor) -> List[ClassificationResult]:
        """Classify a batch of images"""
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        images = images.to(self.device)
        
        with torch.no_grad():
            logits = self.model(images)
            probabilities = torch.softmax(logits, dim=1)
        
        results = []
        for i in range(images.shape[0]):
            confidence, class_id = torch.max(probabilities[i], 0)
            
            prob_dict = {
                self.class_names[j]: probabilities[i, j].item()
                for j in range(self.num_classes)
            }
            
            results.append(ClassificationResult(
                class_id=int(class_id),
                class_name=self.class_names[int(class_id)],
                confidence=float(confidence),
                probabilities=prob_dict,
                processing_time=0
            ))
        
        return results
    
    def load_checkpoint(self, filepath: str):
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    
    def save_checkpoint(self, filepath: str):
        """Save model to checkpoint"""
        try:
            torch.save(self.model.state_dict(), filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")


class EnsembleDetector:
    """Ensemble of detection and classification models"""
    
    def __init__(self, config):
        """Initialize ensemble"""
        self.config = config
        self.detector = YOLODetector(
            model_name=config.model.yolo_model,
            device=config.model.yolo_device
        )
        self.classifier = EfficientNetClassifier(
            model_version=config.model.efficientnet_version,
            num_classes=config.model.num_classes,
            device=config.model.yolo_device
        )
    
    def process(self, image: np.ndarray) -> Tuple[DetectionResult, Optional[ClassificationResult]]:
        """
        Process image with detection and classification
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            Tuple of (DetectionResult, ClassificationResult)
        """
        # Detection
        detection_result = self.detector.detect(
            image,
            conf_threshold=self.config.model.yolo_conf_threshold,
            iou_threshold=self.config.model.yolo_iou_threshold
        )
        
        # Classification (if detections found)
        classification_result = None
        if len(detection_result.boxes) > 0:
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            classification_result = self.classifier.classify(image_tensor)
        
        return detection_result, classification_result


if __name__ == "__main__":
    from config import config
    print("Detection and classification module loaded successfully")
