"""
Detection module package
"""

from .detection import (
    YOLODetector,
    EfficientNetClassifier,
    EnsembleDetector,
    DetectionResult,
    ClassificationResult
)

__all__ = [
    "YOLODetector",
    "EfficientNetClassifier",
    "EnsembleDetector",
    "DetectionResult",
    "ClassificationResult"
]
