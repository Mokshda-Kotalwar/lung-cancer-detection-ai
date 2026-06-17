"""
Configuration Module for Lung Cancer Detection System
Centralized configuration management for all system components
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

# Project Root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
CONFIG_DIR = MODELS_DIR / "configs"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"
CLINICAL_DATA = DATA_DIR / "clinical"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, 
                  CONFIG_DIR, RAW_DATA, PROCESSED_DATA, CLINICAL_DATA]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model Configuration"""
    # YOLOv8 Configuration
    yolo_model: str = "yolov8m"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    yolo_pretrained: bool = True
    yolo_device: str = "cuda"  # cuda or cpu
    yolo_conf_threshold: float = 0.45
    yolo_iou_threshold: float = 0.5
    
    # EfficientNet Configuration
    efficientnet_version: str = "efficientnet-b4"  # b0-b7
    efficientnet_pretrained: bool = True
    num_classes: int = 3  # Benign, Malignant, Uncertain
    
    # Input Configuration
    input_size: int = 512
    batch_size: int = 16
    num_workers: int = 4
    
    # Training Configuration
    learning_rate: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Augmentation
    apply_augmentation: bool = True
    rotation_range: int = 20
    brightness_range: List[float] = None
    
    def __post_init__(self):
        if self.brightness_range is None:
            self.brightness_range = [0.8, 1.2]


@dataclass
class PreprocessingConfig:
    """Preprocessing Configuration"""
    # DICOM Processing
    normalize_method: str = "minmax"  # minmax, zscore, clahe
    window_center: int = 40
    window_width: int = 400
    
    # Image Processing
    target_size: int = 512
    interpolation: str = "bilinear"
    preserve_aspect_ratio: bool = True
    
    # Artifacts Removal
    remove_patient_tags: bool = True
    handle_missing_values: bool = True
    denoise: bool = True
    denoise_sigma: float = 1.0
    
    # Lung Segmentation
    use_lung_mask: bool = True
    lung_threshold: float = -400  # HU units for lung
    connectivity: int = 2


@dataclass
class ExplainabilityConfig:
    """Grad-CAM and Explainability Configuration"""
    # Grad-CAM
    use_gradcam: bool = True
    target_layer: str = "layer4"  # Layer to visualize
    colormap: str = "jet"  # Visualization colormap
    alpha: float = 0.4  # Overlay transparency
    
    # LIME (future)
    use_lime: bool = False
    num_samples: int = 1000
    
    # SHAP (future)
    use_shap: bool = False


@dataclass
class RiskConfig:
    """Risk Assessment Configuration"""
    # Risk Scoring
    risk_threshold_low: float = 0.3
    risk_threshold_high: float = 0.7
    
    # Clinical Integration
    use_clinical_features: bool = True
    clinical_feature_weight: float = 0.2
    
    # Uncertainty Quantification
    compute_uncertainty: bool = True
    monte_carlo_iterations: int = 10
    
    # Recommendation System
    recommendation_model: str = "rule_based"  # rule_based, ml_based


@dataclass
class ReportConfig:
    """Report Generation Configuration"""
    # Report Format
    report_format: str = "pdf"  # pdf, html, both
    include_recommendations: bool = True
    include_visualizations: bool = True
    
    # PDF Settings
    pdf_font_size: int = 12
    pdf_margin: int = 20
    pdf_page_size: str = "A4"
    
    # Content
    include_clinical_summary: bool = True
    include_risk_score: bool = True
    include_gradcam: bool = True
    include_differential_diagnosis: bool = True
    include_next_steps: bool = True


@dataclass
class WebConfig:
    """Streamlit Web Configuration"""
    # Server
    server_port: int = 8501
    server_host: str = "0.0.0.0"
    max_upload_size_mb: int = 100
    
    # UI
    theme: str = "dark"  # light, dark
    display_fps: bool = False
    
    # Paths
    upload_dir: str = str(OUTPUTS_DIR / "uploads")
    results_dir: str = str(OUTPUTS_DIR / "results")


@dataclass
class SystemConfig:
    """System-wide Configuration"""
    # Logging
    log_level: str = "INFO"
    log_file: str = str(PROJECT_ROOT / "logs" / "system.log")
    
    # Cache
    enable_cache: bool = True
    cache_dir: str = str(PROJECT_ROOT / "cache")
    
    # Device
    device: str = "cuda"  # cuda, cpu, mps
    use_mixed_precision: bool = True
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Parallel Processing
    num_workers: int = 4
    batch_size: int = 16


class Config:
    """Main Configuration Manager"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.preprocessing = PreprocessingConfig()
        self.explainability = ExplainabilityConfig()
        self.risk = RiskConfig()
        self.report = ReportConfig()
        self.web = WebConfig()
        self.system = SystemConfig()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        import logging
        log_dir = Path(self.system.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.system.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.system.log_file),
                logging.StreamHandler()
            ]
        )
    
    def save_to_json(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            'model': self.model.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'explainability': self.explainability.__dict__,
            'risk': self.risk.__dict__,
            'report': self.report.__dict__,
            'web': self.web.__dict__,
            'system': self.system.__dict__,
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
    
    def load_from_json(self, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Update configurations (simplified)
        for key, value in config_dict.items():
            if hasattr(self, key):
                getattr(self, key).__dict__.update(value)


# Global configuration instance
config = Config()

if __name__ == "__main__":
    print("Configuration loaded successfully")
    config.save_to_json(CONFIG_DIR / "default_config.json")
    print(f"Config saved to {CONFIG_DIR / 'default_config.json'}")
