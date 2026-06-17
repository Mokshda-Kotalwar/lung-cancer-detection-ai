"""
Example: Configuration Customization
Demonstrates how to customize system configuration
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, CONFIG_DIR


def customize_models():
    """Customize model configuration"""
    print("\n=== Model Configuration ===")
    
    # YOLOv8 settings
    config.model.yolo_model = "yolov8m"  # nano, small, medium, large, xlarge
    config.model.yolo_conf_threshold = 0.45
    config.model.yolo_iou_threshold = 0.5
    
    # EfficientNet settings
    config.model.efficientnet_version = "b4"  # b0-b7
    config.model.num_classes = 3
    
    # Input settings
    config.model.input_size = 512
    config.model.batch_size = 16
    
    print(f"✓ YOLO Model: {config.model.yolo_model}")
    print(f"✓ EfficientNet: {config.model.efficientnet_version}")
    print(f"✓ Input Size: {config.model.input_size}x{config.model.input_size}")


def customize_preprocessing():
    """Customize preprocessing configuration"""
    print("\n=== Preprocessing Configuration ===")
    
    # Normalization method
    config.preprocessing.normalize_method = "minmax"  # minmax, zscore, clahe
    
    # Lung segmentation
    config.preprocessing.use_lung_mask = True
    config.preprocessing.lung_threshold = -400
    
    # Augmentation
    config.model.apply_augmentation = True
    config.preprocessing.rotation_range = 20
    
    print(f"✓ Normalization: {config.preprocessing.normalize_method}")
    print(f"✓ Lung Mask: {config.preprocessing.use_lung_mask}")
    print(f"✓ Augmentation: {config.model.apply_augmentation}")


def customize_risk_assessment():
    """Customize risk assessment configuration"""
    print("\n=== Risk Assessment Configuration ===")
    
    # Risk thresholds
    config.risk.risk_threshold_low = 0.3
    config.risk.risk_threshold_high = 0.7
    
    # Clinical features
    config.risk.use_clinical_features = True
    config.risk.clinical_feature_weight = 0.2
    
    # Uncertainty quantification
    config.risk.compute_uncertainty = True
    config.risk.monte_carlo_iterations = 10
    
    print(f"✓ Risk Threshold (Low): {config.risk.risk_threshold_low}")
    print(f"✓ Risk Threshold (High): {config.risk.risk_threshold_high}")
    print(f"✓ Clinical Features: {config.risk.use_clinical_features}")
    print(f"✓ Uncertainty Computation: {config.risk.compute_uncertainty}")


def customize_reporting():
    """Customize report generation configuration"""
    print("\n=== Report Configuration ===")
    
    # Report format
    config.report.report_format = "pdf"  # pdf, html, both
    config.report.include_recommendations = True
    config.report.include_visualizations = True
    
    # PDF settings
    config.report.include_clinical_summary = True
    config.report.include_gradcam = True
    
    print(f"✓ Report Format: {config.report.report_format}")
    print(f"✓ Include Recommendations: {config.report.include_recommendations}")
    print(f"✓ Include Grad-CAM: {config.report.include_gradcam}")


def customize_system():
    """Customize system configuration"""
    print("\n=== System Configuration ===")
    
    # Device
    config.system.device = "cuda"  # cuda, cpu, mps
    config.system.use_mixed_precision = True
    
    # Logging
    config.system.log_level = "INFO"
    
    # Reproducibility
    config.system.random_seed = 42
    config.system.deterministic = True
    
    print(f"✓ Device: {config.system.device}")
    print(f"✓ Mixed Precision: {config.system.use_mixed_precision}")
    print(f"✓ Random Seed: {config.system.random_seed}")


def save_custom_config():
    """Save custom configuration to file"""
    print("\n=== Saving Configuration ===")
    
    config_path = CONFIG_DIR / "custom_config.json"
    config.save_to_json(str(config_path))
    
    print(f"✓ Configuration saved to: {config_path}")
    
    # Display saved config
    with open(config_path, 'r') as f:
        custom_config = json.load(f)
    
    print(f"\nSaved Configuration:")
    print(json.dumps(custom_config, indent=2))


def main():
    """Run configuration customization example"""
    print("=" * 80)
    print("LUNG CANCER DETECTION - CONFIGURATION CUSTOMIZATION EXAMPLE")
    print("=" * 80)
    
    # Customize all components
    customize_models()
    customize_preprocessing()
    customize_risk_assessment()
    customize_reporting()
    customize_system()
    
    # Save configuration
    save_custom_config()
    
    print("\n" + "=" * 80)
    print("✓ Configuration customization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
