# API Reference - Lung Cancer Detection System

## Core Modules

### 1. Configuration (`config.py`)

#### Class: `Config`
Main configuration manager for the system.

```python
from config import config

# Access configurations
config.model.yolo_model
config.preprocessing.normalize_method
config.risk.risk_threshold_high
config.system.device
```

**Main Config Classes:**
- `ModelConfig`: ML model parameters
- `PreprocessingConfig`: Image processing settings
- `ExplainabilityConfig`: XAI configuration
- `RiskConfig`: Risk assessment parameters
- `ReportConfig`: Report generation settings
- `WebConfig`: Streamlit settings
- `SystemConfig`: System-wide configuration

---

### 2. Preprocessing (`src/preprocessing/`)

#### Class: `ImagePreprocessor`
Main preprocessing pipeline.

```python
from src.preprocessing import ImagePreprocessor
from config import config

preprocessor = ImagePreprocessor(config)

# Single image
processed = preprocessor.preprocess(image, apply_aug=False)

# Batch
batch = preprocessor.batch_preprocess(images, apply_aug=False)
```

#### Class: `DICOMProcessor`
DICOM file handling.

```python
from src.preprocessing import DICOMProcessor

processor = DICOMProcessor(window_center=40, window_width=400)
image = processor.load_dicom('file.dcm')
windowed = processor.apply_window_level(image)
```

#### Class: `ImageNormalizer`
Image normalization methods.

```python
from src.preprocessing import ImageNormalizer

# Min-max normalization
normalized = ImageNormalizer.minmax_normalize(image, min_val=0, max_val=1)

# Z-score normalization
normalized = ImageNormalizer.zscore_normalize(image)

# CLAHE
enhanced = ImageNormalizer.clahe_normalize(image, clip_limit=2.0)
```

#### Class: `LungSegmentation`
Lung region extraction.

```python
from src.preprocessing import LungSegmentation

segmented, mask = LungSegmentation.segment_lungs(image, threshold=-400)
masked = LungSegmentation.apply_mask(image, mask)
```

---

### 3. Detection & Classification (`src/detection/`)

#### Class: `YOLODetector`
YOLOv8-based nodule detection.

```python
from src.detection import YOLODetector

detector = YOLODetector(model_name="yolov8m", device="cuda")
result = detector.detect(image, conf_threshold=0.45, iou_threshold=0.5)

# DetectionResult
result.boxes           # [x1, y1, x2, y2, ...]
result.confidences     # Confidence scores
result.class_ids       # Class IDs
result.image_shape     # (H, W)
result.processing_time # Inference time

# Visualization
image_with_boxes = detector.draw_detections(image, result)
```

#### Class: `EfficientNetClassifier`
EfficientNet-based classification.

```python
from src.detection import EfficientNetClassifier

classifier = EfficientNetClassifier(
    model_version="b4",
    num_classes=3,
    device="cuda"
)

result = classifier.classify(image_tensor)

# ClassificationResult
result.class_id         # Predicted class (0-2)
result.class_name       # "Benign", "Malignant", "Uncertain"
result.confidence       # Confidence score (0-1)
result.probabilities    # {class: prob, ...}
result.processing_time  # Inference time

# Batch classification
results = classifier.batch_classify(images_tensor)

# Model management
classifier.load_checkpoint('path/to/checkpoint.pth')
classifier.save_checkpoint('path/to/checkpoint.pth')
```

#### Class: `EnsembleDetector`
Integrated detection and classification.

```python
from src.detection import EnsembleDetector

ensemble = EnsembleDetector(config)
detection_result, classification_result = ensemble.process(image)
```

---

### 4. Explainability (`src/xai/`)

#### Class: `GradCAM`
Gradient-weighted class activation mapping.

```python
from src.xai import GradCAM

gradcam = GradCAM(model, target_layer="layer4", device="cuda")

# Generate heatmap
heatmap = gradcam.generate(input_tensor, target_class=None)

# Visualize with overlay
overlay = gradcam.visualize(
    input_tensor,
    original_image,
    target_class=None,
    colormap="jet",
    alpha=0.4
)
```

#### Class: `LayerCAM`
Layer-wise relevance propagation variant.

```python
from src.xai import LayerCAM

layercam = LayerCAM(model, target_layer="layer4", device="cuda")
heatmap = layercam.generate(input_tensor, target_class=None)
```

#### Class: `ExplainabilityEngine`
Unified explainability interface.

```python
from src.xai import ExplainabilityEngine

engine = ExplainabilityEngine(classifier_model, config)
explanation = engine.explain(
    input_tensor,
    original_image,
    target_class=None,
    method="gradcam"
)
```

---

### 5. Risk Assessment (`src/risk/`)

#### Class: `RiskScorer`
Multi-factor risk calculation.

```python
from src.risk import RiskScorer

scorer = RiskScorer(config)

risk = scorer.calculate_risk(
    classification_confidence=0.85,
    detection_count=2,
    detection_size=15.0,        # mm
    detection_confidence=0.92,
    clinical_features={
        'smoking_pack_years': 40,
        'age': 0.65,              # normalized 0-1
        'family_history': 0.5
    }
)

# RiskAssessment
risk.risk_score             # 0-1 scale
risk.risk_level             # RiskLevel enum
risk.confidence             # Model confidence
risk.uncertainty            # Model uncertainty
risk.key_features           # List of contributing factors
risk.contributing_factors   # Dict of factor contributions
risk.recommendation         # Text recommendation
risk.followup_period        # Recommended follow-up
```

#### Class: `RecommendationEngine`
Clinical recommendation generation.

```python
from src.risk import RecommendationEngine

rec_engine = RecommendationEngine(config)

recommendations = rec_engine.generate_recommendations(
    risk_assessment,
    patient_info={'age': 65, 'smoker': True}
)

# ClinicalRecommendation list
for rec in recommendations:
    rec.action              # Recommended action
    rec.priority            # Low/Medium/High/Critical
    rec.reasoning           # Justification
    rec.followup_days       # Days until follow-up
    rec.additional_tests    # List of recommended tests
```

#### Class: `DifferentialDiagnosis`
Differential diagnosis suggestions.

```python
from src.risk import DifferentialDiagnosis

diagnoses = DifferentialDiagnosis.get_differential_diagnoses({
    'size': 12.0,
    'location': 'RUL'
})

# List of {condition, probability}
for dx in diagnoses:
    print(f"{dx['condition']}: {dx['probability']:.2f}")
```

---

### 6. Reporting (`src/reporting/`)

#### Class: `ReportGenerator`
PDF and HTML report generation.

```python
from src.reporting import ReportGenerator

report_gen = ReportGenerator(config)

# PDF Report
success = report_gen.generate_pdf_report(
    output_path='report.pdf',
    patient_info={
        'patient_id': 'P001',
        'name': 'John Doe',
        'age': 65,
        'gender': 'M'
    },
    detection_result=detection_result,
    classification_result=classification_result,
    risk_assessment=risk_assessment,
    recommendations=recommendations,
    gradcam_image=gradcam_overlay,
    original_image=original_image
)

# HTML Report
success = report_gen.generate_html_report(
    output_path='report.html',
    patient_info=patient_info,
    detection_result=detection_result,
    classification_result=classification_result,
    risk_assessment=risk_assessment,
    recommendations=recommendations
)
```

---

### 7. Main Pipeline (`main.py`)

#### Class: `LungCancerDetectionPipeline`
Complete analysis pipeline.

```python
from main import LungCancerDetectionPipeline

pipeline = LungCancerDetectionPipeline(config)

# Analyze image
results = pipeline.analyze_image(
    image_path='path/to/image.png',
    patient_info={
        'patient_id': 'P001',
        'name': 'John Doe',
        'age': 65
    },
    generate_report=True
)

# Results structure
results = {
    'status': 'success',                      # success or error
    'image_path': str,
    'detection_result': DetectionResult,
    'classification_result': ClassificationResult,
    'risk_assessment': RiskAssessment,
    'recommendations': List[ClinicalRecommendation],
    'report_path': Path
}
```

---

## Utility Functions (`src/utils.py`)

```python
from src.utils import (
    setup_logging,
    set_seed,
    get_device,
    count_parameters,
    count_trainable_parameters,
    calculate_metrics,
    validate_image_path,
    ProgressTracker
)

# Setup
setup_logging(log_file="logs.txt", level="INFO")
set_seed(42)
device = get_device()

# Model analysis
total_params = count_parameters(model)
trainable_params = count_trainable_parameters(model)

# Metrics
metrics = calculate_metrics(tp=80, fp=10, fn=20, tn=90)
# Returns: {sensitivity, specificity, precision, accuracy, f1_score, dice_coefficient}

# Validation
is_valid = validate_image_path('image.png')

# Progress tracking
tracker = ProgressTracker(total=100, desc="Processing")
for i in range(100):
    tracker.update()
tracker.finish()
```

---

## Streamlit Web Interface (`app/main.py`)

### Modes

**Single Image Analysis**
- Upload single image
- Real-time analysis
- Interactive visualizations
- PDF report generation

**Batch Processing**
- Upload multiple images
- Process queue management
- Results download

**Reports**
- View previous reports
- Archive management
- Batch download

### Features

```python
# Features available in web UI:
- Image upload (DICOM, PNG, JPG)
- Real-time processing
- Detection visualization
- Classification probabilities
- Risk assessment display
- Clinical recommendations
- PDF/HTML report download
- Results caching
```

---

## Data Structures

### DetectionResult
```python
@dataclass
class DetectionResult:
    boxes: np.ndarray              # [N, 4] - bounding boxes
    confidences: np.ndarray        # [N] - confidence scores
    class_ids: np.ndarray         # [N] - class IDs
    image_shape: Tuple[int, int]  # (H, W)
    processing_time: float        # Seconds
```

### ClassificationResult
```python
@dataclass
class ClassificationResult:
    class_id: int                 # 0-2
    class_name: str               # Class name
    confidence: float             # 0-1
    probabilities: Dict           # {class: prob}
    processing_time: float        # Seconds
```

### RiskAssessment
```python
@dataclass
class RiskAssessment:
    risk_score: float             # 0-1
    risk_level: RiskLevel         # Enum
    confidence: float             # 0-1
    uncertainty: float            # 0-1
    key_features: List[str]       # Feature list
    contributing_factors: Dict    # Factor dict
    recommendation: str           # Text
    followup_period: str          # Time period
```

### ClinicalRecommendation
```python
@dataclass
class ClinicalRecommendation:
    action: str                   # Recommended action
    priority: str                 # Priority level
    reasoning: str                # Justification
    followup_days: int            # Days
    additional_tests: List[str]   # Test list
```

---

## Enumerations

### RiskLevel
```python
class RiskLevel(Enum):
    LOW = "Low"
    INTERMEDIATE = "Intermediate"
    HIGH = "High"
    CRITICAL = "Critical"
```

---

## Error Handling

All modules implement comprehensive error handling:

```python
try:
    result = pipeline.analyze_image('image.png')
    if result and result['status'] == 'success':
        # Process results
        pass
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
except Exception as e:
    logger.error(f"Pipeline error: {e}", exc_info=True)
```

---

## Logging

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Logs are saved to: outputs/logs/system.log
```

---

## Configuration Examples

```python
# Enable GPU
config.system.device = "cuda"

# Set model
config.model.yolo_model = "yolov8x"
config.model.efficientnet_version = "b7"

# Adjust thresholds
config.model.yolo_conf_threshold = 0.50
config.risk.risk_threshold_high = 0.80

# Customize reports
config.report.include_gradcam = True
config.report.include_clinical_summary = True
```

---

## Performance Tips

1. **Speed**: Use smaller models (yolov8n, b2)
2. **Accuracy**: Use larger models (yolov8x, b7)
3. **Memory**: Reduce batch_size or use mixed precision
4. **Parallelization**: Process multiple images with batch processing

---

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/)
- [YOLOv8 Guide](https://docs.ultralytics.com/)
- [Streamlit API](https://docs.streamlit.io/)
- [ReportLab Reference](https://www.reportlab.com/docs/reportlab-userguide.pdf)

---

**Last Updated**: 2024  
**Version**: 1.0.0
