# Setup and Usage Guide - Lung Cancer Detection AI System

## 📦 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd lung-cancer-detection-ai
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Or using conda
conda create -n lungcancer python=3.10
conda activate lungcancer
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure System
```bash
# Set environment variables (optional)
export LUNGCANCER_DEVICE=cuda
export LUNGCANCER_LOG_LEVEL=INFO

# On Windows PowerShell:
$env:LUNGCANCER_DEVICE="cuda"
```

---

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

```bash
# Launch Streamlit app
python main.py --web

# Or directly
streamlit run app/main.py
```

The app will open at `http://localhost:8501`

### Option 2: Command Line Interface

```bash
# Analyze single image
python main.py --image path/to/image.png

# With patient information
python main.py --image path/to/image.png \
    --patient-id P001 \
    --name "John Doe" \
    --age 65

# Skip report generation
python main.py --image path/to/image.png --no-report
```

### Option 3: Python API

```python
from config import config
from main import LungCancerDetectionPipeline
import numpy as np

# Initialize
pipeline = LungCancerDetectionPipeline(config)

# Prepare patient data
patient_info = {
    'patient_id': 'P001',
    'name': 'John Doe',
    'age': 65,
    'gender': 'M'
}

# Analyze
results = pipeline.analyze_image(
    'path/to/image.png',
    patient_info=patient_info,
    generate_report=True
)

# Access results
if results['status'] == 'success':
    risk = results['risk_assessment']
    print(f"Risk Score: {risk.risk_score:.3f}")
    print(f"Risk Level: {risk.risk_level.value}")
    
    for rec in results['recommendations']:
        print(f"- {rec.action}")
```

---

## 📋 Configuration

### Basic Configuration
Edit `config.py` to adjust default parameters:

```python
from config import config

# Model settings
config.model.yolo_model = "yolov8m"
config.model.input_size = 512
config.model.yolo_conf_threshold = 0.45

# Preprocessing
config.preprocessing.normalize_method = "minmax"
config.preprocessing.use_lung_mask = True

# Risk assessment
config.risk.risk_threshold_low = 0.3
config.risk.risk_threshold_high = 0.7

# Report generation
config.report.include_recommendations = True
config.report.include_visualizations = True

# Save configuration
config.save_to_json("models/configs/custom_config.json")
```

### Environment Variables
```bash
# Device selection
export LUNGCANCER_DEVICE=cuda    # cuda, cpu, mps

# Logging
export LUNGCANCER_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Model
export LUNGCANCER_BATCH_SIZE=16
export LUNGCANCER_NUM_WORKERS=4
```

---

## 📊 Input Data

### Supported Formats
- **DICOM** (.dcm): Medical imaging standard
- **PNG** (.png): 8-bit or 16-bit grayscale/RGB
- **JPEG** (.jpg, .jpeg): Standard image format
- **NIfTI** (.nii, .nii.gz): 3D medical volumes

### Image Requirements
- **Resolution**: Typically 512x512 pixels
- **Bit Depth**: 8-bit, 16-bit, or 32-bit
- **Format**: Grayscale preferred (RGB auto-converted)
- **File Size**: <100MB per image

### Prepare DICOM Files
```python
import pydicom
import numpy as np

# Load DICOM
ds = pydicom.dcmread('image.dcm')
image = ds.pixel_array.astype(np.float32)

# Apply window/level transformation
from src.preprocessing import DICOMProcessor
processor = DICOMProcessor(window_center=40, window_width=400)
windowed = processor.apply_window_level(image)
```

---

## 🔍 Analysis Workflow

### 1. Image Preprocessing
```python
from src.preprocessing import ImagePreprocessor
from config import config

preprocessor = ImagePreprocessor(config)

# Single image
processed = preprocessor.preprocess(image)

# Batch
batch = preprocessor.batch_preprocess(images)
```

### 2. Detection & Classification
```python
from src.detection import EnsembleDetector

ensemble = EnsembleDetector(config)
detection_result, classification_result = ensemble.process(image)

# Results
print(f"Nodules detected: {len(detection_result.boxes)}")
print(f"Classification: {classification_result.class_name}")
print(f"Confidence: {classification_result.confidence:.3f}")
```

### 3. Explainability
```python
from src.xai import GradCAM

gradcam = GradCAM(classifier.model, target_layer="layer4")
explanation = gradcam.visualize(input_tensor, original_image)
```

### 4. Risk Assessment
```python
from src.risk import RiskScorer

risk_scorer = RiskScorer(config)
risk_assessment = risk_scorer.calculate_risk(
    classification_confidence=0.85,
    detection_count=1,
    detection_size=15.0,
    detection_confidence=0.92
)

print(f"Risk Score: {risk_assessment.risk_score:.3f}")
print(f"Risk Level: {risk_assessment.risk_level.value}")
```

### 5. Recommendations
```python
from src.risk import RecommendationEngine

rec_engine = RecommendationEngine(config)
recommendations = rec_engine.generate_recommendations(
    risk_assessment,
    patient_info={'age': 65, 'smoker': True}
)

for rec in recommendations:
    print(f"{rec.priority}: {rec.action}")
```

### 6. Report Generation
```python
from src.reporting import ReportGenerator

report_gen = ReportGenerator(config)
report_gen.generate_pdf_report(
    output_path="report.pdf",
    patient_info={'patient_id': 'P001', 'name': 'John Doe'},
    detection_result=detection_result,
    classification_result=classification_result,
    risk_assessment=risk_assessment,
    recommendations=recommendations
)
```

---

## 🧪 Testing

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_preprocessing.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Examples
```python
def test_preprocessing():
    from src.preprocessing import ImagePreprocessor
    preprocessor = ImagePreprocessor(config)
    
    image = np.random.rand(512, 512)
    processed = preprocessor.preprocess(image)
    
    assert isinstance(processed, torch.Tensor)
    assert processed.shape[0] == 1

def test_risk_assessment():
    from src.risk import RiskScorer
    scorer = RiskScorer(config)
    
    risk = scorer.calculate_risk(0.8, 1, 15, 0.9)
    assert 0 <= risk.risk_score <= 1
    assert risk.risk_level is not None
```

---

## 📈 Output Files

### Generated Reports
```
outputs/
├── reports/
│   ├── report_20240115_143022.pdf    # PDF report
│   └── report_20240115_143022.html   # HTML report
├── predictions/
│   ├── detections.json
│   └── classifications.json
├── gradcam/
│   └── gradcam_overlay.png
└── logs/
    └── system.log
```

### Report Contents
1. **Title Page**: Report metadata and date
2. **Patient Information**: Demographics, study date
3. **AI Findings**: Detection results, confidence scores
4. **Classification**: Predicted class and probabilities
5. **Risk Assessment**: Risk score, level, and metrics
6. **Recommendations**: Clinical action items with priority
7. **Visualizations**: Original image + Grad-CAM overlay
8. **Disclaimer**: Legal and regulatory notices

---

## 🔐 Security & Privacy

### Data Protection
```python
# Encrypt sensitive information
from pathlib import Path

# Store in secure location
secure_dir = Path("/secure/location")
secure_dir.chmod(0o700)  # Owner read/write/execute only
```

### HIPAA Compliance
- Remove DICOM identifiers: `pydicom.dcmread(...).remove_private_tags()`
- Encrypt patient data at rest and in transit
- Maintain audit logs
- Use role-based access control

### Secure Configuration
```python
import os
from dotenv import load_dotenv

# Load from .env file (NOT version controlled)
load_dotenv()

db_password = os.getenv("DB_PASSWORD")
api_key = os.getenv("API_KEY")
```

---

## 🐛 Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
export LUNGCANCER_DEVICE=cpu
python main.py --image image.png
```

### Model Loading
```bash
# Download models manually
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Or use pre-downloaded weights
export LUNGCANCER_YOLO_WEIGHTS=/path/to/weights/yolov8m.pt
```

### Memory Issues
```python
# Reduce batch size
config.system.batch_size = 8

# Use smaller models
config.model.yolo_model = "yolov8n"
config.model.efficientnet_version = "b2"

# Enable mixed precision
config.system.use_mixed_precision = True
```

### Slow Performance
```python
# Enable GPU
config.system.device = "cuda"

# Increase workers
config.system.num_workers = 8

# Reduce input size
config.model.input_size = 448
```

---

## 📚 Advanced Usage

### Custom Model Training
```python
import torch
from torch.utils.data import DataLoader

# Prepare dataset
# ... (data loading code)

# Fine-tune model
classifier.model.classifier = torch.nn.Linear(1280, 3)
classifier.model.to(config.system.device)

# Training loop
# ... (training code)
```

### Batch Processing
```python
from pathlib import Path
import json

image_dir = Path("data/raw/images")
results = []

for image_path in image_dir.glob("*.png"):
    result = pipeline.analyze_image(str(image_path))
    results.append(result)

# Save results
with open("outputs/batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Custom Callbacks
```python
class AnalysisCallback:
    def on_image_loaded(self, image): pass
    def on_preprocessing_complete(self, processed): pass
    def on_detection_complete(self, detections): pass
    def on_analysis_complete(self, results): pass

# Integrate with pipeline
pipeline.register_callback(AnalysisCallback())
```

---

## 📞 Support

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size or use smaller model |
| Slow inference | Enable GPU, reduce image size |
| Model not found | Check internet, restart app |
| DICOM read error | Verify file integrity, install pydicom |
| Permission denied | Check directory permissions |

### Getting Help
1. Check logs: `outputs/logs/system.log`
2. Run with debug: `python main.py --log-level DEBUG`
3. Review error message carefully
4. Check documentation and examples

---

## 📄 License

See LICENSE file for terms and conditions.

---

## ⚠️ Medical Disclaimer

**This system is provided for RESEARCH PURPOSES ONLY.**

- NOT approved for clinical diagnosis
- NOT a replacement for medical professionals
- Requires validation before clinical use
- User assumes all liability for misuse
- Healthcare provider oversight mandatory
