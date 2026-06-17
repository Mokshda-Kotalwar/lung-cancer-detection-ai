# Lung Cancer Detection AI System - Complete Architecture Guide

## 🏗️ Project Structure

```
lung-cancer-detection-ai/
├── config.py                          # Central configuration management
├── main.py                            # CLI entry point
├── requirements.txt                   # Python dependencies
│
├── app/
│   ├── main.py                       # Streamlit web application
│   ├── pages/                        # Additional Streamlit pages
│   ├── static/                       # Static assets (CSS, JS, images)
│   └── templates/                    # HTML templates
│
├── src/
│   ├── __init__.py
│   ├── utils.py                      # Utility functions
│   ├── preprocessing.py              # Data preprocessing functions
│   ├── detection.py                  # Model detection functions
│   ├── xai.py                        # Explainability (Grad-CAM)
│   ├── risk.py                       # Risk assessment
│   ├── reporting.py                  # Report generation
│   │
│   ├── preprocessing/
│   │   └── __init__.py              # DICOM processing, normalization, augmentation
│   ├── detection/
│   │   └── __init__.py              # YOLOv8 and EfficientNet
│   ├── xai/
│   │   └── __init__.py              # Grad-CAM implementation
│   ├── risk/
│   │   └── __init__.py              # Risk scoring and recommendations
│   ├── reporting/
│   │   └── __init__.py              # ReportLab PDF generation
│   ├── classification/
│   │   └── __init__.py              # Classification models
│   ├── segmentation/
│   │   └── __init__.py              # Segmentation utilities
│   ├── uncertainty/
│   │   └── __init__.py              # Uncertainty quantification
│   └── recommendation/
│       └── __init__.py              # Recommendation engine
│
├── models/
│   ├── checkpoints/                 # Model checkpoints
│   ├── configs/                     # Model configuration files
│   └── trained/                     # Trained models
│
├── data/
│   ├── raw/                         # Raw input data
│   ├── processed/                   # Processed data
│   └── clinical/                    # Clinical metadata
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── outputs/
│   ├── predictions/                 # Model predictions
│   ├── reports/                     # Generated reports
│   ├── gradcam/                     # Visualizations
│   └── logs/                        # Log files
│
├── tests/
│   ├── test_preprocessing.py        # Preprocessing tests
│   └── test_models.py               # Model tests
│
└── docs/
    ├── API.md                       # API documentation
    ├── SETUP.md                     # Setup instructions
    └── METHODOLOGY.md               # Technical methodology
```

---

## 🔧 Core Modules

### 1. **Configuration (`config.py`)**
- Centralized configuration management
- Dataclass-based settings
- Support for environment variables
- Automatic directory creation

**Key Components:**
- `ModelConfig`: ML model parameters
- `PreprocessingConfig`: Image processing settings
- `ExplainabilityConfig`: XAI settings
- `RiskConfig`: Risk assessment parameters
- `ReportConfig`: Report generation settings

### 2. **Preprocessing Module (`src/preprocessing/`)**
- **DICOM Processing**: Window/level transformation, metadata extraction
- **Image Normalization**: MinMax, Z-score, CLAHE
- **Lung Segmentation**: Threshold-based + morphological operations
- **Data Augmentation**: Rotation, brightness, affine transforms

**Classes:**
- `DICOMProcessor`: DICOM file handling
- `ImageNormalizer`: Multiple normalization methods
- `LungSegmentation`: Lung region extraction
- `ImageAugmentation`: Training data augmentation
- `ImagePreprocessor`: Complete pipeline

### 3. **Detection & Classification (`src/detection/`)**
- **YOLOv8**: Real-time nodule detection
- **EfficientNet**: Malignancy classification
- **Ensemble**: Combined detection + classification

**Classes:**
- `YOLODetector`: YOLO-based detection
- `EfficientNetClassifier`: EfficientNet classification
- `EnsembleDetector`: Integration of both

**Output Structures:**
- `DetectionResult`: Bounding boxes, confidences
- `ClassificationResult`: Class prediction, probabilities

### 4. **Explainability (`src/xai/`)**
- **Grad-CAM**: Gradient-weighted class activation maps
- **LayerCAM**: Alternative attention mechanism
- **Visualization**: Heatmap overlay and colorization

**Classes:**
- `GradCAM`: Attention-based explanations
- `LayerCAM`: Layer-wise relevance
- `ExplainabilityEngine`: Unified interface

### 5. **Risk Assessment (`src/risk/`)**
- **Risk Scoring**: Multi-factor risk calculation
- **Clinical Integration**: Patient-specific factors
- **Uncertainty Quantification**: Model confidence estimation
- **Recommendations**: Actionable clinical guidance

**Classes:**
- `RiskScorer`: Risk calculation engine
- `RecommendationEngine`: Recommendation generation
- `DifferentialDiagnosis`: Diagnosis suggestions

**Output Structures:**
- `RiskAssessment`: Score, level, confidence
- `ClinicalRecommendation`: Action items with priority

### 6. **Report Generation (`src/reporting/`)**
- **PDF Reports**: Professional medical reports (ReportLab)
- **HTML Reports**: Web-viewable reports
- **Rich Formatting**: Tables, images, styling
- **Customizable Templates**: Per-institution branding

**Classes:**
- `ReportGenerator`: Multi-format report generation

**Features:**
- Patient information
- AI findings and metrics
- Risk assessment visualization
- Clinical recommendations
- Grad-CAM visualizations
- Medical disclaimer

### 7. **Web Interface (`app/main.py`)**
- **Streamlit**: Interactive web dashboard
- **Single Image Analysis**: Real-time processing
- **Batch Processing**: Multiple images
- **Report Management**: Review and download reports

**Features:**
- Image upload (DICOM, PNG, JPG)
- Real-time analysis with progress
- Interactive visualizations
- PDF report generation
- Result caching

---

## 🚀 Usage

### CLI Interface

```bash
# Analyze single image
python main.py --image path/to/image.png --patient-id P001

# Launch web application
python main.py --web

# With patient info
python main.py --image path/to/image.png --patient-id P001 --name "John Doe" --age 65
```

### Web Interface

```bash
streamlit run app/main.py
```

### Python API

```python
from config import config
from main import LungCancerDetectionPipeline

# Initialize pipeline
pipeline = LungCancerDetectionPipeline(config)

# Analyze image
results = pipeline.analyze_image(
    'path/to/image.png',
    patient_info={'patient_id': 'P001', 'name': 'John Doe'},
    generate_report=True
)

# Access results
risk_score = results['risk_assessment'].risk_score
recommendations = results['recommendations']
```

---

## 📊 Data Flow

```
Image Input
    ↓
Preprocessing (Normalization, Segmentation)
    ↓
Detection (YOLOv8 - Nodule Localization)
    ↓
Classification (EfficientNet - Malignancy)
    ↓
Explainability (Grad-CAM Visualization)
    ↓
Risk Assessment (Multi-factor Scoring)
    ↓
Recommendations (Clinical Guidance)
    ↓
Report Generation (PDF/HTML)
```

---

## 🔐 Safety & Compliance

### Key Features
- **Uncertainty Quantification**: Model confidence metrics
- **Ensemble Predictions**: Multiple models validation
- **Clinical Validation**: Medical professional review required
- **HIPAA Compliance**: Secure data handling
- **Audit Logging**: Complete activity tracking
- **Medical Disclaimers**: Clear AI limitations

### Risk Management
- Input validation
- Error handling and recovery
- Graceful degradation
- Comprehensive logging

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_preprocessing.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB RAM (minimum)
- 16GB RAM (recommended)

### Dependencies
- PyTorch 2.0+
- OpenCV (computer vision)
- Streamlit (web interface)
- ReportLab (PDF generation)
- YOLOv8 (object detection)
- EfficientNet (classification)
- pydicom (medical imaging)

See `requirements.txt` for complete list.

---

## 🔄 Configuration Management

### Environment Variables
```bash
export LUNGCANCER_DEVICE=cuda
export LUNGCANCER_LOG_LEVEL=INFO
export LUNGCANCER_BATCH_SIZE=16
```

### Config Files
```python
from config import config

# Modify configuration
config.model.yolo_conf_threshold = 0.50
config.risk.risk_threshold_high = 0.75

# Save configuration
config.save_to_json("models/configs/custom_config.json")
```

---

## 📈 Performance Metrics

### Medical Metrics
- **Sensitivity**: True positive rate (recall)
- **Specificity**: True negative rate
- **Precision**: Positive predictive value
- **Dice Coefficient**: Image segmentation overlap
- **AUC-ROC**: Classifier performance

### Computational Metrics
- **Inference Time**: <2s per image
- **Memory Usage**: ~2GB GPU / ~4GB CPU
- **Throughput**: ~30 images/minute (single GPU)

---

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Use smaller model variant (yolov8n instead of yolov8x)

2. **Model Loading Fails**
   - Check internet connection for model download
   - Verify disk space for model weights

3. **DICOM Reading Error**
   - Ensure pydicom is installed
   - Check DICOM file validity

### Debug Mode
```bash
python main.py --image path/to/image.png --log-level DEBUG
```

---

## 📚 Additional Resources

- [YOLO Documentation](https://docs.ultralytics.com/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- [ReportLab Documentation](https://www.reportlab.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📄 License & Citation

This system integrates multiple open-source projects. Please cite appropriately:

```bibtex
@article{ultralytics2023yolov8,
  title={YOLOv8: A State-of-the-Art Real-Time Object Detection Model},
  author={Jocher, Glenn},
  year={2023}
}

@article{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V.},
  journal={ICML},
  year={2019}
}
```

---

## 🤝 Contributing

This is a research/development project. For improvements:
1. Test thoroughly
2. Document changes
3. Follow code style guidelines
4. Add unit tests
5. Update documentation

---

## ⚠️ Important Disclaimer

**This system is for research and development purposes only. It is NOT approved for clinical use without proper regulatory approval, clinical validation, and medical professional oversight.**

**Users are responsible for:**
- Compliance with healthcare regulations (HIPAA, GDPR, etc.)
- Medical professional involvement in all clinical decisions
- Proper validation in their specific use case
- Understanding AI limitations and biases
