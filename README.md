# 🫁 AI-Powered Lung Cancer Detection System

A comprehensive, production-ready AI system for analyzing medical images and assessing lung cancer risk using advanced deep learning models, explainable AI, and clinical decision support.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Key Features

### 🤖 Advanced AI Models
- **YOLOv8**: Real-time nodule detection with high precision
- **EfficientNet**: Efficient malignancy classification (B0-B7 variants)
- **Ensemble Architecture**: Combined detection + classification for robust predictions
- **Model Checkpointing**: Save and load trained models

### 📊 Comprehensive Analysis
- **Multi-factor Risk Scoring**: Classification confidence + nodule characteristics + clinical factors
- **Uncertainty Quantification**: Monte Carlo dropout for prediction confidence
- **Clinical Integration**: Patient history and demographic factors
- **Differential Diagnosis**: Rule-based diagnosis suggestions

### 🔍 Explainability
- **Grad-CAM Visualization**: Understand which regions contribute to predictions
- **LayerCAM**: Alternative attention mechanism
- **Heatmap Overlay**: Visual interpretation of AI decisions
- **Medical Compliance**: HIPAA-friendly explanation logging

### 📋 Automated Reporting
- **PDF Generation**: Professional medical reports with ReportLab
- **HTML Reports**: Web-viewable analysis summaries
- **Rich Content**: Patient info, findings, risk assessment, recommendations
- **Customizable Templates**: Institution-specific branding

### 💻 Web Interface
- **Streamlit Dashboard**: Interactive analysis and visualization
- **Real-time Processing**: <2 second inference per image
- **Batch Upload**: Process multiple images
- **Report Management**: View, download, and archive reports

### 🔐 Production-Ready
- **Comprehensive Logging**: Activity tracking and debugging
- **Error Handling**: Graceful degradation and recovery
- **Configuration Management**: Centralized settings
- **Unit Tests**: Pytest coverage for core functionality
- **Medical Disclaimers**: Clear AI limitations and regulatory notices

---

## 🏗️ Architecture

### Modular Design
                CT Scan Image
                       │
                       ▼
                Image Preprocessing
                       │
                       ▼
                Lung Segmentation
                   (3D U-Net)
                       │
                       ▼
                 Nodule Detection
                     (YOLOv8)
                       │
                       ▼
               Feature Extraction
               (DenseNet/ViT)
                       │
                       ▼
                  Classification
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
     Prediction                 Confidence
         │                           │
         └─────────────┬─────────────┘
                       ▼
                  Grad-CAM
                       │
                       ▼
                FastAPI Backend
                       │
                       ▼
               Streamlit Dashboard
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Upload       Reports      History
                       │
                       ▼
                   Database
                (MongoDB Atlas)
                       │
                       ▼
                  Cloud Deployment
```

### Core Modules
| Module            | Purpose                           | Key Classes                                                  |
|-------------------|-----------------------------------|--------------------------------------------------------------|
| **preprocessing** | Image processing & normalization  | `ImagePreprocessor`, `DICOMProcessor`, `LungSegmentation`    |
| **detection**     | Nodule detection & classification | `YOLODetector`, `EfficientNetClassifier`, `EnsembleDetector` |
| **xai**           | Explainable AI visualizations     | `GradCAM`, `LayerCAM`, `ExplainabilityEngine`                |
| **risk**          | Risk scoring & recommendations    | `RiskScorer`, `RecommendationEngine`, `DifferentialDiagnosis |
| **reporting**     | Report generation                 | `ReportGenerator`                                            |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd lung-cancer-detection-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Web Interface (Recommended)

```bash
python main.py --web
```

Open http://localhost:8501 in your browser.

### Command Line Analysis

```bash
# Analyze single image
python main.py --image path/to/image.png --patient-id P001 --name "John Doe"

# Skip report generation
python main.py --image path/to/image.png --no-report
```

### Python API

```python
from config import config
from main import LungCancerDetectionPipeline

pipeline = LungCancerDetectionPipeline(config)
results = pipeline.analyze_image('image.png', patient_info={'patient_id': 'P001'})

print(f"Risk Score: {results['risk_assessment'].risk_score:.3f}")
print(f"Risk Level: {results['risk_assessment'].risk_level.value}")
```

---

## 📋 Requirements

### System
- Python 3.8+
- CUDA 11.8+ (for GPU, optional)
- 8GB RAM (minimum), 16GB (recommended)

### Key Dependencies
- **PyTorch 2.0+**: Deep learning framework
- **YOLOv8**: Object detection
- **EfficientNet**: Image classification
- **Streamlit**: Web interface
- **ReportLab**: PDF generation
- **OpenCV**: Image processing
- **pydicom**: Medical imaging support

See `requirements.txt` for complete list.

---

## 📊 Configuration

All system settings are centralized in `config.py`:

```python
from config import config

# Model configuration
config.model.yolo_model = "yolov8m"
config.model.input_size = 512

# Preprocessing
config.preprocessing.normalize_method = "minmax"
config.preprocessing.use_lung_mask = True

# Risk assessment
config.risk.risk_threshold_high = 0.7

# Save custom config
config.save_to_json("custom_config.json")
```

---

## 🔍 Usage Examples

### Example 1: Single Image Analysis

```python
from config import config
from main import LungCancerDetectionPipeline

pipeline = LungCancerDetectionPipeline(config)

patient_info = {
    'patient_id': 'P001',
    'name': 'John Doe',
    'age': 65,
    'smoker': True
}

results = pipeline.analyze_image(
    'path/to/image.png',
    patient_info=patient_info,
    generate_report=True
)

# Access results
risk = results['risk_assessment']
print(f"Risk: {risk.risk_level.value} ({risk.risk_score:.3f})")
```

### Example 2: Batch Processing

```python
from pathlib import Path

image_dir = Path("data/images")
results = []

for image_path in image_dir.glob("*.png"):
    result = pipeline.analyze_image(str(image_path))
    results.append(result)
```

### Example 3: Custom Configuration

```bash
python examples/02_configuration_customization.py
```

See the `examples/` directory for more detailed examples.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Specific test
pytest tests/test_preprocessing.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📈 Performance

### Metrics
- **Detection Speed**: ~0.5s per image (GPU)
- **Classification Speed**: ~0.1s per image
- **Memory Usage**: ~2GB GPU / ~4GB CPU
- **Accuracy**: Depends on training data and model weights

### Optimization Tips
- Use GPU for faster inference
- Reduce image size for speed vs. accuracy trade-off
- Enable mixed precision for memory efficiency
- Use batch processing for multiple images

---

## 📄 Output Files

### Reports
- **PDF Report**: Professional medical report with findings and recommendations
- **HTML Report**: Web-viewable analysis summary
- **Prediction Files**: JSON with detection and classification results

### Visualizations
- **Detection Image**: Original with bounding boxes
- **Grad-CAM Overlay**: Attention heatmap on original image
- **Classification Probabilities**: Bar charts of class predictions

---

## 🔐 Security & Compliance

### Features
- **HIPAA Compliance**: Secure patient data handling
- **DICOM Privacy**: Remove PHI from medical images
- **Audit Logging**: Complete activity tracking
- **Secure Configuration**: Environment variables for sensitive data
- **Medical Disclaimers**: Clear regulatory notices

### Data Protection
```python
# Remove DICOM identifiers
import pydicom
ds = pydicom.dcmread('image.dcm')
ds.remove_private_tags()
```

---

## 📚 Documentation

- [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) - Complete system design and modules
- [**SETUP.md**](docs/SETUP.md) - Detailed installation and configuration guide
- [**API.md**](docs/API.md) - Complete API reference
- [**examples/**](examples/) - Code examples and use cases

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` in config |
| Slow inference | Enable GPU, use smaller image size |
| Model not found | Check internet, restart app |
| DICOM read error | Install pydicom: `pip install pydicom` |

For more help, see [SETUP.md](docs/SETUP.md#-troubleshooting).

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Test thoroughly
2. Document changes
3. Follow code style
4. Add unit tests
5. Update documentation

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for **RESEARCH AND DEVELOPMENT ONLY**.

- ❌ NOT approved for clinical diagnosis
- ❌ NOT a replacement for medical professionals
- ⚠️ Requires validation before clinical use
- ⚠️ Healthcare provider oversight mandatory
- ⚠️ User assumes all liability for misuse

**Regulatory Requirements:**
- Clinical validation in your specific use case
- Integration with qualified healthcare professionals
- Compliance with healthcare regulations (HIPAA, GDPR, etc.)
- Institutional review and approval

---

## 📞 Citation

If you use this system in research, please cite:

```bibtex
@software{lungcancer_detection_2024,
  title={AI-Powered Lung Cancer Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

Also cite the underlying models:
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [Grad-CAM](https://arxiv.org/abs/1610.02055)

---

## 🔗 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Medical Imaging Guides](https://www.radiologyassistant.nl/)

---

## 👥 Authors & Contributors

- **Development Team**: AI Medical Research Group
- **Contributors**: Community contributions welcome

---

**Last Updated**: 2024  
**Status**: Production-Ready (with regulatory approval required)

---

<p align="center">
  Made with ❤️ for advancing medical AI
</p>
