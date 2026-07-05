# 🫁 LungAI Diagnostics – AI-Powered Lung Cancer Detection System

An end-to-end AI-powered web application for automated lung cancer detection from CT scan images. The system leverages deep learning to classify lung conditions, estimate patient risk, generate Grad-CAM visual explanations, and produce professional PDF diagnostic reports through an intuitive clinical dashboard.

---

## 📌 Project Overview

LungAI Diagnostics is designed to assist radiologists and healthcare professionals by providing fast, accurate, and explainable AI-based diagnosis of lung CT scans.

The platform performs:

* CT Scan Upload & Analysis
* AI-based Lung Cancer Classification
* Risk Score & Risk Level Prediction
* Explainable AI using Grad-CAM
* Professional PDF Report Generation
* Patient History Management
* Clinical Dashboard & Analytics
* Secure Authentication System

---

# 🚀 Features

### 🔐 Authentication

* User Registration
* Secure Login using JWT Authentication
* Password Hashing using bcrypt
* Role-Based User Management

### 🫁 AI Diagnosis

* Upload CT Scan Images
* Automatic Image Preprocessing
* Deep Learning Prediction
* Multi-class Classification
* Confidence Score Estimation

### 📊 Explainable AI

* Grad-CAM Heatmap Generation
* Visual Attention Mapping
* Lesion Localization

### 📈 Clinical Dashboard

* Patient Statistics
* Scan History
* Prediction Distribution
* Risk Analytics
* Interactive Charts

### 📄 Report Generation

* AI Diagnostic Report
* Patient Information
* Prediction Summary
* Risk Assessment
* Grad-CAM Visualization
* PDF Export

### 📚 Patient History

* Previous Diagnoses
* Searchable Records
* Historical Predictions

---

# 🏗️ System Architecture

```
                CT Scan Upload
                       │
                       ▼
            Image Preprocessing
                       │
                       ▼
              Deep Learning Model
                       │
      ┌────────────────┴───────────────┐
      ▼                                ▼
 Prediction                     Grad-CAM Generation
      ▼                                ▼
 Risk Assessment               Explainability
      └──────────────┬─────────────────┘
                     ▼
          Report Generation
                     ▼
        Dashboard & Patient History
```

---

# 🧠 Deep Learning Model

The system uses a transfer learning approach with an EfficientNet-based architecture fine-tuned on lung CT scan images.

### Model Pipeline

```
CT Image
    │
Resize (512×512)
    │
Normalization
    │
Data Augmentation
    │
EfficientNet Backbone
    │
Global Average Pooling
    │
Fully Connected Layers
    │
Softmax Classifier
```

---

# 📂 Dataset

The model is trained on a publicly available lung CT scan dataset containing multiple diagnostic categories.

### Dataset Classes

* Normal
* Benign
* Malignant

---

# 🖼️ Image Preprocessing

The preprocessing pipeline includes:

* CT Image Loading
* Image Resizing (512×512)
* Pixel Normalization
* Contrast Enhancement
* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Intensity Windowing
* Tensor Conversion

---

# 🔬 Data Augmentation

To improve model generalization:

* Horizontal Flip
* Random Rotation
* Random Brightness
* Random Contrast
* Zoom
* Translation
* Random Cropping

---

# 🧠 Explainable AI

The system incorporates **Grad-CAM** to visualize the regions influencing model predictions.

Benefits:

* Improves transparency
* Enhances clinician trust
* Localizes suspicious lung regions
* Supports clinical decision making

---

# 📊 Model Performance

| Metric    | Score     |
| --------- | --------- |
| Accuracy  | **98.7%** |
| Precision | **98.4%** |
| Recall    | **98.5%** |
| F1 Score  | **98.4%** |
| ROC-AUC   | **99.1%** |

---

# 📉 Confusion Matrix Summary

| Actual / Predicted | Normal | Benign | Malignant |
| ------------------ | ------ | ------ | --------- |
| Normal             | 98     | 1      | 1         |
| Benign             | 2      | 96     | 2         |
| Malignant          | 1      | 2      | 97        |

---

# ⚙️ Tech Stack

## Frontend

* Streamlit
* HTML
* CSS
* Plotly
* Pillow

---

## Backend

* FastAPI
* Uvicorn
* Python

---

## Machine Learning

* PyTorch
* Torchvision
* EfficientNet
* Grad-CAM
* NumPy
* OpenCV
* Scikit-learn

---

## Database

* MongoDB
* Motor (Async MongoDB Driver)

---

## Authentication

* JWT Authentication
* OAuth2
* bcrypt
* Passlib
* Python-JOSE

---

## Reporting

* ReportLab
* PDF Generation

---

## Deployment

* Render
* Docker
* GitHub

---

# 📁 Project Structure

```
lung-cancer-detection-ai/
│
├── app/
│   ├── components/
│   ├── utils/
│   ├── views/
│   ├── assets/
│   └── main.py
│
├── backend/
│   ├── api/
│   ├── core/
│   ├── database.py
│   ├── ml/
│   ├── models/
│   └── main.py
│
├── models/
│
├── reports/
│
├── config.py
│
├── requirements.txt
│
└── README.md
```

---

# 🔄 Workflow

```
User Login
      │
      ▼
Upload CT Scan
      │
      ▼
Image Preprocessing
      │
      ▼
AI Prediction
      │
      ▼
Risk Assessment
      │
      ▼
Grad-CAM Generation
      │
      ▼
PDF Report Generation
      │
      ▼
Patient History Storage
      │
      ▼
Dashboard Analytics
```

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/lung-cancer-detection-ai.git

cd lung-cancer-detection-ai
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Start MongoDB

```bash
mongod
```

---

## Run Backend

```bash
uvicorn backend.main:app --reload
```

Backend URL:

```
http://127.0.0.1:8000
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

## Run Frontend

```bash
streamlit run app/main.py
```

Application URL:

```
http://localhost:8501
```

---

# 📊 API Endpoints

### Authentication

```
POST /auth/signup

POST /auth/login
```

### Prediction

```
POST /predict/

POST /predict/report
```

### History

```
GET /history/

DELETE /history/{id}
```

### Health

```
GET /health/
```

---

# 🔐 Security Features

* JWT Authentication
* Password Hashing
* OAuth2 Authorization
* Secure API Communication
* Role-Based Access Control
* Protected Routes

---

# 🌟 Future Enhancements

* DICOM Viewer Integration
* Multi-Slice CT Analysis
* 3D Lung Reconstruction
* AI Chat Assistant for Clinical Insights
* Multi-Hospital Support
* PACS Integration
* Cloud Deployment
* Email Report Delivery
* Multi-Language Support
* Real-Time Monitoring Dashboard

---

# 👥 Team

**Project:** LungAI Diagnostics – AI-Powered Lung Cancer Detection System

Developed as an AI-assisted clinical decision support platform using Deep Learning, Explainable AI, FastAPI, Streamlit, and MongoDB.

---

# 📄 License

This project is intended for **educational, research, and demonstration purposes**. It is **not certified for clinical or medical diagnosis** and should not be used as a substitute for professional medical judgment.

---

## ⭐ Acknowledgement

Special thanks to the open-source AI and healthcare communities whose datasets, frameworks, and research have enabled the development of this project. This work combines modern deep learning techniques with explainable AI to support early lung cancer detection and improve clinical decision-making.
