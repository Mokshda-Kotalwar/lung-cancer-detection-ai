 <div align="center">
🫁 LungAI Diagnostics – AI-Powered Lung Cancer Detection System



AI-Powered Clinical Decision Support System for Lung Cancer Detection using Deep Learning and Explainable AI

📖 Overview

LungAI Diagnostics is an end-to-end Artificial Intelligence powered web application that assists healthcare professionals in the early detection of lung cancer from CT scan images.

The project combines Deep Learning, Explainable AI (XAI), and a modern web application to provide fast, interpretable, and user-friendly diagnostic support.

Unlike traditional classification systems, LungAI Diagnostics not only predicts the likelihood of lung abnormalities but also explains the model's decisions using Grad-CAM visualizations, helping improve transparency and trust.

The application integrates:

🫁 CT Scan Analysis
🤖 Deep Learning Prediction
🔥 Explainable AI (Grad-CAM)
📊 Clinical Dashboard
📄 Automated PDF Report Generation
📚 Patient History Management
🔐 Secure Authentication
⚡ FastAPI Backend
🌐 Streamlit Frontend

Disclaimer: This project is developed for educational and research purposes only. It is not intended for clinical use or medical diagnosis.

🎯 Problem Statement

Lung cancer remains one of the leading causes of cancer-related deaths worldwide. Early diagnosis significantly improves survival rates, but manual interpretation of CT scans is time-consuming and requires experienced radiologists.

LungAI Diagnostics aims to assist healthcare professionals by providing an AI-assisted decision support system capable of:

Detecting lung abnormalities from CT scans
Predicting diagnostic categories
Providing confidence scores
Highlighting important image regions using Explainable AI
Maintaining patient records
Generating diagnostic reports automatically

🚀 Features

🫁 AI-Based Diagnosis
Upload Lung CT Scan Images
Automatic Image Preprocessing
DenseNet121-based Deep Learning Model
Multi-Class Classification
Confidence Score Prediction
Real-Time Inference

<img width="635" height="376" alt="image" src="https://github.com/user-attachments/assets/df5598f9-7cef-43bc-bd05-8e592c4647b8" />

🔥 Explainable AI
Grad-CAM Heatmap Generation
Model Attention Visualization
Region Highlighting
Improved Prediction Transparency

<img width="707" height="418" alt="image" src="https://github.com/user-attachments/assets/f05ce90b-41b3-4c45-9978-b9f682be4796" />


📊 Clinical Dashboard
Patient Statistics
Prediction Analytics
Scan History
Risk Distribution
Interactive Charts
Model Prediction Summary

<img width="958" height="437" alt="image" src="https://github.com/user-attachments/assets/6fb0fad3-6fb6-4075-8727-98d89e367f93" />


📄 Report Generation
Professional PDF Report
Patient Details
Prediction Summary
Confidence Scores
Grad-CAM Visualization
Downloadable Reports

📚 Patient Management
Patient History
Search Records
Previous Predictions
Report Storage
History Analytics

🔐 Authentication & Security
JWT Authentication
Secure Login
Password Hashing using bcrypt
Protected API Routes
User Authentication

<img width="412" height="433" alt="image" src="https://github.com/user-attachments/assets/93fbf024-3949-4259-b03a-1fbbe66de640" />


🏗️ System Architecture
                  Lung CT Scan
                        │
                        ▼
              Image Preprocessing
                        │
                        ▼
             DenseNet121 Classifier
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Prediction Result            Grad-CAM
          │                           │
          └─────────────┬─────────────┘
                        ▼
                Risk Assessment
                        │
                        ▼
              PDF Report Generation
                        │
                        ▼
             Dashboard & Patient History
             
🧠 AI Model Pipeline
Input CT Scan
      │
      ▼
Resize Image
      │
      ▼
Normalization
      │
      ▼
Contrast Enhancement
      │
      ▼
DenseNet121 Feature Extraction
      │
      ▼
Global Average Pooling
      │
      ▼
Fully Connected Layer
      │
      ▼
Softmax Classification
      │
      ▼
Prediction

🧠 Deep Learning Model

The prediction engine is built using DenseNet121, a convolutional neural network known for efficient feature reuse and strong performance in medical image analysis.

Why DenseNet121?
Transfer Learning
Efficient Feature Propagation
Reduced Overfitting
High Performance on Medical Images
Faster Training
Lightweight Compared to Larger CNNs

📂 Dataset

The model is trained and evaluated using the **LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)** dataset, one of the most widely used public datasets for lung nodule analysis and lung cancer research.

The dataset contains thoracic CT scans with expert radiologist annotations, making it suitable for developing and evaluating AI models for lung cancer detection.

### Dataset Information

- **Dataset:** LIDC-IDRI
- **Modality:** Computed Tomography (CT)
- **Image Format:** DICOM
- **Annotations:** Lung nodules annotated by up to four experienced thoracic radiologists
- **Application:** Lung Nodule Detection and Classification

### Target Classes

- Normal
- Benign
- Malignant

> **Note:** CT scans undergo preprocessing and image enhancement before being used for inference by the DenseNet121 classification model.

The preprocessing pipeline includes:

Image Loading
Image Resizing
Pixel Normalization
CLAHE Enhancement
Contrast Improvement
Tensor Conversion
🔬 Data Augmentation

To improve generalization, the following augmentation techniques are used:

Horizontal Flip
Random Rotation
Random Brightness
Random Contrast
Random Crop
Translation
Normalization

# 📊 Explainable AI (Grad-CAM)

Understanding why an AI model makes a particular prediction is essential in medical imaging. LungAI Diagnostics integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to provide visual explanations for model predictions.

Grad-CAM highlights the regions of the CT scan that contribute most to the model's decision, helping users better interpret the classification results.

### Features

- 🔥 Heatmap Generation
- 🫁 Lung Region Localization
- 🎯 Prediction Interpretation
- 📈 Improved Model Transparency
- 👨‍⚕️ Decision Support for Healthcare Professionals

> **Note:** Grad-CAM is intended to improve model interpretability and should not be considered a substitute for clinical expertise.

---

# 📊 Dashboard

The Streamlit dashboard provides an intuitive interface for interacting with the AI model and managing patient records.

## Dashboard Modules

### 🏠 Home Dashboard

- Overall Statistics
- Recent Predictions
- Prediction Distribution
- Risk Analytics
- Interactive Charts

---

### 📤 CT Scan Upload

- Upload CT Scan Images
- Automatic Preprocessing
- One-click Prediction
- Image Preview

---

### 🤖 Prediction Results

After inference, the dashboard displays:

- Predicted Class
- Confidence Score
- Risk Level
- Grad-CAM Visualization
- Clinical Summary

---

### 📄 AI Report

The generated PDF report includes:

- Patient Information
- Prediction Results
- Confidence Scores
- Grad-CAM Heatmap
- Date & Time
- AI Summary

---

### 📚 Patient History

The dashboard securely stores previous predictions for future reference.

Features include:

- Search Records
- View Previous Reports
- Download Reports
- Prediction Timeline

---

# ⚙️ Technology Stack

## Programming Language

- Python 3.10

---

## Deep Learning

- PyTorch
- Torchvision
- DenseNet121
- Grad-CAM

---

## Computer Vision

- OpenCV
- Pillow
- NumPy

---

## Backend

- FastAPI
- Uvicorn
- Pydantic

---

## Frontend

- Streamlit
- Plotly

---

## Database

- MongoDB
- Motor (Async MongoDB Driver)

---

## Authentication

- JWT Authentication
- OAuth2
- Passlib
- bcrypt

---

## Report Generation

- ReportLab

---

## Development Tools

- Git
- GitHub
- VS Code
- Docker (Optional)

---

# 📁 Project Structure

```text
lung-cancer-detection-ai/
│
├── app/                    # Streamlit Frontend
│   ├── components/
│   ├── pages/
│   ├── utils/
│   └── main.py
│
├── backend/                # FastAPI Backend
│   ├── api/
│   ├── core/
│   ├── database/
│   ├── models/
│   ├── services/
│   └── main.py
│
├── models/                 # Trained DenseNet121 Model
│
├── data/                   # Dataset & Preprocessing
│
├── outputs/                # Predictions & Reports
│
├── docs/                   # Documentation
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/lung-cancer-detection-ai.git

cd lung-cancer-detection-ai
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file in the project root.

Example:

```env
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=lung_cancer_db
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

---

# ▶️ Running the Project

## Start MongoDB

Ensure MongoDB is running locally or provide a cloud MongoDB URI.

---

## Start Backend

```bash
cd backend

uvicorn main:app --reload
```

Backend URL

```
http://127.0.0.1:8000
```

---

## Start Streamlit Dashboard

```bash
streamlit run app/main.py
```

Dashboard URL

```
http://localhost:8501
```

---

# 📡 API Endpoints

## Authentication

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/auth/signup` | Register User |
| POST | `/auth/login` | Login |

---

## Prediction

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/predict` | Predict CT Scan |
| POST | `/predict/report` | Generate PDF Report |

---

## History

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/history` | View Patient History |
| DELETE | `/history/{id}` | Delete Record |

---

## Health

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | API Health Check |

---

# 🖼️ Application Preview

> Add screenshots after uploading them to GitHub.

### Login Page

```
docs/images/login.png
```

### Dashboard

```
docs/images/dashboard.png
```

### Prediction Page

```
docs/images/prediction.png
```

### Grad-CAM Result

```
docs/images/gradcam.png
```

### PDF Report

```
docs/images/report.png
```

# 🔄 System Workflow

The following workflow illustrates how LungAI Diagnostics processes a CT scan from upload to report generation.

```text
                 User Login
                     │
                     ▼
             Upload CT Scan
                     │
                     ▼
          Image Preprocessing
                     │
                     ▼
      DenseNet121 Classification
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Prediction Result      Grad-CAM Heatmap
          │                     │
          └──────────┬──────────┘
                     ▼
            Risk Assessment
                     │
                     ▼
        PDF Report Generation
                     │
                     ▼
      Store Patient History (MongoDB)
                     │
                     ▼
      Dashboard Analytics & Visualization
```

---

# 📈 Model Evaluation

The primary objective of this project is to demonstrate an end-to-end AI-powered clinical decision support system for lung CT image analysis.

The current implementation focuses on:

- CT Scan Preprocessing
- DenseNet121-based Image Classification
- Explainable AI using Grad-CAM
- FastAPI Backend Services
- Streamlit Dashboard
- MongoDB Integration
- Automated PDF Report Generation

### Evaluation Metrics

The following evaluation metrics can be generated after training and testing the model:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC Score
- Confusion Matrix

> **Note:** The reported prediction should be interpreted as an AI-assisted result and **not as a confirmed medical diagnosis**.

---

# 🔒 Security Features

LungAI Diagnostics incorporates several security mechanisms to protect user information and application resources.

- JWT Authentication
- OAuth2 Authorization
- Password Hashing using bcrypt
- Protected API Endpoints
- Secure User Authentication
- Environment Variable Configuration
- Role-Based Access Support (Future Enhancement)

---

# 🚀 Future Enhancements

Several improvements can further enhance the capabilities of LungAI Diagnostics.

### AI & Deep Learning

- Multi-class Disease Classification
- Multi-Slice CT Analysis
- 3D Lung Nodule Detection
- Segmentation using U-Net
- Ensemble Deep Learning Models
- Vision Transformer Integration
- Improved Explainable AI Techniques

---

### Clinical Features

- DICOM Viewer Integration
- PACS Integration
- Electronic Health Record (EHR) Support
- Multi-Hospital Deployment
- Doctor Dashboard
- Patient Portal
- Clinical Decision Support

---

### Platform Features

- Cloud Deployment
- Docker & Kubernetes Support
- Email Report Delivery
- Multi-language Support
- Mobile Application
- Dark Mode Interface
- Real-time Notifications

---

# 💻 Development Highlights

This project demonstrates practical implementation of several modern AI and software engineering concepts.

### Machine Learning

- Transfer Learning
- DenseNet121
- Image Classification
- Data Augmentation
- Explainable AI

### Backend Development

- REST API Development
- FastAPI
- Authentication
- MongoDB Integration

### Frontend

- Streamlit Dashboard
- Interactive Charts
- Clinical Visualization
- Responsive Interface

### Software Engineering

- Modular Project Structure
- API Integration
- Version Control using Git
- Documentation
- Report Generation

---

# 🤝 Contributors

This project was collaboratively developed as an academic and research initiative.

## Contributors

**Tanaya Agrawal**

**Mokshda Kotalwar**

---

# 📚 References

The development of this project was inspired by published research in deep learning, medical imaging, and explainable artificial intelligence.

Key resources include:

- LIDC-IDRI Dataset
- PyTorch Documentation
- FastAPI Documentation
- Streamlit Documentation
- MongoDB Documentation
- Grad-CAM Research Paper

---

Special thanks to:

- Lung Image Database Consortium (LIDC-IDRI)
- PyTorch Community
- FastAPI Community
- Streamlit Community
- MongoDB Team
- OpenCV Community
- Scikit-learn Developers


---

# ⚠️ Medical Disclaimer

LungAI Diagnostics is an **AI-assisted clinical decision support system** developed solely for educational, research, and demonstration purposes.

The predictions generated by this application should **not** be interpreted as professional medical advice, diagnosis, or treatment recommendations.

Healthcare professionals should always rely on comprehensive clinical evaluation, radiological expertise, and established medical protocols before making diagnostic or treatment decisions.

The developers assume no responsibility for clinical decisions made using this software.

---

# ⭐ Support

If you found this project useful, please consider:

⭐ Starring this repository

🍴 Forking the repository

🛠️ Contributing improvements

📢 Sharing it with the AI and Healthcare community

---

# 📬 Contact

For questions, suggestions, or collaboration opportunities, feel free to connect through GitHub.

---

<div align="center">

## 🫁 LungAI Diagnostics

**AI-Powered Lung Cancer Detection using Deep Learning, Explainable AI, FastAPI, Streamlit, and MongoDB**

**Developed for Educational, Research, and Demonstration Purposes**

⭐ **If you like this project, don't forget to Star the Repository!**

</div>
