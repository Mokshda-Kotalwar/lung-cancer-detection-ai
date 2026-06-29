# 🫁 AI-Powered Lung Cancer Detection System (Research-Grade)

A comprehensive, production-ready AI system for analyzing medical images and assessing lung cancer risk using advanced deep learning models, explainable AI, and clinical decision support. This system has been updated to a highly scalable Microservices Architecture.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)

---

## ✨ Key Features

### 🤖 Advanced AI Models
- **YOLOv8**: Real-time nodule detection with high precision.
- **DenseNet121 & EfficientNet-B4**: Transfer learning networks customized with robust medical classification heads.
- **3D U-Net**: Volumetric lung nodule segmentation and masking.

### 📊 Comprehensive Analysis & Risk Stratification
- **Multi-factor Risk Scoring**: Combines neural network confidence, bounding box parameters, and patient priors (age, smoking history) into a cohesive Risk Score.
- **Role-Based Access Control**: Secure login mechanisms restricting endpoints by Doctor, Admin, or User roles.

### 🔍 Explainability (XAI)
- **Grad-CAM Visualization**: Uses PyTorch hooks to backpropagate class gradients to convolutional layers, producing visual heatmaps over the original scans to highlight regions of interest.

### 💻 Decoupled Microservices
- **FastAPI Backend**: Handles all heavy PyTorch inference processing and asynchronously connects to MongoDB Atlas.
- **Streamlit Frontend**: A lightweight, glassmorphism-styled medical dashboard acting purely as a presentation layer.

---

## 🚀 Quick Start (Docker Orchestration)

The easiest way to spin up the entire application stack is via `docker-compose`.

```bash
# Clone repository
git clone <repository-url>
cd lung-cancer-detection-ai

# Start the stack (MongoDB, FastAPI, Streamlit)
docker-compose up --build
```

- **Dashboard**: http://localhost:8501
- **API Swagger Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🏗️ Architecture & Documentation

- [**ARCHITECTURE.md**](ARCHITECTURE.md) - High-level system design and ML pipeline.
- [**API_DOCS.md**](API_DOCS.md) - Complete documentation for the FastAPI endpoints.
- [**RENDER_DEPLOY.md**](RENDER_DEPLOY.md) - Cloud hosting instructions for Render PaaS.

---

## 🧪 Testing

We use `pytest` for comprehensive backend and ML testing.

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
pytest tests/ -v
```

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for **RESEARCH AND DEVELOPMENT ONLY**.

- ❌ NOT approved for clinical diagnosis
- ❌ NOT a replacement for medical professionals
- ⚠️ Requires validation before clinical use
- ⚠️ Healthcare provider oversight mandatory
