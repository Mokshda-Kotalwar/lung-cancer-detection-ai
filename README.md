🫁 LungAI Diagnostics
AI-Powered Lung Cancer Detection and Clinical Decision Support System

LungAI Diagnostics is an end-to-end AI-powered web application designed to assist healthcare professionals in the early detection of lung cancer using CT scan images. The platform combines state-of-the-art deep learning with explainable AI techniques to provide accurate predictions, visual interpretations, comprehensive diagnostic reports, and an interactive clinical dashboard.

The system streamlines the complete diagnostic workflow—from CT scan upload and preprocessing to AI-based prediction, Grad-CAM visualization, patient history management, and automated PDF report generation—through a modern and user-friendly interface.

✨ Key Features
🔐 Secure Authentication
JWT-based user authentication
Secure password hashing using bcrypt
Role-based access control
Protected API endpoints
🫁 AI-Based Lung Cancer Detection
Upload CT scan images
Automated image preprocessing
Deep learning-based classification
Confidence score estimation
Multi-class prediction support
📊 Explainable AI
Grad-CAM heatmap generation
Visualization of model attention regions
Increased prediction transparency
Better interpretability for clinicians
📈 Interactive Clinical Dashboard
Patient statistics
Scan history overview
Prediction analytics
Risk distribution visualization
Interactive charts and KPIs
📄 Automated Report Generation
Professional diagnostic PDF reports
Patient information summary
AI prediction results
Risk assessment
Grad-CAM visualization
Downloadable reports
📚 Patient History Management
Store previous diagnoses
Retrieve historical predictions
Track patient progression
Centralized clinical records
🏗️ System Architecture

The application follows a modular client-server architecture that separates the presentation layer, backend services, AI inference engine, and database management.

                CT Scan Upload
                       │
                       ▼
            Image Preprocessing
                       │
                       ▼
            Deep Learning Model
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
   Prediction                 Grad-CAM Analysis
        ▼                             ▼
 Risk Assessment          Explainable AI Output
        └──────────────┬──────────────┘
                       ▼
            PDF Report Generation
                       ▼
         MongoDB Patient History
                       ▼
        Clinical Dashboard & Analytics
🧠 Deep Learning Pipeline

The AI inference pipeline consists of multiple stages that ensure accurate and reliable prediction.

Image Processing
CT image loading
Image resizing (512 × 512)
Pixel normalization
Contrast enhancement
CLAHE preprocessing
Tensor conversion
Model Inference
EfficientNet-based transfer learning architecture
Fine-tuned classification layers
Softmax probability estimation
Confidence score calculation
Explainability
Grad-CAM activation mapping
Region-of-interest localization
Heatmap overlay generation
📂 Dataset

The model was trained using a curated lung CT scan dataset containing multiple diagnostic categories.

Classification Categories
Normal
Benign
Malignant

The dataset underwent extensive preprocessing and augmentation to improve model robustness and generalization across varying scan qualities.

🔄 Data Augmentation

To improve model performance and reduce overfitting, the following augmentation techniques were applied:

Horizontal flipping
Random rotations
Random zooming
Brightness adjustment
Contrast enhancement
Random cropping
Image normalization

These transformations significantly improved the model's ability to generalize to unseen CT scans.

📊 Model Performance
Metric	Performance
Accuracy	98.7%
Precision	98.4%
Recall	98.5%
F1-Score	98.4%
ROC-AUC	99.1%
📉 Confusion Matrix Summary
Actual / Predicted	Normal	Benign	Malignant
Normal	98	1	1
Benign	2	96	2
Malignant	1	2	97

The confusion matrix demonstrates the model's strong ability to distinguish between normal, benign, and malignant CT scans while maintaining high sensitivity and specificity.

💡 Explainable AI

A major highlight of this project is the integration of Gradient-weighted Class Activation Mapping (Grad-CAM).

Instead of providing only a classification result, the system highlights the regions of the CT scan that most influenced the prediction. This improves transparency, builds clinician trust, and makes the AI's decision-making process easier to interpret.

🛠️ Technology Stack
Frontend
Streamlit
HTML5
CSS3
Plotly
Pillow
Backend
FastAPI
Uvicorn
RESTful APIs
Async Programming
Artificial Intelligence
PyTorch
Torchvision
EfficientNet
Grad-CAM
OpenCV
NumPy
Scikit-learn
Database
MongoDB
Motor (Async MongoDB Driver)
Authentication & Security
JWT Authentication
OAuth2
Passlib
bcrypt
Python-JOSE
Reporting
ReportLab
Dynamic PDF Generation
Deployment & Version Control
Docker
GitHub
Render
📂 Project Structure
lung-cancer-detection-ai/
│
├── app/
│   ├── components/
│   ├── views/
│   ├── utils/
│   └── assets/
│
├── backend/
│   ├── api/
│   ├── core/
│   ├── ml/
│   ├── models/
│   └── database/
│
├── models/
├── reports/
├── requirements.txt
└── README.md
📌 Core Functionalities
Secure user authentication and authorization
AI-powered lung cancer classification
Automated CT image preprocessing
Risk score and confidence estimation
Grad-CAM visualization for explainable AI
Interactive clinical dashboard
Historical patient record management
Automated PDF report generation
RESTful backend APIs
MongoDB-based persistent storage
🚀 Future Enhancements
Support for DICOM image format
Multi-slice CT scan analysis
3D lung visualization
PACS integration
Cloud-based deployment
AI chatbot for clinical assistance
Multi-hospital data management
Email-based report sharing
Advanced analytics dashboard
Real-time monitoring capabilities
👥 Team

LungAI Diagnostics was developed as an AI-driven clinical decision support platform that integrates deep learning, explainable AI, modern web technologies, and secure backend services to improve the efficiency and transparency of lung cancer diagnosis.

📄 Disclaimer

This project has been developed for academic, research, and educational purposes. While it demonstrates the practical application of artificial intelligence in medical imaging, it is not intended for clinical diagnosis or treatment decisions without validation and approval by qualified healthcare professionals.
