"""
Streamlit Web Application for Lung Cancer Detection System
A state-of-the-art clinical dashboard for medical image analysis, classification, and explainability.
Author: Senior AI Engineer & UX/UI Designer
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from app.db import db
from config import config, OUTPUTS_DIR, MODELS_DIR
from src.preprocessing import ImagePreprocessor
from src.detection import YOLODetector, EfficientNetClassifier, EnsembleDetector, ClassificationResult
from src.models.classifier import DenseNetClassifier
from src.xai import GradCAM
from src.risk import RiskScorer, RecommendationEngine
from src.reporting import ReportGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LungAI Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Glassmorphism CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background & Text Color */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .glass-header {
        font-size: 2.2em;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 0.85em;
        font-weight: 600;
        text-align: center;
    }
    
    .status-active {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Risk levels color coding */
    .risk-badge {
        font-size: 1.1em;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 8px;
        text-transform: uppercase;
        display: inline-block;
        margin-top: 8px;
    }
    
    .risk-low { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
    .risk-intermediate { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
    .risk-high { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }
    .risk-critical { background-color: rgba(220, 38, 38, 0.35); color: #fca5a5; border: 2px solid #ef4444; font-weight: 900; }
    
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #60a5fa;
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar glass effect */
    section[data-testid="stSidebar"] {
        background-color: #070a13;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_cached():
    """Load AI models (cached for performance)"""
    try:
        ensemble = EnsembleDetector(config)
        preprocessor = ImagePreprocessor(config)
        return ensemble, preprocessor
    except Exception as e:
        logger.error(f"Error loading ensemble: {e}")
        return None, None


@st.cache_resource
def load_densenet_classifier():
    """Load pretrained DenseNet121 model weights if available."""
    model = DenseNetClassifier(num_classes=3, pretrained=True)
    checkpoint_dir = MODELS_DIR / "checkpoints"
    
    # Try custom path first, then test path
    best_path = checkpoint_dir / "best_densenet.pth"
    test_path = checkpoint_dir / "test_densenet.pth"
    
    loaded = False
    for path in [best_path, test_path]:
        if path.exists():
            try:
                model.load_state_dict(torch.load(path, map_location="cpu"))
                logger.info(f"Loaded DenseNet121 weights from {path}")
                loaded = True
                break
            except Exception as e:
                logger.error(f"Failed to load checkpoint {path}: {e}")
                
    if not loaded:
        logger.warning("No classification model weights found. Initializing with default weights.")
        
    model.eval()
    return model


def ensure_3_channels(image_tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has 3 channels for backbones expecting RGB inputs."""
    if image_tensor.dim() == 3:  # (C, H, W)
        if image_tensor.shape[0] == 1:
            image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], dim=0)
    elif image_tensor.dim() == 4:  # (B, C, H, W)
        if image_tensor.shape[1] == 1:
            image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], dim=1)
    return image_tensor


def classify_densenet(model, image_tensor: torch.Tensor) -> ClassificationResult:
    """Classify image using DenseNet121 and format results."""
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Ensure input shape is correct
    image_tensor = ensure_3_channels(image_tensor)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, class_id = torch.max(probabilities[0], 0)
        
    class_names = ["Benign", "Malignant", "Uncertain"]
    prob_dict = {
        class_names[i]: probabilities[0, i].item()
        for i in range(len(class_names))
    }
    
    processing_time = time.time() - start_time
    
    return ClassificationResult(
        class_id=int(class_id),
        class_name=class_names[int(class_id)],
        confidence=float(confidence),
        probabilities=prob_dict,
        processing_time=processing_time
    )


def run_pipeline(image: np.ndarray, model_choice: str, ensemble: EnsembleDetector, 
                 preprocessor: ImagePreprocessor, densenet_model: DenseNetClassifier):
    """Run preprocessing, detection, and selected classification model."""
    start_time = time.time()
    
    # 1. Preprocess input image
    processed_image = preprocessor.preprocess(image, apply_aug=False)
    
    # 2. Run Nodule Detection (YOLOv8)
    detection_result = ensemble.detector.detect(
        image,
        conf_threshold=config.model.yolo_conf_threshold,
        iou_threshold=config.model.yolo_iou_threshold
    )
    
    # 3. Run Classification on detected nodules (or full image if none detected)
    classification_result = None
    if model_choice == "DenseNet121 (Research-grade)":
        classification_result = classify_densenet(densenet_model, processed_image)
    else:
        # Use Standard EfficientNet
        image_tensor = processed_image.unsqueeze(0) if processed_image.dim() == 3 else processed_image
        classification_result = ensemble.classifier.classify(image_tensor)
        
    processing_time = time.time() - start_time
    return processed_image, detection_result, classification_result, processing_time


def main():
    # Sidebar Logo and Model Settings
    with st.sidebar:
        st.markdown("<div style='text-align: center;'><h2 style='color: #60a5fa; margin-bottom: 0;'>🫁 LungAI</h2><p style='color: #64748b; font-size: 0.9em; margin-top:0;'>Medical Diagnosis System</p></div>", unsafe_allow_html=True)
        st.divider()
        
        st.subheader("⚙️ Analysis Settings")
        model_choice = st.selectbox(
            "Classification Model",
            ["DenseNet121 (Research-grade)", "EfficientNet (Standard)"]
        )
        
        # Database Connection Check
        db_ok, db_name = db.health_check()
        if "MongoDB" in db_name:
            st.markdown(f"<div class='status-badge status-active'>● Connected: {db_name}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-badge status-warning'>▲ Backup: {db_name}</div>", unsafe_allow_html=True)
            
        st.divider()
        
        # Medical Disclaimer Sidebar Expander
        with st.expander("⚠️ Medical Disclaimer", expanded=False):
            st.warning("""
            This AI system is designed for clinical decision support.
            It should not be used as a standalone diagnostic tool.
            
            - All findings must be validated by a certified radiologist/oncologist.
            - Patient management decisions are the responsibility of licensed healthcare providers.
            """)
            
    # Load all models
    ensemble, preprocessor = load_models_cached()
    densenet_model = load_densenet_classifier()
    
    if ensemble is None or preprocessor is None:
        st.error("Failed to load backend AI models. Please verify installation.")
        return

    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Scan Upload", "Diagnosis Results", "Analytics & Calibration", "Patient History Log"],
        icons=["house", "cloud-upload", "heart-pulse", "bar-chart", "clock-history"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#070a13", "border": "1px solid rgba(255,255,255,0.05)"},
            "icon": {"color": "#60a5fa", "font-size": "15px"},
            "nav-link": {"font-size": "15px", "text-align": "center", "margin": "0px", "--hover-color": "#1e293b", "color": "#94a3b8"},
            "nav-link-selected": {"background-color": "#1e40af", "color": "#ffffff", "font-weight": "600"},
        }
    )

    # Initialize session state for persistent patient data between views
    if "patient_data" not in st.session_state:
        st.session_state["patient_data"] = {}
    if "results" not in st.session_state:
        st.session_state["results"] = None

    # Page 1: Home
    if selected == "Home":
        st.markdown("<h1 class='glass-header'>🫁 AI-Powered Lung Cancer Detection System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1em;'>Advanced computer vision and deep learning pipeline for nodule detection, risk stratification, and patient management.</p>", unsafe_allow_html=True)
        st.spacer = st.empty()
        
        # Hero Stats
        history_records = db.get_history(limit=500)
        total_scans = len(history_records)
        high_risk_scans = sum(1 for r in history_records if r.get("risk_level") in ["High", "Critical"])
        avg_nodules = np.mean([r.get("nodules_detected", 0) for r in history_records]) if total_scans > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-label'>Total Scans Analyzed</div>
                    <div class='metric-value'>{total_scans}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-label'>High/Critical Risk Findings</div>
                    <div class='metric-value' style='color: #f87171;'>{high_risk_scans}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-label'>Average Nodules Per Scan</div>
                    <div class='metric-value' style='color: #34d399;'>{avg_nodules:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.subheader("🔍 Integrated Clinical Pipeline Architecture")
        st.markdown("""
        The system employs a multi-tiered research-level pipeline:
        1. **Segmentation Engine**: Preprocesses DICOM/PNG slices and masks lung regions using a **3D U-Net** structure.
        2. **Nodule Detection Engine**: Employs an optimized **YOLOv8** model to detect localized bounding boxes.
        3. **Classification Network**: Integrates a transfer-learned **DenseNet121** model to categorize nodules as **Benign**, **Malignant**, or **Uncertain**.
        4. **Explainability Engine (Grad-CAM)**: Backpropagates class gradients to compute activation maps of regions of interest.
        """)
        
    # Page 2: Scan Upload
    elif selected == "Scan Upload":
        st.subheader("📋 Enter Patient Metadata")
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("Patient ID", value=st.session_state["patient_data"].get("patient_id", f"PAT_{int(time.time()) % 100000}"))
            name = st.text_input("Full Name", value=st.session_state["patient_data"].get("name", "John Doe"))
        with col2:
            age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state["patient_data"].get("age", 45))
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state["patient_data"].get("gender", "Male")))
        with col3:
            smoker = st.checkbox("Smoking History", value=st.session_state["patient_data"].get("smoker", False))
            study_date = st.date_input("Study Date", value=datetime.strptime(st.session_state["patient_data"].get("study_date", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d"))
            
        st.session_state["patient_data"] = {
            "patient_id": patient_id,
            "name": name,
            "age": age,
            "gender": gender,
            "smoker": smoker,
            "study_date": study_date.strftime("%Y-%m-%d")
        }
        
        st.subheader("📤 Upload Medical Image")
        uploaded_file = st.file_uploader(
            "Choose a CT scan slice (DICOM, PNG, JPG, JPEG)",
            type=["dcm", "png", "jpg", "jpeg"]
        )
        
        if uploaded_file is not None:
            st.info("Medical scan uploaded successfully. Adjust preprocessing settings if required.")
            
            # Read image
            image = None
            if uploaded_file.name.endswith(".dcm"):
                try:
                    import pydicom
                    ds = pydicom.dcmread(uploaded_file)
                    image = ds.pixel_array.astype(np.float32)
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                except Exception as e:
                    st.error(f"Error parsing DICOM: {e}")
                    return
            else:
                image = PILImage.open(uploaded_file)
                image = np.array(image)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Save uploaded image to temp directory for processing
            temp_path = OUTPUTS_DIR / "uploads"
            temp_path.mkdir(parents=True, exist_ok=True)
            img_file_path = temp_path / f"temp_{patient_id}.png"
            cv2.imwrite(str(img_file_path), image)
            
            st.session_state["patient_data"]["temp_image_path"] = str(img_file_path)
            
            # Run Analysis Button
            st.divider()
            if st.button("🚀 Run Diagnostic Pipeline", use_container_width=True):
                with st.spinner("Processing scans and running diagnostic classifiers..."):
                    processed, det_res, class_res, runtime = run_pipeline(
                        image=image,
                        model_choice=model_choice,
                        ensemble=ensemble,
                        preprocessor=preprocessor,
                        densenet_model=densenet_model
                    )
                    
                    # Compute risk scoring
                    risk_scorer = RiskScorer(config)
                    risk_assessment = risk_scorer.calculate_risk(
                        classification_confidence=class_res.confidence if class_res else 0.0,
                        detection_count=len(det_res.boxes) if det_res else 0,
                        detection_size=12.5 if len(det_res.boxes) > 0 else 0.0, # Nodule size placeholder
                        detection_confidence=det_res.confidences.mean() if det_res and len(det_res.confidences) > 0 else 0.0,
                        clinical_features={"age": age / 100.0, "smoking_pack_years": 0.8 if smoker else 0.0}
                    )
                    
                    # Generate clinical recommendations
                    rec_engine = RecommendationEngine(config)
                    recommendations = rec_engine.generate_recommendations(
                        risk_assessment, 
                        patient_info={"age": age, "smoker": smoker}
                    )
                    
                    # Save results in session state
                    st.session_state["results"] = {
                        "detection_result": det_res,
                        "classification_result": class_res,
                        "risk_assessment": risk_assessment,
                        "recommendations": recommendations,
                        "runtime": runtime,
                        "original_image": image,
                        "processed_image": processed.cpu().squeeze().numpy() if isinstance(processed, torch.Tensor) else processed
                    }
                    
                    # Save to Database History
                    db_record = {
                        "patient_id": patient_id,
                        "name": name,
                        "age": age,
                        "gender": gender,
                        "smoker": smoker,
                        "study_date": st.session_state["patient_data"]["study_date"],
                        "nodules_detected": len(det_res.boxes) if det_res else 0,
                        "detection_confidence": float(det_res.confidences.mean()) if det_res and len(det_res.confidences) > 0 else 0.0,
                        "classification": class_res.class_name if class_res else "None",
                        "classification_confidence": class_res.confidence if class_res else 0.0,
                        "probabilities": class_res.probabilities if class_res else {},
                        "risk_score": risk_assessment.risk_score,
                        "risk_level": risk_assessment.risk_level.value,
                        "recommendation": risk_assessment.recommendation,
                        "image_path": str(img_file_path),
                        "report_path": ""
                    }
                    db.save_record(db_record)
                    
                    st.success(f"Diagnosis completed successfully in {runtime:.2f}s! Proceed to the 'Diagnosis Results' tab.")
                    
    # Page 3: Diagnosis Results
    elif selected == "Diagnosis Results":
        res = st.session_state.get("results")
        if res is None:
            st.warning("Please upload a scan and run the diagnostic pipeline first in the 'Scan Upload' tab.")
            return
            
        p_data = st.session_state["patient_data"]
        
        st.markdown(f"<h2>Patient Diagnostic Report: {p_data.get('name')} ({p_data.get('patient_id')})</h2>", unsafe_allow_html=True)
        st.write(f"**Study Date:** {p_data.get('study_date')} | **Age:** {p_data.get('age')} | **Gender:** {p_data.get('gender')} | **Smoker:** {'Yes' if p_data.get('smoker') else 'No'}")
        
        st.divider()
        
        # Overview Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            det_count = len(res["detection_result"].boxes) if res["detection_result"] else 0
            st.markdown(f"""
                <div class='glass-card'>
                    <div class='metric-label'>Nodules Detected</div>
                    <div class='metric-value'>{det_count}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            pred_class = res["classification_result"].class_name if res["classification_result"] else "N/A"
            pred_conf = res["classification_result"].confidence * 100 if res["classification_result"] else 0.0
            st.markdown(f"""
                <div class='glass-card'>
                    <div class='metric-label'>Classifier Predicton</div>
                    <div class='metric-value'>{pred_class} <span style='font-size:0.55em; color:#a855f7;'>({pred_conf:.1f}%)</span></div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            r_level = res["risk_assessment"].risk_level.value
            r_class = f"risk-{r_level.lower()}"
            st.markdown(f"""
                <div class='glass-card'>
                    <div class='metric-label'>Stratified Risk Level</div>
                    <div class='risk-badge {r_class}'>{r_level}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        # Visualizations
        st.subheader("🖼️ Nodule Detection & Classification Overlay")
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.image(res["original_image"], caption="Original Grayscale CT Slice", use_column_width=True)
            
        with col_img2:
            # Draw detections
            detection_image = ensemble.detector.draw_detections(res["original_image"], res["detection_result"])
            st.image(detection_image, caption="YOLOv8 Detected Lung Nodules", use_column_width=True)
            
        st.divider()
        
        # Classification probabilities & Explainability
        col_prob, col_grad = st.columns(2)
        
        with col_prob:
            st.subheader("📊 Malignancy Probability Breakdown")
            if res["classification_result"]:
                probs = res["classification_result"].probabilities
                df_probs = pd.DataFrame({
                    "Diagnosis": list(probs.keys()),
                    "Probability (%)": [val * 100 for val in probs.values()]
                })
                st.bar_chart(df_probs.set_index("Diagnosis"), use_container_width=True)
                for c_name, val in probs.items():
                    st.write(f"**{c_name}**: {val*100:.2f}%")
            else:
                st.info("No nodules detected to run classification models on.")
                
        with col_grad:
            st.subheader("💡 Grad-CAM Visual Explainability Map")
            if res["classification_result"]:
                # Load GradCAM dynamically for explainability
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Choose correct model and layer
                if model_choice == "DenseNet121 (Research-grade)":
                    active_model = densenet_model
                    t_layer = "backbone.features.norm5"
                else:
                    active_model = ensemble.classifier.model
                    t_layer = "conv_head"
                    
                try:
                    gradcam = GradCAM(active_model, target_layer=t_layer, device=device)
                    processed_tensor = preprocessor.preprocess(res["original_image"], apply_aug=False).unsqueeze(0)
                    processed_tensor = ensure_3_channels(processed_tensor)
                    
                    # Generate Grad-CAM heat overlay
                    gradcam_overlay = gradcam.visualize(
                        input_tensor=processed_tensor,
                        original_image=res["original_image"],
                        target_class=res["classification_result"].class_id,
                        colormap="jet",
                        alpha=0.4
                    )
                    st.image(gradcam_overlay, caption="Grad-CAM Hotspot Activation Overlay", use_column_width=True)
                    st.session_state["results"]["gradcam_image"] = gradcam_overlay
                except Exception as e:
                    st.error(f"Could not compute Grad-CAM hooks: {e}")
            else:
                st.info("Grad-CAM overlay requires detected nodules.")
                
        st.divider()
        
        # Clinical Recommendations & Report Generation
        st.subheader("🏥 Clinical Recommendations & Next Steps")
        st.write(f"**Risk Scorer Reasoning**: {res['risk_assessment'].recommendation}")
        
        for i, rec in enumerate(res["recommendations"], 1):
            with st.expander(f"Recommendation {i}: {rec.action} ({rec.priority} Priority)", expanded=(i==1)):
                st.write(f"**Reasoning**: {rec.reasoning}")
                st.write(f"**Suggested Follow-up Window**: {rec.followup_days} days")
                if rec.additional_tests:
                    st.write(f"**Suggested Adjuvant Diagnostics**: {', '.join(rec.additional_tests)}")
                    
        # PDF Generation
        st.divider()
        st.subheader("📄 Generate Clinical PDF Report")
        if st.button("Generate Diagnostic PDF Document"):
            report_name = f"report_{p_data.get('patient_id')}_{int(time.time())}.pdf"
            report_path = OUTPUTS_DIR / report_name
            report_gen = ReportGenerator(config)
            
            gradcam_img = res.get("gradcam_image")
            
            success = report_gen.generate_pdf_report(
                output_path=str(report_path),
                patient_info=p_data,
                detection_result=res["detection_result"],
                classification_result=res["classification_result"],
                risk_assessment=res["risk_assessment"],
                recommendations=res["recommendations"],
                gradcam_image=gradcam_img,
                original_image=res["original_image"]
            )
            
            if success:
                st.success("PDF report generated successfully.")
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=f.read(),
                        file_name=report_name,
                        mime="application/pdf"
                    )
            else:
                st.error("Error creating reportlab document structure.")

    # Page 4: Analytics
    elif selected == "Analytics & Calibration":
        st.subheader("📈 Model Training Analytics & Calibration Curves")
        st.write("Below are the evaluation metrics and validation plots generated from training of our classification models.")
        
        plot_dir = OUTPUTS_DIR / "plots"
        roc_path = plot_dir / "roc_curve.png"
        cm_path = plot_dir / "confusion_matrix.png"
        grad_sample = plot_dir / "densenet_gradcam_sample.png"
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            if roc_path.exists():
                st.image(str(roc_path), caption="Validation ROC-AUC Curves", use_column_width=True)
            else:
                st.info("No ROC curves plot found. Train DenseNet121 to generate curves.")
        with col_plot2:
            if cm_path.exists():
                st.image(str(cm_path), caption="Validation Confusion Matrix", use_column_width=True)
            else:
                st.info("No confusion matrix plot found. Train DenseNet121 to generate metrics.")
                
        if grad_sample.exists():
            st.divider()
            st.subheader("🔍 Grad-CAM Calibration Sample")
            st.image(str(grad_sample), caption="Grad-CAM activation overlay validation sample", width=600)
            
    # Page 5: History Log
    elif selected == "Patient History Log":
        st.subheader("🗄️ Database Record Explorer")
        
        search_q = st.text_input("🔍 Search history by Patient ID or Name")
        limit = st.slider("Records Limit", min_value=5, max_value=100, value=20)
        
        history = db.get_history(limit=limit, search_query=search_q)
        
        if len(history) > 0:
            df = pd.DataFrame(history)
            # Filter display columns
            disp_cols = ["patient_id", "name", "age", "gender", "study_date", "nodules_detected", "classification", "classification_confidence", "risk_level", "timestamp"]
            st.dataframe(df[disp_cols], use_container_width=True)
            
            st.divider()
            st.subheader("📂 Manage Records")
            selected_patient = st.selectbox(
                "Select Patient Record",
                options=[f"{r['patient_id']} - {r['name']} ({r['timestamp']})" for r in history],
                index=0
            )
            
            selected_idx = [f"{r['patient_id']} - {r['name']} ({r['timestamp']})" for r in history].index(selected_patient)
            selected_rec = history[selected_idx]
            
            col1, col2 = st.columns(2)
            with col1:
                st.json({k: v for k, v in selected_rec.items() if k not in ["image_path", "report_path", "probabilities"]})
            with col2:
                # Retrieve saved image
                img_p = selected_rec.get("image_path")
                if img_p and os.path.exists(img_p):
                    st.image(img_p, caption="Historical scan image preview", use_column_width=True)
                else:
                    st.info("No image preview stored for this record.")
                    
            if st.button("🗑️ Delete Record", type="secondary"):
                if db.delete_record(selected_rec["id"]):
                    st.success("Record deleted successfully. Refreshing view...")
                    time.sleep(1)
                    st.experimental_rerun()
        else:
            st.info("No historical records matching search query found in the database.")


if __name__ == "__main__":
    main()
