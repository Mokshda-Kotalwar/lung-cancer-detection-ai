"""
Streamlit Web Application for Lung Cancer Detection System
Interactive dashboard for medical image analysis
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import logging
import torch
import time
from PIL import Image as PILImage
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from config import config, OUTPUTS_DIR
from src.preprocessing import ImagePreprocessor
from src.detection import YOLODetector, EfficientNetClassifier, EnsembleDetector
from src.xai import GradCAM
from src.risk import RiskScorer, RecommendationEngine
from src.reporting import ReportGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { color: #1f4788; font-size: 2.5em; text-align: center; }
    .metric-box { 
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 8px; 
        margin: 10px 0;
    }
    .risk-low { background-color: #d4edda; color: #155724; }
    .risk-intermediate { background-color: #fff3cd; color: #856404; }
    .risk-high { background-color: #f8d7da; color: #721c24; }
    .risk-critical { background-color: #f5c6cb; color: #721c24; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load AI models (cached for performance)"""
    try:
        with st.spinner("Loading AI models..."):
            ensemble = EnsembleDetector(config)
            preprocessor = ImagePreprocessor(config)
            return ensemble, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def display_header():
    """Display application header"""
    st.markdown("<h1 class='main-header'>🫁 AI-Powered Lung Cancer Detection System</h1>", 
                unsafe_allow_html=True)
    st.markdown("""
        <p style="text-align: center; color: #666;">
        Advanced AI system for analyzing medical images and assessing lung cancer risk
        </p>
    """, unsafe_allow_html=True)


def display_disclaimer():
    """Display medical disclaimer"""
    with st.sidebar.expander("⚠️ Medical Disclaimer"):
        st.warning("""
        **IMPORTANT DISCLAIMER:**
        
        This AI system is designed for **supporting clinical decision-making only**. 
        It should NOT be used as a standalone diagnostic tool.
        
        - All findings must be validated by qualified medical professionals
        - Final diagnosis and treatment decisions are the responsibility of healthcare providers
        - This system does not replace professional medical evaluation
        - Always consult with licensed physicians for medical decisions
        
        By using this system, you acknowledge these limitations.
        """)


def process_image(image, ensemble, preprocessor):
    """Process image with AI models"""
    try:
        # Preprocess
        processed_image = preprocessor.preprocess(image, apply_aug=False)
        
        # Detection and classification
        detection_result, classification_result = ensemble.process(image)
        
        return processed_image, detection_result, classification_result
    except Exception as e:
        st.error(f"Error processing image: {e}")
        logger.error(f"Processing error: {e}")
        return None, None, None


def display_analysis_results(detection_result, classification_result, original_image):
    """Display analysis results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Nodules Detected",
            len(detection_result.boxes) if detection_result else 0,
            delta="potential lesions"
        )
    
    with col2:
        if detection_result and len(detection_result.confidences) > 0:
            avg_conf = detection_result.confidences.mean() * 100
            st.metric("Average Detection Confidence", f"{avg_conf:.1f}%")
        else:
            st.metric("Average Detection Confidence", "N/A")
    
    with col3:
        if classification_result:
            st.metric(
                "Classification",
                classification_result.class_name,
                f"({classification_result.confidence * 100:.1f}%)"
            )
        else:
            st.metric("Classification", "N/A")
    
    # Display detection visualization
    if detection_result and len(detection_result.boxes) > 0:
        detection_image = ensemble.detector.draw_detections(original_image, detection_result)
        
        st.subheader("Detection Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.image(detection_image, caption="Detected Nodules", use_column_width=True)
    
    # Display classification probabilities
    if classification_result:
        st.subheader("Classification Probabilities")
        
        prob_dict = classification_result.probabilities
        cols = st.columns(len(prob_dict))
        
        for i, (class_name, prob) in enumerate(prob_dict.items()):
            with cols[i]:
                st.metric(class_name, f"{prob * 100:.1f}%")


def display_risk_assessment(risk_assessment):
    """Display risk assessment"""
    st.subheader("Risk Assessment")
    
    # Risk level with color coding
    risk_class_map = {
        "Low": "risk-low",
        "Intermediate": "risk-intermediate",
        "High": "risk-high",
        "Critical": "risk-critical"
    }
    
    risk_class = risk_class_map.get(risk_assessment.risk_level.value, "")
    
    st.markdown(f"""
        <div class="metric-box {risk_class}">
            <h3>Risk Level: {risk_assessment.risk_level.value}</h3>
            <p><strong>Risk Score:</strong> {risk_assessment.risk_score:.3f} (0-1 scale)</p>
            <p><strong>Model Confidence:</strong> {risk_assessment.confidence * 100:.1f}%</p>
            <p><strong>Model Uncertainty:</strong> {risk_assessment.uncertainty * 100:.1f}%</p>
            <p><strong>Recommendation:</strong> {risk_assessment.recommendation}</p>
            <p><strong>Follow-up Period:</strong> {risk_assessment.followup_period}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key features
    if risk_assessment.key_features:
        st.write("**Key Contributing Factors:**")
        for feature in risk_assessment.key_features:
            st.write(f"• {feature}")


def display_recommendations(recommendations):
    """Display clinical recommendations"""
    st.subheader("Clinical Recommendations")
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"Recommendation {i}: {rec.action}", expanded=(i == 1)):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Priority:** {rec.priority}")
                st.write(f"**Follow-up (days):** {rec.followup_days}")
            
            with col2:
                st.write(f"**Reasoning:** {rec.reasoning}")
            
            if rec.additional_tests:
                st.write(f"**Additional Tests:** {', '.join(rec.additional_tests)}")


def main():
    """Main application"""
    display_header()
    display_disclaimer()
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        
        mode = st.radio(
            "Select Mode",
            ["Single Image Analysis", "Batch Processing", "Reports"],
            key="mode_selector"
        )
    
    # Load models
    ensemble, preprocessor = load_models()
    
    if ensemble is None or preprocessor is None:
        st.error("Failed to load models. Please check your installation.")
        return
    
    # Single Image Analysis
    if mode == "Single Image Analysis":
        st.subheader("Upload Medical Image")
        
        uploaded_file = st.file_uploader(
            "Choose a medical image (DICOM, JPG, PNG)",
            type=["dcm", "jpg", "png", "jpeg"]
        )
        
        if uploaded_file is not None:
            # Load image
            if uploaded_file.type == "application/octet-stream":
                # DICOM file
                try:
                    import pydicom
                    ds = pydicom.dcmread(uploaded_file)
                    image = ds.pixel_array.astype(np.float32)
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                except:
                    st.error("Error reading DICOM file")
                    return
            else:
                # Standard image format
                image = PILImage.open(uploaded_file)
                image = np.array(image)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Process
            with st.spinner("Analyzing image..."):
                start_time = time.time()
                processed, detection_result, classification_result = process_image(
                    image, ensemble, preprocessor
                )
                processing_time = time.time() - start_time
            
            # Calculate risk assessment
            if detection_result and classification_result:
                risk_scorer = RiskScorer(config)
                risk_assessment = risk_scorer.calculate_risk(
                    classification_confidence=classification_result.confidence,
                    detection_count=len(detection_result.boxes),
                    detection_size=10.0,  # Placeholder
                    detection_confidence=detection_result.confidences.mean() if len(detection_result.confidences) > 0 else 0
                )
                
                # Generate recommendations
                rec_engine = RecommendationEngine(config)
                recommendations = rec_engine.generate_recommendations(risk_assessment)
                
                # Display results
                st.success(f"Analysis completed in {processing_time:.2f}s")
                
                st.divider()
                display_analysis_results(detection_result, classification_result, image)
                
                st.divider()
                display_risk_assessment(risk_assessment)
                
                st.divider()
                display_recommendations(recommendations)
                
                # Generate report
                st.divider()
                st.subheader("Report Generation")
                
                if st.button("Generate PDF Report"):
                    report_path = OUTPUTS_DIR / f"report_{time.time():.0f}.pdf"
                    report_gen = ReportGenerator(config)
                    
                    success = report_gen.generate_pdf_report(
                        str(report_path),
                        patient_info={"patient_id": "TEST001", "study_date": "2024-01-01"},
                        detection_result=detection_result,
                        classification_result=classification_result,
                        risk_assessment=risk_assessment,
                        recommendations=recommendations,
                        original_image=image
                    )
                    
                    if success:
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="Download PDF Report",
                                data=f.read(),
                                file_name=report_path.name,
                                mime="application/pdf"
                            )
                    else:
                        st.error("Failed to generate PDF report")
    
    elif mode == "Batch Processing":
        st.info("Batch processing mode coming soon!")
    
    elif mode == "Reports":
        st.info("Reports management coming soon!")


if __name__ == "__main__":
    main()
