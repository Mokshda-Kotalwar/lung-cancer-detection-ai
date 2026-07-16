import streamlit as st
import time
from datetime import datetime
import json
import io
from PIL import Image
from app.utils.api import run_prediction, get_gradcam

def render_upload(token):
    """Renders the scan upload and metadata entry view."""
    
    st.markdown('<div class="premium-header">Upload CT Scan</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>Enter patient metadata and upload a medical image for AI analysis.</p>", unsafe_allow_html=True)
    
    # Initialize session state for patient data if not present
    if "patient_data" not in st.session_state:
        st.session_state["patient_data"] = {
            "patient_id": f"PAT_{int(time.time()) % 100000}",
            "name": "",
            "age": 45,
            "gender": "Male",
            "smoker": False,
            "study_date": datetime.now()
        }

    st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 20px;'>Patient Information</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", value=st.session_state["patient_data"]["patient_id"])
        name = st.text_input("Full Name", value=st.session_state["patient_data"]["name"], placeholder="John Doe")
        age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state["patient_data"]["age"])
        
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state["patient_data"]["gender"]))
        study_date = st.date_input("Study Date", value=st.session_state["patient_data"]["study_date"])
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        smoker = st.checkbox("History of Smoking", value=st.session_state["patient_data"]["smoker"])
        
    st.session_state["patient_data"] = {
        "patient_id": patient_id, "name": name, "age": age,
        "gender": gender, "smoker": smoker, "study_date": study_date
    }
    
    
    st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 20px;'>Medical Image Upload</h4>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drag and drop DICOM, PNG, or JPEG", type=["png", "jpg", "jpeg", "dcm"])
    
    if uploaded_file is not None:
        col_img, col_act = st.columns([1, 2])
        with col_img:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Scan", use_column_width=True)
            
        with col_act:
            st.info("Image successfully loaded and ready for analysis.")
            
            if st.button("Initialize AI Analysis Pipeline", use_container_width=True):
                # Trigger the processing view
                st.session_state["pdf_report"] = None
                st.session_state["current_upload"] = uploaded_file
                st.session_state["active_view"] = "Processing"
                st.rerun()
                
