"""
Streamlit Web Application for Lung Cancer Detection System
A state-of-the-art clinical dashboard for medical image analysis, classification, and explainability.
Refactored to operate as a microservice frontend calling a FastAPI backend.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage
import json
import io

import cv2
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="LungAI Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://127.0.0.1:8000"

print("Backend:", BACKEND_URL)

try:
    r = requests.get(f"{BACKEND_URL}/health/", timeout=5)
    print(r.status_code)
    print(r.text)
except Exception as e:
    print(e)

from streamlit_option_menu import option_menu
import requests

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from config import OUTPUTS_DIR


# Premium Dark Glassmorphism CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    .glass-card { background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); }
    .glass-header { font-size: 2.2em; font-weight: 700; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 8px; }
    .status-badge { display: inline-block; padding: 6px 12px; border-radius: 9999px; font-size: 0.85em; font-weight: 600; text-align: center; }
    .status-active { background-color: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-warning { background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .risk-badge { font-size: 1.1em; font-weight: 700; padding: 8px 16px; border-radius: 8px; text-transform: uppercase; display: inline-block; margin-top: 8px; }
    .risk-low { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
    .risk-intermediate { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
    .risk-high { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }
    .risk-critical { background-color: rgba(220, 38, 38, 0.35); color: #fca5a5; border: 2px solid #ef4444; font-weight: 900; }
    .metric-value { font-size: 2em; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.9em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    section[data-testid="stSidebar"] { background-color: #070a13; border-right: 1px solid rgba(255, 255, 255, 0.05); }
    </style>
""", unsafe_allow_html=True)


def check_backend():
    try:
        url = f"{BACKEND_URL}/health/"
        print("Checking:", url)

        response = requests.get(url, timeout=5)

        print("Status:", response.status_code)
        print("Body:", response.text)

        return response.status_code == 200

    except Exception as e:
        print("Backend Error:", repr(e))
        return False

def login(email, password):
    data = {"username": email, "password": password}
    try:
        response = requests.post(f"{BACKEND_URL}/auth/login", data=data)
        if response.status_code == 200:
            return response.json().get("access_token")
    except:
        pass
    return None

def fetch_history(token):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(f"{BACKEND_URL}/history/", headers=headers)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

def main():
    if "token" not in st.session_state:
        st.session_state["token"] = None

    backend_ok = check_backend()

    with st.sidebar:
        st.markdown("<div style='text-align: center;'><h2 style='color: #60a5fa; margin-bottom: 0;'>🫁 LungAI</h2><p style='color: #64748b; font-size: 0.9em; margin-top:0;'>Medical Diagnosis System</p></div>", unsafe_allow_html=True)
        st.divider()
        
        if backend_ok:
            st.markdown(f"<div class='status-badge status-active'>● Backend Connected</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-badge status-warning'>▲ Backend Disconnected</div>", unsafe_allow_html=True)
            
        st.divider()

        if not st.session_state["token"]:
            st.subheader("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Sign In"):
                token = login(email, password)
                if token:
                    st.session_state["token"] = token
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Invalid credentials or backend unavailable.")
        else:
            st.success("Authenticated")
            if st.button("Log Out"):
                st.session_state["token"] = None
                st.rerun()

        st.divider()
        with st.expander("⚠️ Medical Disclaimer", expanded=False):
            st.warning("This AI system is designed for clinical decision support. It should not be used as a standalone diagnostic tool.")
            
    if not st.session_state["token"]:
        st.info("Please log in via the sidebar to access the dashboard.")
        return

    selected = option_menu(
        menu_title=None,
        options=["Home", "Scan Upload", "Diagnosis Results", "Patient History Log"],
        icons=["house", "cloud-upload", "heart-pulse", "clock-history"],
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

    if "patient_data" not in st.session_state:
        st.session_state["patient_data"] = {}
    if "results" not in st.session_state:
        st.session_state["results"] = None

    if selected == "Home":
        st.markdown("<h1 class='glass-header'>🫁 AI-Powered Lung Cancer Detection System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1em;'>Advanced computer vision and deep learning pipeline.</p>", unsafe_allow_html=True)
        
        history_records = fetch_history(st.session_state["token"])
        total_scans = len(history_records)
        high_risk_scans = sum(1 for r in history_records if r.get("risk_level") in ["High", "Critical"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='glass-card' style='text-align: center;'><div class='metric-label'>Your Total Scans</div><div class='metric-value'>{total_scans}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='glass-card' style='text-align: center;'><div class='metric-label'>High/Critical Findings</div><div class='metric-value' style='color: #f87171;'>{high_risk_scans}</div></div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("ℹ️ About Lung Cancer")
        st.markdown("""
        Lung cancer is the leading cause of cancer deaths worldwide. Early detection can significantly increase the chances of successful treatment and survival. 
        Our AI-powered diagnostic system is designed to assist radiologists and healthcare professionals by analyzing CT scan slices to identify potential pulmonary nodules and classify their malignancy risk.
        
        **Key Facts:**
        - **Early Detection:** When diagnosed at an early stage, survival rates are much higher.
        - **Risk Factors:** Smoking is the primary risk factor, but exposure to radon, asbestos, and family history also play significant roles.
        - **Our Technology:** Uses state-of-the-art Deep Learning (DenseNet & YOLOv8) to highlight areas of concern (Grad-CAM) and estimate clinical risk levels.
        """)
            
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
            "patient_id": patient_id, "name": name, "age": age,
            "gender": gender, "smoker": smoker, "study_date": study_date.strftime("%Y-%m-%d")
        }
        
        st.subheader("📤 Upload Medical Image")
        uploaded_file = st.file_uploader("Choose a CT scan slice (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            st.info("Medical scan uploaded successfully.")
            image_bytes = uploaded_file.getvalue()
            
            if st.button("🚀 Run Diagnostic Pipeline", use_container_width=True):
                with st.spinner("Calling Inference Backend..."):
                    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                    files = {"file": (uploaded_file.name, image_bytes, "image/png")}
                    data = {"patient_details_json": json.dumps(st.session_state["patient_data"])}
                    
                    try:
                        res = requests.post(f"{BACKEND_URL}/predict/", headers=headers, files=files, data=data)
                        if res.status_code == 200:
                            pred_data = res.json()
                            
                            # Fetch GradCAM
                            files_g = {"file": (uploaded_file.name, image_bytes, "image/png")}
                            res_g = requests.post(f"{BACKEND_URL}/predict/gradcam", headers=headers, files=files_g)
                            
                            gradcam_image = None
                            if res_g.status_code == 200:
                                gradcam_image = PILImage.open(io.BytesIO(res_g.content))
                                
                            st.session_state["results"] = {
                                "prediction_data": pred_data,
                                "gradcam_image": gradcam_image,
                                "original_image": PILImage.open(io.BytesIO(image_bytes))
                            }
                            st.success("Diagnosis completed successfully! Proceed to the 'Diagnosis Results' tab.")
                        else:
                            st.error(f"Backend error: {res.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to backend: {e}")
                    
    elif selected == "Diagnosis Results":
        res = st.session_state.get("results")
        if res is None:
            st.warning("Please upload a scan and run the diagnostic pipeline first.")
            return
            
        p_data = st.session_state["patient_data"]
        pred_data = res["prediction_data"]
        
        st.markdown(f"<h2>Patient Diagnostic Report: {p_data.get('name')} ({p_data.get('patient_id')})</h2>", unsafe_allow_html=True)
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            pred_class = pred_data.get("prediction", "N/A")
            pred_conf = pred_data.get("confidence", 0.0) * 100
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Classifier Predicton</div><div class='metric-value'>{pred_class} <span style='font-size:0.55em; color:#a855f7;'>({pred_conf:.1f}%)</span></div></div>", unsafe_allow_html=True)
            
        with col2:
            r_level = pred_data.get("risk_level", "Unknown")
            r_class = f"risk-{r_level.lower()}"
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Stratified Risk Level</div><div class='risk-badge {r_class}'>{r_level}</div></div>", unsafe_allow_html=True)
            
        st.divider()
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.image(res["original_image"], caption="Original Scan", use_column_width=True)
            
        with col_img2:
            if res.get("gradcam_image"):
                st.image(res["gradcam_image"], caption="Grad-CAM Activation Map", use_column_width=True)
            else:
                st.info("Grad-CAM image could not be retrieved from backend.")
                
        st.divider()
        st.subheader("📊 Malignancy Probability Breakdown")
        probs = pred_data.get("probabilities", {})
        if probs:
            df_probs = pd.DataFrame({"Diagnosis": list(probs.keys()), "Probability (%)": [v * 100 for v in probs.values()]})
            st.bar_chart(df_probs.set_index("Diagnosis"), use_container_width=True)
            
        st.subheader("🏥 Clinical Recommendations & Next Steps")
        st.write(f"**Risk Scorer Reasoning**: {pred_data.get('recommendation', 'N/A')}")
        
        st.divider()
        st.subheader("📄 Export Report")
        
        if st.button("Download PDF Report", use_container_width=True):
            with st.spinner("Generating PDF Report..."):
                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                
                # We need to send the original image bytes stored in session_state, but we only have the PIL Image.
                # Let's convert it back to bytes.
                img_byte_arr = io.BytesIO()
                res["original_image"].save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                
                files = {"file": ("scan.png", img_bytes, "image/png")}
                data = {"patient_details_json": json.dumps(st.session_state["patient_data"])}
                
                try:
                    pdf_res = requests.post(f"{BACKEND_URL}/predict/report", headers=headers, files=files, data=data)
                    if pdf_res.status_code == 200:
                        st.download_button(
                            label="Click here to save the PDF",
                            data=pdf_res.content,
                            file_name=f"lung_cancer_report_{p_data.get('patient_id', 'unknown')}.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Failed to generate report: {pdf_res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend for report generation: {e}")
        
    elif selected == "Patient History Log":
        st.subheader("🗄️ Database Record Explorer")
        
        history = fetch_history(st.session_state["token"])
        
        if len(history) > 0:
            flat_history = []
            for r in history:
                pdets = r.get("patient_details", {})
                flat_history.append({
                    "Patient ID": pdets.get("patient_id"),
                    "Name": pdets.get("name"),
                    "Prediction": r.get("prediction"),
                    "Confidence": r.get("confidence"),
                    "Risk Level": r.get("risk_level"),
                    "Date": r.get("created_at")
                })
            df = pd.DataFrame(flat_history)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No historical records found for your account.")

if __name__ == "__main__":
    main()
