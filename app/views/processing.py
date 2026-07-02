import streamlit as st
import time
import io
from PIL import Image
from app.utils.api import run_prediction, get_gradcam

def render_processing(token):
    """Renders an animated processing pipeline view."""
    
    st.markdown('<div class="premium-header">Processing Analysis</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>Executing DenseNet121 deep learning pipeline...</p>", unsafe_allow_html=True)
    
    uploaded_file = st.session_state.get("current_upload")
    patient_data = st.session_state.get("patient_data")
    
    if not uploaded_file:
        st.error("No file uploaded.")
        st.session_state["active_view"] = "Upload CT Scan"
        st.rerun()
        return

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    steps_container = st.empty()
    
    steps = [
        "Image Upload & Verification",
        "Preprocessing & Noise Reduction",
        "Lung Region Detection",
        "DenseNet121 Feature Extraction",
        "Risk Calculation & Stratification",
        "Grad-CAM Generation",
        "Finalizing Results"
    ]
    
    # We will simulate the steps visually, but the actual backend call is monolithic.
    # To provide the premium feel requested, we'll run the backend call, then animate the steps.
    
    # First, make the API call while showing a spinner
    image_bytes = uploaded_file.getvalue()
    
    with st.spinner("Connecting to inference cluster..."):
        # Format date for JSON serialization
        pt_data_copy = patient_data.copy()
        if hasattr(pt_data_copy["study_date"], "strftime"):
            pt_data_copy["study_date"] = pt_data_copy["study_date"].strftime("%Y-%m-%d")
            
        try:
            res = run_prediction(token, image_bytes, uploaded_file.name, pt_data_copy)
            if res.status_code == 200:
                pred_data = res.json()
                
                # Fetch GradCAM
                gradcam_img = get_gradcam(token, image_bytes, uploaded_file.name)
                
                st.session_state["results"] = {
                    "prediction_data": pred_data,
                    "gradcam_image": gradcam_img,
                    "original_image": Image.open(io.BytesIO(image_bytes))
                }
                
                # Animate the success steps for premium UX
                for i, step in enumerate(steps):
                    progress = int(((i + 1) / len(steps)) * 100)
                    progress_bar.progress(progress)
                    status_text.markdown(f"**Executing:** {step}...")
                    
                    # Build checkmark list
                    completed_steps = "".join([f"<div style='margin-bottom: 8px; color: #10b981;'>✓ {s}</div>" for s in steps[:i+1]])
                    steps_container.markdown(completed_steps, unsafe_allow_html=True)
                    
                    time.sleep(0.3)
                
                time.sleep(0.5)
                st.session_state["active_view"] = "Results"
                st.rerun()
            else:
                st.error(f"Backend error: {res.text}")
                if st.button("Return to Upload"):
                    st.session_state["active_view"] = "Upload CT Scan"
                    st.rerun()
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            if st.button("Return to Upload"):
                st.session_state["active_view"] = "Upload CT Scan"
                st.rerun()
                
    st.markdown('</div>', unsafe_allow_html=True)
