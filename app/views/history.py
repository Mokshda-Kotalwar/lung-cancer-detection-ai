import streamlit as st
import pandas as pd
from app.utils.api import get_patient_history

def render_history(token):
    """Renders the prediction history view."""
    
    st.markdown('<div class="premium-header">Prediction History</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>Review past clinical predictions and model inferences.</p>", unsafe_allow_html=True)
    
    history_data = get_patient_history(token)
    
    if not history_data:
        st.info("No prediction history found. Upload a scan to generate your first prediction.")
        return
        
    df = pd.DataFrame(history_data)
    
    # Clean up the dataframe for display
    if "patient_details" in df.columns:
        df["Patient ID"] = df["patient_details"].apply(lambda x: x.get("patient_id", "Unknown") if isinstance(x, dict) else "Unknown")
        df["Patient Name"] = df["patient_details"].apply(lambda x: x.get("name", "Unknown") if isinstance(x, dict) else "Unknown")
        df["Age"] = df["patient_details"].apply(lambda x: x.get("age", "Unknown") if isinstance(x, dict) else "Unknown")
        
    # Reorder and format columns
    display_cols = ["Patient ID", "Patient Name", "Age", "prediction", "confidence", "risk_level"]
    available_cols = [col for col in display_cols if col in df.columns]
    df = df[available_cols]
    
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].apply(lambda x: f"{x:.2%}" if isinstance(x, (float, int)) else x)
        
    df.rename(columns={
        "prediction": "Diagnosis",
        "confidence": "Confidence",
        "risk_level": "Risk Level"
    }, inplace=True)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
