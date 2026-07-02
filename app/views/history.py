import streamlit as st
import pandas as pd
from app.utils.api import get_patient_history

def render_history(token):
    """Renders the prediction history view."""
    
    st.markdown('<div class="premium-header">Patient History Database</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>Search and filter past diagnostic records.</p>", unsafe_allow_html=True)
    
    history_data = get_patient_history(token)
    
    if not history_data:
        st.info("No historical records found for this account.")
        return
        
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    
    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        search_pt = st.text_input("Search Patient", placeholder="Name or ID...")
    with col_f2:
        filter_pred = st.selectbox("Filter by Prediction", ["All", "Normal", "Benign", "Malignant"])
    with col_f3:
        filter_risk = st.selectbox("Filter by Risk", ["All", "Low", "Intermediate", "High", "Critical"])
        
    # Process data
    flat_history = []
    for r in history_data:
        pdets = r.get("patient_details", {})
        
        # Apply filters
        p_name = pdets.get("name", "").lower()
        p_id = pdets.get("patient_id", "").lower()
        pred = r.get("prediction", "")
        risk = r.get("risk_level", "")
        
        if search_pt and (search_pt.lower() not in p_name and search_pt.lower() not in p_id):
            continue
        if filter_pred != "All" and filter_pred.lower() != pred.lower():
            continue
        if filter_risk != "All" and filter_risk.lower() != risk.lower():
            continue
            
        flat_history.append({
            "Date": r.get("created_at", "").split("T")[0],
            "Patient ID": pdets.get("patient_id", "Unknown"),
            "Name": pdets.get("name", "Unknown"),
            "Prediction": pred,
            "Confidence": f"{r.get('confidence', 0)*100:.1f}%",
            "Risk": risk
        })
        
    if flat_history:
        df = pd.DataFrame(flat_history)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No records match your filters.")
        
    st.markdown('</div>', unsafe_allow_html=True)
