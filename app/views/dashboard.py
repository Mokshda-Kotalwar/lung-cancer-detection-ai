import streamlit as st
import pandas as pd
from app.utils.api import get_patient_history
from app.components.cards import render_kpi_card
from app.components.charts import render_history_chart

def render_dashboard(token):
    """Renders the home dashboard view."""
    
    st.markdown('<div class="premium-header">Clinical Dashboard</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>Overview of recent diagnostic activities and patient outcomes.</p>", unsafe_allow_html=True)
    
    history_data = get_patient_history(token)
    
    # Calculate KPIs
    total_patients = len(set([r.get("patient_details", {}).get("patient_id") for r in history_data if r.get("patient_details")]))
    total_scans = len(history_data)
    high_risk_scans = sum(1 for r in history_data if r.get("risk_level") in ["High", "Critical"])
    normal_scans = sum(1 for r in history_data if r.get("prediction", "").lower() == "normal")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Patients", total_patients, "12%", is_positive=True)
    with col2:
        render_kpi_card("Total Scans", total_scans, "5%", is_positive=True)
    with col3:
        render_kpi_card("Normal Cases", normal_scans, "2%", is_positive=True)
    with col4:
        render_kpi_card("High Risk Cases", high_risk_scans, "1%", is_positive=False)
        
    st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 20px;'>Prediction Distribution</h4>", unsafe_allow_html=True)
        if history_data:
            preds = [r.get("prediction", "Unknown") for r in history_data]
            df_preds = pd.DataFrame(preds, columns=["Prediction"])
            counts = df_preds["Prediction"].value_counts()
            st.bar_chart(counts, use_container_width=True)
        else:
            st.info("No data available.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_chart2:
        render_history_chart(history_data)
        
    st.markdown('<div class="section-header">Recent Activities</div>', unsafe_allow_html=True)
    
    if history_data:
        # Get last 5 records
        recent = history_data[:5]
        flat_recent = []
        for r in recent:
            pdets = r.get("patient_details", {})
            flat_recent.append({
                "Date": r.get("created_at", "").split("T")[0],
                "Patient": pdets.get("name", "Unknown"),
                "Prediction": r.get("prediction", "Unknown"),
                "Risk": r.get("risk_level", "Unknown")
            })
            
        df = pd.DataFrame(flat_recent)
        
        st.markdown('<div class="premium-card" style="padding:0; overflow:hidden;">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No recent activities found.")
