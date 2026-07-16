import streamlit as st
import json
import io
from app.utils.api import generate_report
from app.components.charts import create_risk_gauge, create_probabilities_bar_chart

def render_results(token):
    """Renders the prediction results and Grad-CAM view."""
    
    res = st.session_state.get("results")
    if not res:
        st.warning("No results available. Please run an analysis first.")
        if st.button("Go to Upload"):
            st.session_state["active_view"] = "Upload CT Scan"
            st.rerun()
        return

    pt_data = st.session_state.get("patient_data", {})
    pred_data = res.get("prediction_data", {})
    
    st.markdown('<div class="premium-header">Diagnostic Results</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>Patient: {pt_data.get('name', 'Unknown')} | ID: {pt_data.get('patient_id', 'Unknown')}</p>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 15px;'>Imaging Analysis</h4>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Original Scan", "Grad-CAM Overlay"])
        
        with tab1:
            st.image(res["original_image"], use_column_width=True)
            
        with tab2:
            if res.get("gradcam_image"):
                st.image(res["gradcam_image"], use_column_width=True)
                st.info("The highlighted red regions contributed most to the model's prediction.")
            else:
                st.warning("Grad-CAM image not available.")
                
        
    with col_right:
        pred_class = pred_data.get("prediction", "N/A")
        pred_conf = pred_data.get("confidence", 0.0)
        r_level = pred_data.get("risk_level", "Unknown")
        risk_score = pred_data.get("risk_score", 0.0)
        
        # We can use the gauge for risk score
        st.plotly_chart(create_risk_gauge(risk_score, r_level), use_container_width=True)
        
        st.markdown("#### Class Probabilities")
        probs = pred_data.get("probabilities", {})
        if probs:
            st.plotly_chart(create_probabilities_bar_chart(probs), use_container_width=True)
            
        st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 10px;'>Clinical Recommendation</h4>", unsafe_allow_html=True)
        st.write(pred_data.get('recommendation', 'Consult with a senior radiologist.'))
        
        # Report Generation
        if "pdf_report" not in st.session_state:
            st.session_state["pdf_report"] = None

        if st.session_state["pdf_report"] is None:
            if st.button("Generate PDF Report", use_container_width=True):
                with st.spinner("Compiling clinical report..."):
                    uploaded_file = st.session_state.get("current_upload")
                    if uploaded_file:
                        uploaded_file.seek(0)
                        img_bytes = uploaded_file.getvalue()
                        filename = uploaded_file.name
                    else:
                        img_byte_arr = io.BytesIO()
                        res["original_image"].save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        filename = "scan.png"
                    
                    pt_data_copy = pt_data.copy()
                    if hasattr(pt_data_copy["study_date"], "strftime"):
                        pt_data_copy["study_date"] = pt_data_copy["study_date"].strftime("%Y-%m-%d")
                        
                    pdf_res = generate_report(token, img_bytes, filename, pt_data_copy)
                    
                    if pdf_res and pdf_res.status_code == 200:
                        st.session_state["pdf_report"] = pdf_res.content
                        st.rerun()
                    else:
                        st.error(f"Failed to generate report: {pdf_res.text if hasattr(pdf_res, 'text') else 'Unknown Error'}")
        else:
            st.download_button(
                label="Download Generated PDF",
                data=st.session_state["pdf_report"],
                file_name=f"LungAI_Report_{pt_data.get('patient_id', 'unknown')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
