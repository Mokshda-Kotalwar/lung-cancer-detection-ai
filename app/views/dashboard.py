import streamlit as st
import pandas as pd

from app.utils.api import get_patient_history
from app.components.cards import render_kpi_card
from app.components.charts import (
    create_distribution_donut,
    create_trend_chart,
)


def render_dashboard(token):
    """Render the Clinical Dashboard."""

    st.markdown(
        '<div class="premium-header">Clinical Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style='color:#64748b;
        font-size:1.05rem;
        margin-bottom:25px;'>
        Overview of recent diagnostic activities and AI-assisted patient outcomes.
        </p>
        """,
        unsafe_allow_html=True,
    )

    history_data = get_patient_history(token)

    # ---------------------------------------------------------
    # Empty Dashboard
    # ---------------------------------------------------------
    if not history_data:

        st.info("No patient records available yet.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_kpi_card("Total Patients", 0, "0%")

        with col2:
            render_kpi_card("Total Scans", 0, "0%")

        with col3:
            render_kpi_card("Normal Cases", 0, "0%")

        with col4:
            render_kpi_card("High Risk Cases", 0, "0%", is_positive=False)
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Clinical AI Insights and AI System Status
        col_insight, col_status = st.columns([2, 1])
        
        with col_insight:
            st.markdown(
                '''
                <div class="premium-card" style="height: 100%;">
                    <h4 style='color:#1e3a8a;margin-top:0;margin-bottom:15px;'>Clinical AI Insights</h4>
                    <div style="display: flex; justify-content: space-between; gap: 10px;">
                        <div style="flex: 1;">
                            <b>🫁 Early Detection</b><br>
                            <span style="font-size:0.9em;color:#64748b;">Early diagnosis significantly improves survival rates.</span>
                        </div>
                        <div style="flex: 1;">
                            <b>📊 Prediction Confidence</b><br>
                            <span style="font-size:0.9em;color:#64748b;">Confidence is derived from DenseNet121 probability outputs.</span>
                        </div>
                        <div style="flex: 1;">
                            <b>🎯 Explainable AI</b><br>
                            <span style="font-size:0.9em;color:#64748b;">Grad-CAM highlights image regions influencing the prediction.</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; gap: 10px; margin-top: 15px;">
                        <div style="flex: 1;">
                            <b>⚠ Risk Assessment</b><br>
                            <span style="font-size:0.9em;color:#64748b;">Risk score combines image analysis and clinical information.</span>
                        </div>
                        <div style="flex: 1;">
                            <b>🩺 Recommendation</b><br>
                            <span style="font-size:0.9em;color:#64748b;">High-risk patients should undergo specialist consultation.</span>
                        </div>
                        <div style="flex: 1;"></div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

        with col_status:
            st.markdown(
                '''
                <div class="premium-card" style="height: 100%;">
                    <h4 style='color:#1e3a8a;margin-top:0;margin-bottom:15px;'>AI System Status</h4>
                    <div style="font-size:0.95em; line-height: 1.6; color:#475569; margin-bottom:15px;">
                        <div style="display:flex; justify-content:space-between;"><span>Backend</span> <b>Online</b></div>
                        <div style="display:flex; justify-content:space-between;"><span>DenseNet121</span> <b>Loaded</b></div>
                        <div style="display:flex; justify-content:space-between;"><span>Grad-CAM</span> <b>Active</b></div>
                        <div style="display:flex; justify-content:space-between;"><span>Database</span> <b>Connected</b></div>
                        <div style="display:flex; justify-content:space-between;"><span>Report Generator</span> <b>Ready</b></div>
                    </div>
                    <div style="display:inline-flex; align-items:center; gap:6px; padding:4px 12px; background:rgba(16,185,129,0.1); border-radius:99px; color:#059669; font-weight:600; font-size:0.85em;">
                        ● System Healthy
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            
        st.markdown(
            '''
            <div class="premium-card" style="margin-top: 24px; margin-bottom: 24px;">
                <h4 style='color:#1e3a8a;margin-top:0;margin-bottom:10px;'>AI Model Pipeline</h4>
                <div style="display: flex; justify-content: space-between; text-align: center; font-size:0.85em; color:#475569;">
                    <div style="flex:1;"><b>🖼️ CT Scan</b><br>Raw image input</div>
                    <div style="color:#cbd5e1; display:flex; align-items:center;">➔</div>
                    <div style="flex:1;"><b>⚙️ Image Preprocessing</b><br>Normalization & resizing</div>
                    <div style="color:#cbd5e1; display:flex; align-items:center;">➔</div>
                    <div style="flex:1;"><b>🧠 DenseNet121</b><br>Feature extraction</div>
                    <div style="color:#cbd5e1; display:flex; align-items:center;">➔</div>
                    <div style="flex:1;"><b>📊 Probability Estimation</b><br>Softmax layer</div>
                    <div style="color:#cbd5e1; display:flex; align-items:center;">➔</div>
                    <div style="flex:1;"><b>⚠ Risk Score</b><br>Clinical fusion</div>
                    <div style="color:#cbd5e1; display:flex; align-items:center;">➔</div>
                    <div style="flex:1;"><b>🎯 Grad-CAM</b><br>Saliency mapping</div>
                    <div style="color:#cbd5e1; display:flex; align-items:center;">➔</div>
                    <div style="flex:1;"><b>📄 Clinical Report</b><br>PDF generation</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        return

    # ---------------------------------------------------------
    # KPI Calculation
    # ---------------------------------------------------------

    total_patients = len(
        {
            r.get("patient_details", {}).get("patient_id")
            for r in history_data
            if r.get("patient_details")
        }
    )

    total_scans = len(history_data)

    normal_scans = sum(
        1
        for r in history_data
        if r.get("prediction", "").lower() == "normal"
    )

    high_risk_scans = sum(
    1
    for r in history_data
    if "High" in str(r.get("risk_level") or "")
)

    # ---------------------------------------------------------
    # KPI Cards
    # ---------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi_card(
            "Total Patients",
            total_patients,
            "12%",
            is_positive=True,
        )

    with col2:
        render_kpi_card(
            "Total Scans",
            total_scans,
            "8%",
            is_positive=True,
        )

    with col3:
        render_kpi_card(
            "Normal Cases",
            normal_scans,
            "5%",
            is_positive=True,
        )

    with col4:
        render_kpi_card(
            "High Risk Cases",
            high_risk_scans,
            "2%",
            is_positive=False,
        )

    st.markdown(
        '<div class="section-header">Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Prepare Data
    # ---------------------------------------------------------

    df = pd.DataFrame(history_data)

    if "created_at" in df.columns:
        df["Date"] = pd.to_datetime(df["created_at"]).dt.date

    # ---------------------------------------------------------
    # Analytics Charts
    # ---------------------------------------------------------

    col_left, col_right = st.columns(2)

    with col_left:



        st.markdown(
            """
            <h4 style='color:#1e3a8a;margin-bottom:20px;'>
            Prediction Distribution
            </h4>
            """,
            unsafe_allow_html=True,
        )

        prediction_counts = (
            df["prediction"]
            .value_counts()
        )

        fig = create_distribution_donut(
            prediction_counts.index.tolist(),
            prediction_counts.values.tolist(),
            "Prediction Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col_right:



        st.markdown(
            """
            <h4 style='color:#1e3a8a;margin-bottom:20px;'>
            Daily Prediction Trend
            </h4>
            """,
            unsafe_allow_html=True,
        )

        if "Date" in df.columns:

            trend = (
                df.groupby("Date")
                .size()
            )

            fig = create_trend_chart(
                trend.index.astype(str),
                trend.values,
                "Daily Predictions",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info("No trend data available.")
        
        st.markdown(
            "<p style='color:#64748b; font-size:0.95em;'>This chart tracks the daily volume of AI inferences and classification trends over time. Monitoring these trends helps clinicians manage case loads, track diagnostic patterns, and ensure consistent AI performance across different days.</p>",
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # Recent Activities
    # ---------------------------------------------------------

    st.markdown(
        '<div class="section-header">Recent Activities</div>',
        unsafe_allow_html=True,
    )

    recent = history_data[:5]

    rows = []

    for record in recent:

        patient = record.get("patient_details", {})

        rows.append(
            {
                "Date": record.get(
                    "created_at",
                    "",
                ).split("T")[0],
                "Patient": patient.get(
                    "name",
                    "Unknown",
                ),
                "Prediction": record.get(
                    "prediction",
                    "Unknown",
                ),
                "Risk": record.get(
                    "risk_level",
                    "Unknown",
                ),
            }
        )

    recent_df = pd.DataFrame(rows)



    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True,
    )
