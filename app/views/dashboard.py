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
            '<div class="premium-card">',
            unsafe_allow_html=True,
        )

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

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with col_right:

        st.markdown(
            '<div class="premium-card">',
            unsafe_allow_html=True,
        )

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
            "</div>",
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

    st.markdown(
        '<div class="premium-card" style="padding:0;overflow:hidden;">',
        unsafe_allow_html=True,
    )

    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )