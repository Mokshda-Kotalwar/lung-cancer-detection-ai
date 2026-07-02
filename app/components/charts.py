import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ----------------------------------------------------
# Theme Colors
# ----------------------------------------------------
def get_theme_colors():
    is_dark = st.session_state.get("theme", "Light (Default)") == "Dark"

    return {
        "bg": "#1e293b" if is_dark else "#ffffff",
        "text": "#f8fafc" if is_dark else "#0f172a",
        "primary": "#3b82f6" if is_dark else "#1e3a8a",
        "grid": "#334155" if is_dark else "#e2e8f0"
    }


# ----------------------------------------------------
# Risk Gauge
# ----------------------------------------------------
def create_risk_gauge(risk_score, risk_level):

    colors = get_theme_colors()

    if risk_level == "Low Risk":
        bar_color = "#10b981"

    elif risk_level == "Moderate Risk":
        bar_color = "#f59e0b"

    else:
        bar_color = "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            number={"suffix": "%"},
            title={"text": "Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 33], "color": "rgba(16,185,129,0.15)"},
                    {"range": [33, 66], "color": "rgba(245,158,11,0.15)"},
                    {"range": [66, 100], "color": "rgba(239,68,68,0.15)"}
                ],
            }
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text"])
    )

    return fig


# ----------------------------------------------------
# Probability Bar Chart
# ----------------------------------------------------
def create_probabilities_bar_chart(probabilities):

    colors = get_theme_colors()

    labels = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]

    fig = px.bar(
        x=values,
        y=labels,
        orientation="h",
        text=[f"{v:.1f}%" for v in values],
    )

    fig.update_traces(marker_color=colors["primary"])

    fig.update_layout(
        xaxis_title="Probability (%)",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=10, r=10, t=20, b=20),
        font=dict(color=colors["text"])
    )

    return fig


# ----------------------------------------------------
# Trend Chart
# ----------------------------------------------------
def create_trend_chart(dates, values, title="Prediction Trend"):

    colors = get_theme_colors()

    fig = px.line(
        x=dates,
        y=values,
        markers=True
    )

    fig.update_traces(
        line=dict(width=3)
    )

    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        font=dict(color=colors["text"])
    )

    return fig


# ----------------------------------------------------
# Donut Chart
# ----------------------------------------------------
def create_distribution_donut(labels, values, title="Prediction Distribution"):

    colors = get_theme_colors()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text"])
    )

    return fig


# ----------------------------------------------------
# Prediction History Chart
# ----------------------------------------------------
def render_history_chart(history_data):
    """
    Displays prediction history chart.

    Expected format:

    [
        {
            "prediction":"Normal"
        },
        {
            "prediction":"Benign"
        },
        {
            "prediction":"Malignant"
        }
    ]
    """

    if history_data is None or len(history_data) == 0:
        st.info("No prediction history available.")
        return

    df = pd.DataFrame(history_data)

    if "prediction" not in df.columns:
        st.warning("Prediction field not found.")
        return

    counts = (
        df["prediction"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Prediction",
        "Count"
    ]

    fig = px.bar(
        counts,
        x="Prediction",
        y="Count",
        color="Prediction",
        text="Count",
        title="Prediction History"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )