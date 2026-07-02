import streamlit as st

def render_kpi_card(title, value, trend, is_positive=True):
    """Renders a standard KPI metric card."""
    trend_class = "trend-up" if is_positive else "trend-down"
    trend_symbol = "▲" if is_positive else "▼"
    
    html = f"""
    <div class="premium-card kpi-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-trend {trend_class}">{trend_symbol} {trend} from last week</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_prediction_card(prediction, confidence, risk_level):
    """Renders a card summarizing the prediction results."""
    risk_color = "#10b981" if risk_level == "Low" else "#f59e0b" if risk_level == "Intermediate" else "#ef4444"
    
    html = f"""
    <div class="premium-card" style="border-left: 4px solid {risk_color};">
        <h3 style="margin-top:0; color:#1e3a8a;">AI Assessment</h3>
        <div style="margin-top:15px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.9em; color:#64748b;">Classification</div>
                <div style="font-size:1.5em; font-weight:700;">{prediction}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.9em; color:#64748b;">Confidence</div>
                <div style="font-size:1.5em; font-weight:700; color:#0f766e;">{confidence:.1f}%</div>
            </div>
        </div>
        <div style="margin-top:20px; padding-top:15px; border-top: 1px solid #e2e8f0;">
            <div style="font-size:0.9em; color:#64748b;">Risk Stratification</div>
            <div style="font-size:1.2em; font-weight:600; color:{risk_color};">{risk_level} Risk</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
