import streamlit as st
from app.components.charts import create_trend_chart, create_distribution_donut

def render_analytics(token):
    """Renders the Analytics dashboard."""
    st.markdown('<div class="premium-header">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>System-wide insights and historical trends</p>", unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="premium-card kpi-card">
            <div class="kpi-label">Total Predictions</div>
            <div class="kpi-value">1,284</div>
            <div class="kpi-trend trend-up">↑ 12% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="premium-card kpi-card">
            <div class="kpi-label">Average Confidence</div>
            <div class="kpi-value">94.2%</div>
            <div class="kpi-trend trend-up">↑ 1.5% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="premium-card kpi-card">
            <div class="kpi-label">High Risk Cases</div>
            <div class="kpi-value">142</div>
            <div class="kpi-trend trend-down">↓ 3% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="premium-card kpi-card">
            <div class="kpi-label">Avg Process Time</div>
            <div class="kpi-value">1.2s</div>
            <div class="kpi-trend trend-up">↓ 0.3s vs last month</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Charts Row 1
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        # Mock Data for Trends
        dates = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
        values = [120, 150, 180, 170, 210, 250, 194]
        st.plotly_chart(create_trend_chart(dates, values, "Monthly Predictions (Last 7 Months)"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_r:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        # Mock Data for Distribution
        labels = ["Benign", "Malignant", "Uncertain"]
        values = [850, 314, 120]
        st.plotly_chart(create_distribution_donut(labels, values, "Diagnosis Distribution"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
