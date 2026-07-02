import streamlit as st
import pandas as pd

def render_probability_bars(probabilities):
    """Renders customized probability bars for classes."""
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 20px;'>Probability Distribution</h4>", unsafe_allow_html=True)
    
    for cls, prob in probabilities.items():
        percentage = prob * 100
        color = "#10b981" if cls.lower() == "normal" else "#f59e0b" if cls.lower() == "benign" else "#ef4444"
        
        st.markdown(f"""
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 500; color: #334155;">{cls}</span>
                <span style="font-weight: 600; color: #1e293b;">{percentage:.1f}%</span>
            </div>
            <div style="height: 8px; background-color: #e2e8f0; border-radius: 4px; overflow: hidden;">
                <div style="height: 100%; width: {percentage}%; background-color: {color}; border-radius: 4px; transition: width 1s ease-in-out;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

def render_history_chart(history_data):
    """Renders a chart showing prediction trends over time."""
    if not history_data:
        st.info("Not enough data to render charts.")
        return
        
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 20px;'>Recent Prediction Trends</h4>", unsafe_allow_html=True)
    
    # Process data
    dates = []
    risks = []
    for r in history_data:
        dates.append(r.get("created_at", "Unknown").split("T")[0])
        risks.append(r.get("risk_level", "Unknown"))
        
    if dates:
        df = pd.DataFrame({"Date": dates, "Risk": risks})
        counts = df.groupby(["Date", "Risk"]).size().unstack(fill_value=0)
        
        # We will use Streamlit's native bar chart but it will fit in the card
        st.bar_chart(counts, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
