import streamlit as st
from streamlit_option_menu import option_menu
from app.utils.styles import render_badge

def render_sidebar(backend_ok=True):
    """Renders the main navigation sidebar."""
    with st.sidebar:
        st.markdown(
            "<div style='text-align: center; margin-bottom: 20px;'>"
            "<h2 style='color: #1e3a8a; margin-bottom: 0; font-weight: 700;'>🫁 LungAI</h2>"
            "<p style='color: #64748b; font-size: 0.9em; margin-top:0;'>Clinical Decision Support</p>"
            "</div>", 
            unsafe_allow_html=True
        )
        

        
        selected = option_menu(
            menu_title="Navigation",
            options=["Dashboard", "Upload CT Scan", "Prediction History", "Settings"],
            icons=["grid", "cloud-arrow-up", "clock-history", "gear"],
            menu_icon="compass",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#64748b", "font-size": "16px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin": "4px 0", "--hover-color": "#f1f5f9", "color": "#475569"},
                "nav-link-selected": {"background-color": "#1e3a8a", "color": "#ffffff", "font-weight": "600"},
            }
        )

        st.divider()
        
        # System Status
        st.markdown("<h4 style='font-size:0.9em; color:#64748b; text-transform:uppercase; margin-bottom:10px;'>System Status</h4>", unsafe_allow_html=True)
        if backend_ok:
            render_badge("Backend: Online", "success")
        else:
            render_badge("Backend: Offline", "error")
            
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        render_badge("AI Model: DenseNet121", "info")

        st.divider()

        if st.button("Log Out", use_container_width=True, type="secondary"):
            st.session_state["token"] = None
            st.rerun()

    return selected
