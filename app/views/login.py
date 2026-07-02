import streamlit as st
from app.utils.api import authenticate_user

def render_login():
    """Renders the premium login page."""
    
    # Hide sidebar during login
    st.markdown("""
        <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            background-size: cover;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    
    st.markdown('<div class="login-title">🫁 LungAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Clinical Intelligence Platform</div>', unsafe_allow_html=True)
    
    email = st.text_input("Professional Email", placeholder="dr.smith@hospital.org")
    password = st.text_input("Password", type="password", placeholder="••••••••")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.checkbox("Remember Me")
    with col2:
        st.markdown("<div style='text-align: right; padding-top: 5px;'><a href='#' style='color: #1e3a8a; font-size: 0.9em; text-decoration: none;'>Forgot Password?</a></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Secure Login", use_container_width=True):
        if email and password:
            with st.spinner("Authenticating credentials..."):
                token = authenticate_user(email, password)
                if token:
                    st.session_state["token"] = token
                    st.toast("Authentication successful!", icon="✅")
                    st.rerun()
                else:
                    st.error("Invalid credentials or backend unavailable.")
        else:
            st.warning("Please enter both email and password.")
            
    st.markdown('</div></div>', unsafe_allow_html=True)
