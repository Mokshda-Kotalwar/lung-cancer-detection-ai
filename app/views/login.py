import streamlit as st
from app.utils.api import authenticate_user

def render_login():
    """Renders the login view with a premium design."""
    
    # Hide sidebar on login
    st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="premium-header" style="text-align: center; margin-bottom: 0.5rem; margin-top: 5rem;">LungAI Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #64748b; margin-bottom: 3rem;">Clinical Intelligence Platform</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; margin-bottom: 30px;'>Sign in to your account</h3>", unsafe_allow_html=True)
        
        email = st.text_input("Email Address", placeholder="dr.smith@hospital.org")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        
        col_rm, col_fp = st.columns(2)
        with col_rm:
            st.checkbox("Remember me")
        with col_fp:
            st.markdown("<p style='text-align: right;'><a href='#' style='color: #3b82f6; text-decoration: none;'>Forgot password?</a></p>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Secure Login", use_container_width=True, type="primary"):
            if email and password:
                with st.spinner("Authenticating credentials..."):
                    token = authenticate_user(email, password)
                    if token:
                        st.session_state["token"] = token
                        st.session_state["authenticated"] = True
                        st.session_state["active_view"] = "Dashboard"
                        st.toast("Authentication successful!", icon="✅")
                        st.rerun()
                    else:
                        st.error("Invalid credentials or backend unavailable.")
            else:
                st.warning("Please enter both email and password.")
                    
        st.markdown("<p style='text-align: center; margin-top: 20px; color: #64748b;'>Don't have an account? <a href='#' style='color: #3b82f6; text-decoration: none;'>Request access</a></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
