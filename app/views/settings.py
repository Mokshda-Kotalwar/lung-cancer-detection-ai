import streamlit as st

def render_settings():
    """Renders the settings page."""
    st.markdown('<div class="premium-header">Application Settings</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>Manage your account preferences and application configuration.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Preferences")
        
        current_theme = st.session_state.get("theme", "Light (Default)")
        theme_index = 0 if current_theme == "Light (Default)" else 1
        
        new_theme = st.selectbox("UI Theme", ["Light (Default)", "Dark"], index=theme_index)
        if new_theme != current_theme:
            st.session_state["theme"] = new_theme
            st.rerun()
            
        st.selectbox("Language", ["English", "Spanish", "French"])
        
    with col2:
        st.markdown("#### Notifications")
        st.checkbox("Email Alerts for High Risk Cases", value=True)
        st.checkbox("System Updates", value=False)
        st.checkbox("Daily Summary Report", value=True)
        
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")
