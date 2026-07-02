import streamlit as st

def render_settings():
    """Renders the settings view."""
    
    st.markdown('<div class="premium-header">Application Settings</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>Manage your account preferences and application configuration.</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 20px;'>Profile Information</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Full Name", value="Dr. Sarah Jenkins")
        st.text_input("Email", value="sarah.jenkins@hospital.org")
    with col2:
        st.text_input("Department", value="Oncology")
        st.text_input("Hospital ID", value="HOSP-4921")
        
    if st.button("Update Profile"):
        st.toast("Profile updated successfully", icon="✅")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 20px;'>Preferences</h4>", unsafe_allow_html=True)
    
    st.selectbox("Theme", ["Light (Default)", "Dark", "System Match"])
    st.checkbox("Enable Email Notifications for Critical Findings", value=True)
    st.checkbox("Show Tooltips", value=True)
    
    if st.button("Save Preferences"):
        st.toast("Preferences saved", icon="✅")
        
    st.markdown('</div>', unsafe_allow_html=True)
