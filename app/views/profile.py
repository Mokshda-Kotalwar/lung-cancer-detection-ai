import streamlit as st

def render_profile(token):
    """Renders the user profile page."""
    st.markdown('<div class="premium-header">User Profile</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em; margin-bottom: 30px;'>Manage your account and preferences</p>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown('<div class="premium-card" style="text-align: center;">', unsafe_allow_html=True)
        # Mock Avatar
        st.markdown(f"""
        <div style="width: 120px; height: 120px; border-radius: 50%; background-color: #3b82f6; color: white; display: flex; align-items: center; justify-content: center; font-size: 48px; font-weight: bold; margin: 0 auto 20px auto;">
            DR
        </div>
        <h3 style="margin-bottom: 5px;">Dr. Radiologist</h3>
        <p style="color: #64748b; margin-top: 0;">Senior Specialist</p>
        <span class="status-badge status-success">Active Account</span>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_right:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### Account Information")
        
        st.text_input("Full Name", value="Dr. Radiologist", disabled=True)
        st.text_input("Email Address", value="doctor@hospital.com", disabled=True)
        st.text_input("Department", value="Oncology & Radiology", disabled=True)
        
        st.markdown("---")
        st.markdown("#### Change Password")
        st.text_input("Current Password", type="password")
        st.text_input("New Password", type="password")
        st.text_input("Confirm New Password", type="password")
        
        if st.button("Update Profile", type="primary"):
            st.success("Profile updated successfully (Mock).")
            
        st.markdown('</div>', unsafe_allow_html=True)
