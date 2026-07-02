import streamlit as st

def apply_custom_css():
    """Injects premium enterprise healthcare-themed CSS into the Streamlit app."""
    
    # Check session state for theme
    theme = st.session_state.get("theme", "Light (Default)")
    is_dark = theme == "Dark"
    
    bg_color = "#0f172a" if is_dark else "#f8fafc"
    card_bg = "#1e293b" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    subtext_color = "#94a3b8" if is_dark else "#64748b"
    border_color = "#334155" if is_dark else "#e2e8f0"
    primary_color = "#3b82f6" if is_dark else "#1e3a8a"
    
    css = f"""
        <style>
        /* Google Fonts Import */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Global typography and background */
        html, body, [class*="css"] {{
            font-family: 'Outfit', sans-serif;
            background-color: {bg_color};
            color: {text_color};
        }}
        
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {bg_color};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {border_color};
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {subtext_color};
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {card_bg};
            border-right: 1px solid {border_color};
            box-shadow: 2px 0 15px rgba(0, 0, 0, 0.03);
        }}

        /* Glassmorphism Cards */
        .premium-card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
            border: 1px solid {border_color};
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        .premium-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
            border-color: {primary_color}40;
        }}

        /* KPI Card specific */
        .kpi-card {{
            text-align: center;
            padding: 20px;
        }}
        .kpi-label {{
            font-size: 0.85em;
            color: {subtext_color};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 12px;
        }}
        .kpi-value {{
            font-size: 2.5em;
            font-weight: 700;
            color: {primary_color};
            line-height: 1.1;
        }}
        .kpi-trend {{
            font-size: 0.85em;
            margin-top: 12px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            padding: 4px 10px;
            border-radius: 20px;
            background: rgba(0,0,0,0.03);
        }}
        .trend-up {{ color: #059669; background: rgba(16, 185, 129, 0.1); }}
        .trend-down {{ color: #dc2626; background: rgba(239, 68, 68, 0.1); }}

        /* Headers */
        .premium-header {{
            font-size: 2.5em;
            font-weight: 700;
            color: {text_color};
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }}
        
        .section-header {{
            font-size: 1.4em;
            font-weight: 600;
            color: {text_color};
            margin-top: 24px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .section-header::after {{
            content: '';
            flex-grow: 1;
            height: 1px;
            background: {border_color};
        }}

        /* Badges */
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85em;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .status-success {{ background-color: rgba(16, 185, 129, 0.15); color: #059669; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .status-warning {{ background-color: rgba(245, 158, 11, 0.15); color: #d97706; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .status-error {{ background-color: rgba(239, 68, 68, 0.15); color: #dc2626; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .status-info {{ background-color: rgba(59, 130, 246, 0.15); color: #2563eb; border: 1px solid rgba(59, 130, 246, 0.3); }}

        /* Hide Streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Button Styling */
        .stButton>button {{
            background-color: {primary_color};
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            border: none;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 15px {primary_color}40;
            border: none;
            color: white;
        }}
        .stButton>button:active {{
            transform: translateY(1px);
        }}
        
        /* Input Fields */
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {{
            border-radius: 8px;
            border: 1px solid {border_color};
            padding: 10px 14px;
            background-color: {card_bg};
            color: {text_color};
            transition: all 0.2s ease;
        }}
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {{
            border-color: {primary_color};
            box-shadow: 0 0 0 3px {primary_color}20;
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            color: {text_color};
            font-weight: 600;
        }}
        
        /* Dataframes */
        [data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {border_color};
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            background-color: {card_bg};
        }}
        
        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background-color: {primary_color};
            border-radius: 999px;
        }}
        .stProgress > div > div {{
            background-color: {border_color};
            border-radius: 999px;
            height: 8px;
        }}

        /* Animations */
        @keyframes pulse-ring {{
            0% {{ transform: scale(0.8); opacity: 0.5; }}
            100% {{ transform: scale(1.3); opacity: 0; }}
        }}
        .pulse-indicator {{
            position: relative;
            display: inline-flex;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #10b981;
        }}
        .pulse-indicator::before {{
            content: '';
            position: absolute;
            left: -5px;
            top: -5px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: #10b981;
            animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
        }}
        
        /* Skeleton Loading */
        .skeleton {{
            animation: skeleton-loading 1.5s linear infinite alternate;
            border-radius: 4px;
        }}
        @keyframes skeleton-loading {{
            0% {{ background-color: {border_color}80; }}
            100% {{ background-color: {border_color}30; }}
        }}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_badge(text, status="info"):
    """Render a styled badge."""
    st.markdown(f'<span class="status-badge status-{status}">{text}</span>', unsafe_allow_html=True)

def render_pulse_indicator(text):
    """Render a live status indicator with text."""
    st.markdown(f'<div style="display:flex; align-items:center; gap:8px;"><span class="pulse-indicator"></span><span style="font-size:0.9em; font-weight:500;">{text}</span></div>', unsafe_allow_html=True)
