import streamlit as st

def apply_custom_css():
    """Injects premium healthcare-themed CSS into the Streamlit app."""
    st.markdown("""
        <style>
        /* Google Fonts Import */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Global typography and background */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }
        
        .stApp {
            background-color: #f8fafc; /* Light gray background */
            color: #0f172a;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
        }

        /* Glassmorphism Cards */
        .premium-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            border: 1px solid #f1f5f9;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        
        .premium-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        }

        /* KPI Card specific */
        .kpi-card {
            text-align: center;
        }
        .kpi-label {
            font-size: 0.9em;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 2.2em;
            font-weight: 700;
            color: #1e3a8a; /* Deep Blue */
        }
        .kpi-trend {
            font-size: 0.8em;
            margin-top: 4px;
        }
        .trend-up { color: #0f766e; } /* Teal */
        .trend-down { color: #dc2626; } /* Red */

        /* Headers */
        .premium-header {
            font-size: 2.5em;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 12px;
        }
        
        .section-header {
            font-size: 1.5em;
            font-weight: 600;
            color: #334155;
            margin-top: 16px;
            margin-bottom: 16px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
        }

        /* Badges */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .status-success { background-color: #d1fae5; color: #065f46; border: 1px solid #34d399; }
        .status-warning { background-color: #fef3c7; color: #92400e; border: 1px solid #fbbf24; }
        .status-error { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
        .status-info { background-color: #dbeafe; color: #1e40af; border: 1px solid #60a5fa; }

        /* Login Screen specific */
        .login-container {
            max-width: 450px;
            margin: 10vh auto;
        }
        
        .login-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(30, 58, 138, 0.1);
            border: 1px solid #e2e8f0;
        }
        
        .login-title {
            text-align: center;
            color: #1e3a8a;
            font-weight: 700;
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .login-subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 30px;
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Button Styling */
        .stButton>button {
            background-color: #1e3a8a;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            border: none;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            background-color: #172554;
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
            border: none;
            color: white;
        }
        
        /* Secondary Button (Warning/Danger) Removed to prevent white-on-white text */
        
        /* Input Fields */
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
            border-radius: 8px;
            border: 1px solid #cbd5e1;
            padding: 10px;
            background-color: #ffffff;
        }
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
            border-color: #1e3a8a;
            box-shadow: 0 0 0 2px rgba(30, 58, 138, 0.2);
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            color: #1e3a8a;
            font-weight: 600;
        }
        
        /* Dataframes */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background-color: #0f766e;
        }
        </style>
    """, unsafe_allow_html=True)

def render_badge(text, status="info"):
    """Render a styled badge."""
    st.markdown(f'<span class="status-badge status-{status}">{text}</span>', unsafe_allow_html=True)
