import streamlit as st
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from app.utils.styles import apply_custom_css
from app.utils.api import check_backend_health
from app.components.sidebar import render_sidebar

from app.views.login import render_login
from app.views.dashboard import render_dashboard
from app.views.upload import render_upload
from app.views.processing import render_processing
from app.views.results import render_results
from app.views.history import render_history
from app.views.settings import render_settings
from app.views.analytics import render_analytics
from app.views.profile import render_profile


st.set_page_config(
    page_title="LungAI Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state."""

    defaults = {
        "token": None,
        "active_view": "Dashboard",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():

    apply_custom_css()
    init_session_state()

    # Login
    if st.session_state.token is None:
        render_login()
        return

    backend_ok = check_backend_health()

    # Sidebar handles navigation
    current_view = render_sidebar(backend_ok)

    # Preserve automatic page flow
    if st.session_state.active_view in ["Processing", "Results"]:
        current_view = st.session_state.active_view

    # Top Header
    col1, col2, col3 = st.columns([6, 1, 1])

    with col1:
        st.markdown(
            f"""
            <div style="
                padding-top:10px;
                color:#64748b;
                font-weight:500;
                font-size:15px;
            ">
                LungAI Platform /
                <span style="color:#1e3a8a;">
                    {current_view}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            "<div style='text-align:right;padding-top:10px;'>🔔</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            "<div style='text-align:right;padding-top:10px;'>⚙️</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Routing
    try:

        if current_view == "Dashboard":
            render_dashboard(st.session_state.token)

        elif current_view == "Upload CT Scan":
            render_upload(st.session_state.token)

        elif current_view == "Processing":
            render_processing(st.session_state.token)

        elif current_view == "Results":
            render_results(st.session_state.token)

        elif current_view == "History":
            render_history(st.session_state.token)

        elif current_view == "Analytics":
            render_analytics(st.session_state.token)

        elif current_view == "Profile":
            render_profile(st.session_state.token)

        elif current_view == "Settings":
            render_settings()

    except Exception:
        st.exception(traceback.format_exc())


if __name__ == "__main__":
    main()