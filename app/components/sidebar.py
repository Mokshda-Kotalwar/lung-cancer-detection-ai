import streamlit as st
from streamlit_option_menu import option_menu
from app.utils.styles import render_badge


def render_sidebar(backend_ok=True):
    """Render sidebar navigation."""

    pages = [
        "Dashboard",
        "Upload CT Scan",
        "Results",
        "History",
        "Analytics",
        "Profile",
        "Settings",
    ]

    icons = [
        "house",
        "cloud-upload",
        "clipboard-data",
        "clock-history",
        "bar-chart",
        "person",
        "gear",
    ]

    # Initialize session state
    if "active_view" not in st.session_state:
        st.session_state.active_view = "Dashboard"

    # Processing is an internal page, not part of sidebar
    current_page = st.session_state.active_view

    if current_page not in pages:
        current_page = "Upload CT Scan"

    with st.sidebar:

        st.markdown(
            """
            <div style="text-align:center;margin-bottom:25px;">
                <h2 style="color:#1e3a8a;margin-bottom:0;">🫁 LungAI</h2>
                <p style="color:#64748b;font-size:0.9rem;margin-top:4px;">
                    Clinical Decision Support
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title="Navigation",
            options=pages,
            icons=icons,
            menu_icon="cast",
            default_index=pages.index(current_page),
            key="main_navigation",
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",
                },
                "icon": {
                    "color": "#64748b",
                    "font-size": "17px",
                },
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "5px 0",
                    "padding": "10px",
                    "border-radius": "8px",
                    "--hover-color": "#eef4ff",
                    "color": "#475569",
                },
                "nav-link-selected": {
                    "background-color": "#1e3a8a",
                    "color": "white",
                    "font-weight": "600",
                },
            },
        )

        # Only update active_view if we're not inside internal pages
        if st.session_state.active_view not in ["Processing"]:
            st.session_state.active_view = selected

        st.divider()

        st.markdown(
            """
            <h4 style="
                font-size:0.85rem;
                color:#64748b;
                margin-bottom:12px;
                text-transform:uppercase;
            ">
            System Status
            </h4>
            """,
            unsafe_allow_html=True,
        )

        if backend_ok:
            render_badge("Backend: Online", "success")
        else:
            render_badge("Backend: Offline", "error")

        st.markdown("<br>", unsafe_allow_html=True)

        render_badge("AI Model: DenseNet121", "info")

        st.divider()

        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    return st.session_state.active_view