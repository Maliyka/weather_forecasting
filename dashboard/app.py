"""
Global Weather Repository — Analytical Dashboard
=====================================================
Principal entrypoint. Run with:

    streamlit run dashboard/app.py

from the project root (so `src` and `dashboard` resolve as packages).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))  # repo root on sys.path

from dashboard.components import diagnostic_eda, environmental, kpis, ml_diagnostics, spatial  # noqa: E402
from dashboard.utils.data_loader import load_data  # noqa: E402
from dashboard.utils.styling import apply_corporate_theme  # noqa: E402

# --------------------------------------------------------------------------- #
# PM ACCELERATOR MISSION STATEMENT
# --------------------------------------------------------------------------- #
# NOTE FOR SUBMITTER: Paste PM Accelerator's official mission statement
# VERBATIM here, copied directly from their LinkedIn "About" section or
# https://www.pmaccelerator.io — exact wording matters for this assessment
# requirement. The text below is a placeholder paraphrase, not the official
# copy, and should be replaced before submission.
PM_ACCELERATOR_MISSION_STATEMENT = (
    "PM Accelerator's mission is to help people break into and accelerate "
    "their product management careers through hands-on training, mentorship, "
    "and real-world project experience — for professionals at every stage, "
    "from aspiring PMs to current leaders aiming for Director-level roles. "
    "By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills."
)

st.set_page_config(
    page_title="Global Weather Repository — Analytical Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_corporate_theme()


def render_header() -> None:
    st.title("🌍 Global Weather Repository — Analytical Dashboard")
    st.markdown(
        """
        <div style="background-color:#F0F4F8; border-left:6px solid #1F4E79;
                    padding:16px 20px; border-radius:6px; margin-bottom:1.2rem;">
            <span style="font-weight:700; color:#1F4E79; font-size:1.05rem;">
                🚀 PM Accelerator Mission Statement
            </span>
            <p style="margin-top:8px; margin-bottom:0; color:#333333; line-height:1.5;">
                {mission}
            </p>
        </div>
        """.format(mission=PM_ACCELERATOR_MISSION_STATEMENT),
        unsafe_allow_html=True,
    )
    st.caption(
        "Tech Assessment — Advanced Track | Global Weather Repository (Kaggle) | "
        "Built with Streamlit, Plotly, scikit-learn, XGBoost, Prophet, and SHAP."
    )


def render_sidebar() -> None:
    st.sidebar.header("📊 About")
    st.sidebar.markdown(
        "**Dataset:** Global Weather Repository\n\n"
        "Daily weather and air-quality readings for cities worldwide."
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Navigation**\n\n"
        "Use the tabs at the top of the page to move between Executive KPIs, "
        "Diagnostic EDA, Advanced ML Diagnostics, Spatial Insights, and the "
        "Environmental Matrix."
    )


def main() -> None:
    render_header()
    render_sidebar()

    try:
        df = load_data()
    except Exception as e:  # noqa: BLE001 — last-resort guard; load_data already handles most cases
        st.error(f"Fatal error while initializing the dashboard: {e}")
        st.stop()

    if df.empty:
        st.warning(
            "No data is currently loaded. Confirm `GlobalWeatherRepository.csv` is present in `data/`."
        )
        st.stop()

    tab_labels = [
        "📊 Executive KPIs",
        "🔍 Diagnostic EDA",
        "🤖 Advanced ML Diagnostics",
        "🌍 Spatial Insights",
        "🧪 Environmental Matrix",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        kpis.render(df)
    with tabs[1]:
        diagnostic_eda.render(df)
    with tabs[2]:
        ml_diagnostics.render(df)
    with tabs[3]:
        spatial.render(df)
    with tabs[4]:
        environmental.render(df)

    st.divider()
    st.caption(
        "Built for the PM Accelerator Technical Assessment — Advanced Track. "
        "Dataset: Nelgiriyewithana, *World Weather Repository*, Kaggle."
    )


if __name__ == "__main__":
    main()
