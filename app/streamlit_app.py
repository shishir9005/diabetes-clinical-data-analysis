"""Educational Streamlit app for the diabetes questionnaire project.

Start it from the project folder:  streamlit run app/streamlit_app.py

The app only reads saved files (reports/, reports/figures/, models/). It does not train or tune anything, read
row-level data, write a file, store what is entered, or call an external service.
"""
import sys
from pathlib import Path

import streamlit as st

# Make the helper modules next to this file importable however the app is started.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app_common as c  # noqa: E402

st.set_page_config(
    page_title="Diabetes Questionnaire Study",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="auto",  # open on a desktop, collapsed on a phone so the page is visible first
)

st.title("Diabetes Questionnaire Data: a Leakage-Aware Classification Study")
st.caption("An educational data-science project on the UCI *Early Stage Diabetes Risk Prediction* dataset")
st.warning(c.DISCLAIMER, icon=":material/info:")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/home:", url_path="overview", default=True),
    st.Page("views/exploration.py", title="Exploratory analysis", icon=":material/bar_chart:", url_path="exploration"),
    st.Page("views/model_comparison.py", title="Model comparison", icon=":material/compare_arrows:", url_path="models"),
    st.Page("views/evaluation.py", title="Evaluation", icon=":material/fact_check:", url_path="evaluation"),
    st.Page("views/prediction.py", title="Educational prediction", icon=":material/tune:", url_path="prediction"),
]
navigation = st.navigation(pages)

with st.sidebar:
    st.caption(
        "Data: UCI Machine Learning Repository, *Early Stage Diabetes Risk Prediction*, "
        "[doi:10.24432/C5VG8H](https://doi.org/10.24432/C5VG8H), licence CC BY 4.0."
    )

navigation.run()
