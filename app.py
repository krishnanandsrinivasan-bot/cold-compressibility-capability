from __future__ import annotations

import streamlit as st

from ui import inject_css

st.set_page_config(
    page_title="Astemo Pad Development Tools",
    page_icon="🧰",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

pages = {
    "Workspace": [
        st.Page("pages/home.py", title="Dashboard", icon="🏠", default=True),
    ],
    "Process & Quality": [
        st.Page("pages/process_capability.py", title="Cpk / Ppk Analysis", icon="📊"),
        st.Page("pages/compressibility.py", title="Compressibility Analysis", icon="🗜️"),
    ],
    "Performance": [
        st.Page("pages/friction_analysis.py", title="Friction Analysis", icon="📈"),
        st.Page("pages/wear_analysis.py", title="Wear Analysis", icon="🛞"),
    ],
    "NVH & Emissions": [
        st.Page("pages/nvh_analysis.py", title="NVH Analysis", icon="🔊"),
        st.Page("pages/emission_analysis.py", title="Brake Emission Analysis", icon="🌫️"),
    ],
    "Engineering": [
        st.Page("pages/supplier_comparison.py", title="Supplier Comparison", icon="⚖️"),
        st.Page("pages/dvp_report.py", title="DVP / Report Generator", icon="📑"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
