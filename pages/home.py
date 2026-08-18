import streamlit as st

from ui import hero, tool_card

hero(
    "Pad Development Tools",
    "One engineering workspace for friction-pad development, process capability, performance, NVH, wear, emissions and supplier evaluation.",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Live tools", "1", help="Fully usable modules in this version")
c2.metric("Roadmap modules", "7")
c3.metric("Data input", "Excel · Paste · Manual")
c4.metric("Sample size", "Dynamic", help="No fixed 50/75/100/500-pad limit")

st.markdown("### Start here")
left, right = st.columns([1.6, 1])
with left:
    tool_card(
        "📊 Cpk / Ppk Analysis",
        "General process-capability tool for cold compressibility and other numerical pad characteristics. Import Excel/CSV, paste values, or enter data manually.",
        "Live",
    )
    st.page_link("pages/process_capability.py", label="Open Cpk / Ppk Analysis", icon="➡️", use_container_width=True)
with right:
    st.markdown(
        """
        <div class="pdt-card">
            <span class="pdt-badge-live">Platform concept</span>
            <h3>One app, multiple engineering tools</h3>
            <p>The capability calculator is now one module inside a larger pad-development workspace. Future tools can be added without creating a new website or changing the navigation structure.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Tool portfolio")
row1 = st.columns(4)
with row1[0]:
    tool_card("🗜️ Compressibility Analysis", "Cold/hot compressibility comparison, distributions, batch overlays and requirement evaluation.")
with row1[1]:
    tool_card("📈 Friction Analysis", "µ behaviour versus pressure, speed, temperature; fade/recovery and test comparison.")
with row1[2]:
    tool_card("🛞 Wear Analysis", "Pad/disc wear, side comparison, wear rate and projected service mileage.")
with row1[3]:
    tool_card("🔊 NVH Analysis", "Noise occurrence, frequency bands, side/configuration comparison and event visualization.")

row2 = st.columns(4)
with row2[0]:
    tool_card("🌫️ Brake Emission Analysis", "WLTP / Euro 7 PM analysis, cycle trends, repeatability and configuration comparison.")
with row2[1]:
    tool_card("⚖️ Supplier Comparison", "Compare materials, suppliers and batches using common engineering KPIs.")
with row2[2]:
    tool_card("📋 DVP Evaluation", "Requirement mapping, pass/fail status and test-program overview.")
with row2[3]:
    tool_card("📑 Report Generator", "Standardized engineering summaries and export-ready test reports.")

st.markdown("### Design principle")
st.write(
    "The platform separates the **engineering calculation engine** from the **test-specific modules**. "
    "For example, the Cpk/Ppk engine can be used for cold compressibility, thickness, density, weight or another numerical characteristic simply by changing the characteristic, unit and specification limits."
)

st.caption("Pad Development Tools · V2 platform shell")
