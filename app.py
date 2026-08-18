from __future__ import annotations

from io import BytesIO
import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from capability import (
    build_excel_report,
    calculate_capability,
    clean_numeric,
    distribution_chart,
    parse_pasted_values,
    prepare_results_table,
    scenario_chart,
    sequence_chart,
)

st.set_page_config(
    page_title="Cold Compressibility Capability",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stSidebar"] {border-right: 1px solid #e8edf2;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e4e9ef; border-radius: 12px;
        padding: 14px 16px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] {font-weight: 600;}
    .hero {
        padding: 18px 22px; border-radius: 16px;
        background: linear-gradient(120deg, #0f3557 0%, #1f5f8b 100%);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 {margin: 0 0 4px 0; font-size: 2rem;}
    .hero p {margin: 0; opacity: .9;}
    .status-pass {background:#e8f5e9; color:#1b5e20; border:1px solid #c8e6c9; padding:10px 14px; border-radius:10px; font-weight:700;}
    .status-fail {background:#ffebee; color:#b71c1c; border:1px solid #ffcdd2; padding:10px 14px; border-radius:10px; font-weight:700;}
    .small-note {color:#5b6573; font-size:.9rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Cold Compressibility — Cpk / Ppk Capability App</h1>
  <p>Import, paste or manually enter brake-pad compressibility data. Any sample size. Interactive capability analysis in µm.</p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------- Sidebar: specification ---------------------------
with st.sidebar:
    st.header("Specification")
    spec_mode = st.radio("Limit definition", ["Nominal ± tolerance", "Direct LSL / USL"], index=0)

    if spec_mode == "Nominal ± tolerance":
        target = st.number_input("Nominal / Target (µm)", value=140.0, step=1.0)
        tolerance = st.number_input("Tolerance ± (µm)", value=25.0, min_value=0.001, step=1.0)
        lsl = float(target - tolerance)
        usl = float(target + tolerance)
        st.caption(f"LSL = {lsl:g} µm · USL = {usl:g} µm")
    else:
        lsl = st.number_input("LSL (µm)", value=115.0, step=1.0)
        usl = st.number_input("USL (µm)", value=165.0, step=1.0)
        target_default = (lsl + usl) / 2 if usl > lsl else 140.0
        target = st.number_input("Target (µm)", value=float(target_default), step=1.0)

    st.divider()
    st.header("Capability criteria")
    cpk_req = st.number_input("Cpk requirement", value=1.67, min_value=0.0, step=0.01, format="%.2f")
    ppk_req = st.number_input("Ppk requirement", value=1.33, min_value=0.0, step=0.01, format="%.2f")

    method = st.selectbox(
        "Sigma method",
        ["Direct STDEV.S", "I-MR (within sigma = MR̄ / 1.128)"],
        index=0,
        help="Direct STDEV.S matches your current Excel approach. I-MR estimates within-process sigma from sequential moving ranges.",
    )
    if method == "Direct STDEV.S":
        st.info("Direct mode uses the same STDEV.S for Cpk and Ppk. Therefore Cpk and Ppk will be equal by definition.", icon="ℹ️")

# --------------------------- Data input ---------------------------
st.subheader("1 · Load measurement data")
input_mode = st.segmented_control(
    "Input method",
    options=["Excel / CSV", "Paste values", "Manual table", "Demo data"],
    default="Excel / CSV",
    selection_mode="single",
    required=True,
)

values = pd.Series(dtype=float)
removed_count = 0
input_message = ""

if input_mode == "Excel / CSV":
    uploaded = st.file_uploader("Upload measurement file", type=["xlsx", "xls", "csv", "txt"])
    if uploaded is not None:
        try:
            file_name = uploaded.name.lower()
            if file_name.endswith((".xlsx", ".xls")):
                excel_bytes = uploaded.getvalue()
                xls = pd.ExcelFile(BytesIO(excel_bytes))
                sheet = st.selectbox("Sheet", xls.sheet_names)
                df = pd.read_excel(BytesIO(excel_bytes), sheet_name=sheet)
            else:
                raw_bytes = uploaded.getvalue()
                df = pd.read_csv(BytesIO(raw_bytes), sep=None, engine="python")

            if df.empty:
                st.warning("The selected file/sheet is empty.")
            else:
                numeric_counts = {col: pd.to_numeric(df[col], errors="coerce").notna().sum() for col in df.columns}
                best_col = max(numeric_counts, key=numeric_counts.get)
                col_index = list(df.columns).index(best_col)
                selected_col = st.selectbox("Measurement column", df.columns.tolist(), index=col_index)
                values, removed_count = clean_numeric(df[selected_col].tolist())
                input_message = f"Loaded {len(values):,} numeric values from ‘{selected_col}’."
                with st.expander("Preview imported file"):
                    st.dataframe(df.head(100), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")

elif input_mode == "Paste values":
    pasted = st.text_area(
        "Paste a column or list of compressibility values",
        height=220,
        placeholder="Example:\n139\n140\n141\n145\n...\n\nGerman decimal comma is also accepted: 139,5",
    )
    values, ignored = parse_pasted_values(pasted)
    removed_count = len(ignored)
    if pasted.strip():
        input_message = f"Parsed {len(values):,} numeric values."
        if ignored:
            st.warning(f"Ignored {len(ignored)} non-numeric token(s): {', '.join(ignored[:8])}{'…' if len(ignored) > 8 else ''}")

elif input_mode == "Manual table":
    if "manual_df" not in st.session_state:
        st.session_state.manual_df = pd.DataFrame({"Compressibility (µm)": [None] * 15})
    edited = st.data_editor(
        st.session_state.manual_df,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={"Compressibility (µm)": st.column_config.NumberColumn(format="%.2f")},
        key="manual_editor",
    )
    st.session_state.manual_df = edited
    values, removed_count = clean_numeric(edited["Compressibility (µm)"].tolist())
    input_message = f"Using {len(values):,} manually entered values."

else:
    rng = np.random.default_rng(42)
    values = pd.Series(np.round(rng.normal(loc=140.0, scale=4.4, size=100), 1))
    input_message = "Loaded a synthetic 100-pad demo dataset (not taken from your uploaded workbook)."

if input_message:
    st.success(input_message)
if removed_count and input_mode != "Paste values":
    st.caption(f"Ignored {removed_count:,} blank or non-numeric cell(s).")

# --------------------------- Analysis ---------------------------
if not (math.isfinite(float(lsl)) and math.isfinite(float(usl))) or lsl >= usl:
    st.error("USL must be greater than LSL before capability can be calculated.")
    st.stop()

if len(values) < 2:
    st.info("Load or enter at least two valid measurements to start the analysis.")
    st.stop()

try:
    result = calculate_capability(values, float(lsl), float(usl), method=method)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("2 · Capability summary")

cpk_pass = bool(math.isfinite(result.cpk) and result.cpk >= cpk_req)
ppk_pass = bool(math.isfinite(result.ppk) and result.ppk >= ppk_req)

def metric_value(x, decimals=3):
    return "—" if not math.isfinite(float(x)) else f"{x:.{decimals}f}"

r1 = st.columns(6)
r1[0].metric("Pads / values (N)", f"{result.n:,}")
r1[1].metric("Mean", f"{result.mean:.2f} µm")
r1[2].metric("Overall σ (STDEV.S)", f"{result.overall_std:.3f} µm")
r1[3].metric("Within σ", f"{result.within_std:.3f} µm")
r1[4].metric("Cpk", metric_value(result.cpk))
r1[5].metric("Ppk", metric_value(result.ppk))

r2 = st.columns(6)
r2[0].metric("Cp", metric_value(result.cp))
r2[1].metric("Pp", metric_value(result.pp))
r2[2].metric("Minimum", f"{result.minimum:.2f} µm")
r2[3].metric("Maximum", f"{result.maximum:.2f} µm")
r2[4].metric("Outside spec", f"{result.outside_count} ({result.outside_percent:.2f}%)")
r2[5].metric("Predicted outside", f"{result.predicted_ppm:,.0f} ppm")

status_cols = st.columns(2)
with status_cols[0]:
    cls = "status-pass" if cpk_pass else "status-fail"
    st.markdown(f'<div class="{cls}">Cpk {result.cpk:.3f} — {"PASS" if cpk_pass else "FAIL"} (requirement ≥ {cpk_req:.2f})</div>', unsafe_allow_html=True)
with status_cols[1]:
    cls = "status-pass" if ppk_pass else "status-fail"
    st.markdown(f'<div class="{cls}">Ppk {result.ppk:.3f} — {"PASS" if ppk_pass else "FAIL"} (requirement ≥ {ppk_req:.2f})</div>', unsafe_allow_html=True)

if result.method == "Direct STDEV.S":
    st.caption("Direct STDEV.S mode: Cpk and Ppk use the same sample standard deviation, matching the approach in your supplied Excel tool.")
else:
    st.caption("I-MR mode: Cpk uses within sigma estimated from the average moving range; Ppk uses overall STDEV.S.")

st.subheader("3 · Visual analysis")
chart_cols = st.columns(2)
with chart_cols[0]:
    st.plotly_chart(sequence_chart(values, lsl, usl, target, result.mean), width="stretch")
with chart_cols[1]:
    st.plotly_chart(distribution_chart(values, lsl, usl, target, result.mean, result.overall_std), width="stretch")

with st.expander("Distribution diagnostics", expanded=False):
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Below LSL", result.below_lsl)
    d2.metric("Above USL", result.above_usl)
    d3.metric("Skewness", "—" if result.skewness is None else f"{result.skewness:.3f}")
    d4.metric("Excess kurtosis", "—" if result.excess_kurtosis is None else f"{result.excess_kurtosis:.3f}")
    if result.shapiro_p is not None:
        st.write(f"**Shapiro–Wilk normality p-value:** {result.shapiro_p:.4f}")
        if result.shapiro_p < 0.05:
            st.warning("The normality test suggests the data may not be well represented by a normal distribution. Interpret normal-model capability and predicted ppm with care.")
        else:
            st.info("The normality test does not show strong evidence against normality at the 5% level. This is not proof that the process is normal; use the plot and engineering knowledge too.")

st.subheader("4 · Capability what-if")
st.caption("Use this to show a supplier or colleague how process centering and variation independently change capability.")
what_cols = st.columns([1, 1, 2.5])
scenario_mean = what_cols[0].number_input("Scenario mean (µm)", value=float(result.mean), step=0.1, format="%.2f")
scenario_std = what_cols[1].number_input("Scenario σ (µm)", value=max(float(result.overall_std), 0.01), min_value=0.001, step=0.1, format="%.3f")
scenario_cpl = (scenario_mean - lsl) / (3 * scenario_std)
scenario_cpu = (usl - scenario_mean) / (3 * scenario_std)
scenario_cpk = min(scenario_cpl, scenario_cpu)
scenario_ppm = (norm.cdf(lsl, loc=scenario_mean, scale=scenario_std) + 1 - norm.cdf(usl, loc=scenario_mean, scale=scenario_std)) * 1_000_000
with what_cols[2]:
    w1, w2 = st.columns(2)
    w1.metric("Scenario Cpk", f"{scenario_cpk:.3f}", delta=f"{scenario_cpk - result.cpk:+.3f} vs current")
    w2.metric("Predicted outside", f"{scenario_ppm:,.0f} ppm")
st.plotly_chart(scenario_chart(lsl, usl, target, scenario_mean, scenario_std), width="stretch")

st.subheader("5 · Measurement table & export")
results_df = prepare_results_table(values, lsl, usl, result.mean, result.overall_std)
st.dataframe(
    results_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Compressibility (µm)": st.column_config.NumberColumn(format="%.2f"),
        "Z from mean": st.column_config.NumberColumn(format="%.3f"),
    },
)

excel_report = build_excel_report(values, result, lsl, usl, target, cpk_req, ppk_req)
csv_data = results_df.to_csv(index=False).encode("utf-8")

ex1, ex2 = st.columns(2)
ex1.download_button(
    "Download capability report (.xlsx)",
    data=excel_report,
    file_name="cold_compressibility_capability_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch",
)
ex2.download_button(
    "Download cleaned measurements (.csv)",
    data=csv_data,
    file_name="cold_compressibility_cleaned_data.csv",
    mime="text/csv",
    width="stretch",
)

st.divider()
st.caption("Engineering note: capability indices assume a stable process and are most meaningful when the selected sigma method and distribution model match the way the measurements were produced. The app does not impose a fixed sample count.")
