from __future__ import annotations

from datetime import date
from io import BytesIO
import math
import re
import inspect

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
from ui import hero


# Guard against partial GitHub updates where the V2 page is used with the old V1
# capability engine. Without this check Streamlit shows a cryptic TypeError.
_expected_apis = {
    "sequence_chart": (sequence_chart, 7),
    "distribution_chart": (distribution_chart, 9),
    "scenario_chart": (scenario_chart, 7),
    "prepare_results_table": (prepare_results_table, 7),
    "build_excel_report": (build_excel_report, 10),
}
_outdated = [
    name for name, (func, min_params) in _expected_apis.items()
    if len(inspect.signature(func).parameters) < min_params
]
if _outdated:
    st.error(
        "App file version mismatch: this V2 page is running with an older capability.py. "
        "Replace capability.py in the GitHub repository with the V2.0.1 file, then Streamlit will rebuild automatically."
    )
    st.caption("Outdated API detected: " + ", ".join(_outdated))
    st.stop()


hero(
    "Cpk / Ppk Analysis",
    "General process-capability analysis for cold compressibility or any numerical brake-pad characteristic. Import, paste or manually enter any number of measurements.",
)

# --------------------------- Test information ---------------------------
st.markdown("### Test information")
info1 = st.columns([1.25, 1, 1, 1])
project = info1[0].text_input("Project", placeholder="e.g. project / platform")
supplier = info1[1].text_input("Supplier", placeholder="e.g. supplier")
material = info1[2].text_input("Material / Grade", placeholder="e.g. friction material")
batch = info1[3].text_input("Batch / Lot", placeholder="e.g. batch number")

info2 = st.columns([1.4, .65, .9, 1.05])
characteristic = info2[0].text_input("Characteristic", value="Cold Compressibility")
unit = info2[1].text_input("Unit", value="µm")
test_date = info2[2].date_input("Test date", value=date.today())
source_note = info2[3].text_input("Test / source", placeholder="e.g. serial check, supplier data")

characteristic = characteristic.strip() or "Measurement"
unit = unit.strip()
unit_label = f" ({unit})" if unit else ""
unit_suffix = f" {unit}" if unit else ""

# --------------------------- Sidebar specification ---------------------------
with st.sidebar:
    st.divider()
    st.subheader("Capability setup")
    spec_mode = st.radio("Limit definition", ["Nominal ± tolerance", "Direct LSL / USL"], index=0)

    if spec_mode == "Nominal ± tolerance":
        target = st.number_input(f"Nominal / Target{unit_label}", value=100.0, step=1.0)
        tolerance = st.number_input(f"Tolerance ±{unit_label}", value=25.0, min_value=0.001, step=1.0)
        lsl = float(target - tolerance)
        usl = float(target + tolerance)
        st.caption(f"LSL = {lsl:g}{unit_suffix} · USL = {usl:g}{unit_suffix}")
    else:
        lsl = st.number_input(f"LSL{unit_label}", value=75.0, step=1.0)
        usl = st.number_input(f"USL{unit_label}", value=125.0, step=1.0)
        target_default = (lsl + usl) / 2 if usl > lsl else 100.0
        target = st.number_input(f"Target{unit_label}", value=float(target_default), step=1.0)

    st.divider()
    st.subheader("Acceptance criteria")
    cpk_req = st.number_input("Cpk requirement", value=1.67, min_value=0.0, step=0.01, format="%.2f")
    ppk_req = st.number_input("Ppk requirement", value=1.33, min_value=0.0, step=0.01, format="%.2f")

    method = st.selectbox(
        "Sigma method",
        ["Direct STDEV.S", "I-MR (within sigma = MR̄ / 1.128)"],
        index=0,
        help=(
            "Direct STDEV.S uses the sample standard deviation for both Cpk and Ppk. "
            "I-MR estimates within-process sigma from sequential moving ranges for Cpk while Ppk uses overall STDEV.S."
        ),
    )
    if method == "Direct STDEV.S":
        st.info("Direct mode: Cpk and Ppk use the same STDEV.S, therefore they are equal by definition.", icon="ℹ️")

# --------------------------- Data input ---------------------------
st.markdown("### Load measurement data")
input_mode = st.segmented_control(
    "Input method",
    options=["Excel / CSV", "Paste values", "Manual table", "Demo data"],
    default="Excel / CSV",
    selection_mode="single",
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
                    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")

elif input_mode == "Paste values":
    pasted = st.text_area(
        "Paste a column or list of values",
        height=210,
        placeholder="Example:\n98\n101\n103\n99\n...\n\nGerman decimal comma is accepted: 99,5",
    )
    values, ignored = parse_pasted_values(pasted)
    removed_count = len(ignored)
    if pasted.strip():
        input_message = f"Parsed {len(values):,} numeric values."
        if ignored:
            st.warning(f"Ignored {len(ignored)} non-numeric token(s): {', '.join(ignored[:8])}{'…' if len(ignored) > 8 else ''}")

elif input_mode == "Manual table":
    manual_col = f"{characteristic}{unit_label}"
    if "manual_capability_df" not in st.session_state:
        st.session_state.manual_capability_df = pd.DataFrame({"Measurement": [None] * 15})
    edited = st.data_editor(
        st.session_state.manual_capability_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={"Measurement": st.column_config.NumberColumn(manual_col, format="%.3f")},
        key="manual_capability_editor",
    )
    st.session_state.manual_capability_df = edited
    values, removed_count = clean_numeric(edited["Measurement"].tolist())
    input_message = f"Using {len(values):,} manually entered values."

else:
    rng = np.random.default_rng(42)
    demo_sigma = max((float(usl) - float(lsl)) / 10.5, 0.01)
    values = pd.Series(np.round(rng.normal(loc=float(target), scale=demo_sigma, size=100), 2))
    input_message = "Loaded a synthetic 100-measurement demo dataset."

if input_message:
    st.success(input_message)
if removed_count and input_mode != "Paste values":
    st.caption(f"Ignored {removed_count:,} blank or non-numeric cell(s).")

# --------------------------- Validation ---------------------------
if not (math.isfinite(float(lsl)) and math.isfinite(float(usl))) or lsl >= usl:
    st.error("USL must be greater than LSL before capability can be calculated.")
    st.stop()

if len(values) < 2:
    st.info("Load or enter at least two valid measurements. The tool has no fixed upper sample-size limit.")
    st.stop()

try:
    result = calculate_capability(values, float(lsl), float(usl), method=method)
except ValueError as exc:
    st.error(str(exc))
    st.stop()


def metric_value(x: float, decimals: int = 3) -> str:
    return "—" if not math.isfinite(float(x)) else f"{x:.{decimals}f}"


cpk_pass = bool(math.isfinite(result.cpk) and result.cpk >= cpk_req)
ppk_pass = bool(math.isfinite(result.ppk) and result.ppk >= ppk_req)

# --------------------------- Main summary ---------------------------
st.markdown("### Capability summary")
r1 = st.columns(6)
r1[0].metric("Measurements (N)", f"{result.n:,}")
r1[1].metric("Mean", f"{result.mean:.2f}{unit_suffix}")
r1[2].metric("Overall σ", f"{result.overall_std:.3f}{unit_suffix}")
r1[3].metric("Within σ", f"{result.within_std:.3f}{unit_suffix}")
r1[4].metric("Cpk", metric_value(result.cpk))
r1[5].metric("Ppk", metric_value(result.ppk))

status_cols = st.columns(2)
with status_cols[0]:
    cls = "pdt-status-pass" if cpk_pass else "pdt-status-fail"
    status = "PASS" if cpk_pass else "FAIL"
    st.markdown(f'<div class="{cls}">Cpk {result.cpk:.3f} — {status} &nbsp;·&nbsp; requirement ≥ {cpk_req:.2f}</div>', unsafe_allow_html=True)
with status_cols[1]:
    cls = "pdt-status-pass" if ppk_pass else "pdt-status-fail"
    status = "PASS" if ppk_pass else "FAIL"
    st.markdown(f'<div class="{cls}">Ppk {result.ppk:.3f} — {status} &nbsp;·&nbsp; requirement ≥ {ppk_req:.2f}</div>', unsafe_allow_html=True)

# --------------------------- Detailed tabs ---------------------------
tab_data, tab_dist, tab_cap, tab_what, tab_export = st.tabs([
    "Data & Sequence",
    "Distribution",
    "Capability Detail",
    "What-if",
    "Export",
])

with tab_data:
    st.plotly_chart(
        sequence_chart(values, lsl, usl, target, result.mean, characteristic, unit),
        use_container_width=True,
    )

    results_df = prepare_results_table(values, lsl, usl, result.mean, result.overall_std, characteristic, unit)
    value_col = f"{characteristic}{unit_label}"
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            value_col: st.column_config.NumberColumn(format="%.3f"),
            "Z from mean": st.column_config.NumberColumn(format="%.3f"),
        },
    )

with tab_dist:
    st.plotly_chart(
        distribution_chart(values, lsl, usl, target, result.mean, result.overall_std, characteristic, unit),
        use_container_width=True,
    )

    spread_width = 6 * result.overall_std
    spec_width = usl - lsl
    spread_share = 100 * spread_width / spec_width if spec_width > 0 else float("nan")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Observed 6σ width", f"{spread_width:.2f}{unit_suffix}")
    d2.metric("Specification width", f"{spec_width:.2f}{unit_suffix}")
    d3.metric("6σ / spec width", f"{spread_share:.1f}%")
    d4.metric("Actual outside spec", f"{result.outside_count} ({result.outside_percent:.2f}%)")

    with st.expander("Normality diagnostics"):
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Below LSL", result.below_lsl)
        n2.metric("Above USL", result.above_usl)
        n3.metric("Skewness", "—" if result.skewness is None else f"{result.skewness:.3f}")
        n4.metric("Excess kurtosis", "—" if result.excess_kurtosis is None else f"{result.excess_kurtosis:.3f}")
        if result.shapiro_p is not None:
            st.write(f"**Shapiro–Wilk p-value:** {result.shapiro_p:.4f}")
            if result.shapiro_p < 0.05:
                st.warning("The normality test suggests that a normal distribution may not represent these data well. Interpret normal-model capability and predicted ppm with care.")
            else:
                st.info("The test does not show strong evidence against normality at the 5% level. Use the plot and engineering knowledge as well; this is not proof of normality.")
        elif result.n > 5000:
            st.caption("Shapiro–Wilk is intentionally not calculated above 5,000 values in this tool.")

with tab_cap:
    r2 = st.columns(6)
    r2[0].metric("Cp", metric_value(result.cp))
    r2[1].metric("CPL", metric_value(result.cpl))
    r2[2].metric("CPU", metric_value(result.cpu))
    r2[3].metric("Pp", metric_value(result.pp))
    r2[4].metric("PPL", metric_value(result.ppl))
    r2[5].metric("PPU", metric_value(result.ppu))

    r3 = st.columns(4)
    r3[0].metric("Minimum", f"{result.minimum:.2f}{unit_suffix}")
    r3[1].metric("Maximum", f"{result.maximum:.2f}{unit_suffix}")
    r3[2].metric("Predicted outside", f"{result.predicted_ppm:,.0f} ppm")
    center_offset = result.mean - float(target)
    r3[3].metric("Mean vs target", f"{center_offset:+.2f}{unit_suffix}")

    st.markdown("#### Engineering interpretation")
    nearest_margin = min(result.mean - lsl, usl - result.mean)
    if result.overall_std > 0:
        nearest_sigma = nearest_margin / result.overall_std
    else:
        nearest_sigma = float("inf")
    st.write(
        f"The process mean is **{result.mean:.2f}{unit_suffix}**. The nearest specification limit is "
        f"**{nearest_margin:.2f}{unit_suffix}** away, equivalent to approximately **{nearest_sigma:.2f} overall σ**. "
        f"The observed 6σ distribution width uses approximately **{spread_share:.1f}%** of the total specification window."
    )
    if result.outside_count == 0 and (not cpk_pass or not ppk_pass):
        st.warning(
            "All measured parts can be inside the specification and the process can still fail capability. "
            "Part conformity asks whether the observed values are inside LSL/USL; capability also asks whether the process distribution is sufficiently narrow and centered to remain inside those limits consistently."
        )

    with st.expander("How this tool calculates Cpk and Ppk"):
        st.latex(r"C_p = \frac{USL-LSL}{6\sigma_{within}}")
        st.latex(r"C_{pk}=\min\left(\frac{USL-\bar{x}}{3\sigma_{within}},\frac{\bar{x}-LSL}{3\sigma_{within}}\right)")
        st.latex(r"P_p = \frac{USL-LSL}{6s_{overall}}")
        st.latex(r"P_{pk}=\min\left(\frac{USL-\bar{x}}{3s_{overall}},\frac{\bar{x}-LSL}{3s_{overall}}\right)")
        if result.method == "Direct STDEV.S":
            st.info("In Direct STDEV.S mode, σwithin = soverall = STDEV.S. Therefore Cp = Pp and Cpk = Ppk.")
        else:
            st.info("In I-MR mode, σwithin = MR̄ / 1.128 while Ppk uses the overall sample standard deviation.")

with tab_what:
    st.caption("Change the process center or variation to see what the capability would become without altering the original measurements.")
    what_cols = st.columns([1, 1, 1, 2])
    scenario_mean = what_cols[0].number_input(f"Scenario mean{unit_label}", value=float(result.mean), step=0.1, format="%.3f")
    scenario_std = what_cols[1].number_input(f"Scenario σ{unit_label}", value=max(float(result.overall_std), 0.001), min_value=0.001, step=0.1, format="%.3f")
    scenario_cpl = (scenario_mean - lsl) / (3 * scenario_std)
    scenario_cpu = (usl - scenario_mean) / (3 * scenario_std)
    scenario_cpk = min(scenario_cpl, scenario_cpu)
    scenario_ppm = (norm.cdf(lsl, loc=scenario_mean, scale=scenario_std) + 1 - norm.cdf(usl, loc=scenario_mean, scale=scenario_std)) * 1_000_000
    what_cols[2].metric("Scenario Cpk", f"{scenario_cpk:.3f}", delta=f"{scenario_cpk - result.cpk:+.3f} vs current")
    with what_cols[3]:
        q1, q2 = st.columns(2)
        q1.metric("Predicted outside", f"{scenario_ppm:,.0f} ppm")
        required_sigma = min(scenario_mean - lsl, usl - scenario_mean) / (3 * cpk_req) if cpk_req > 0 else float("nan")
        q2.metric(f"Max σ for Cpk {cpk_req:.2f}", "—" if not math.isfinite(required_sigma) or required_sigma <= 0 else f"{required_sigma:.3f}{unit_suffix}")

    st.plotly_chart(
        scenario_chart(lsl, usl, target, scenario_mean, scenario_std, characteristic, unit),
        use_container_width=True,
    )

with tab_export:
    metadata = {
        "Project": project,
        "Supplier": supplier,
        "Material / Grade": material,
        "Batch / Lot": batch,
        "Test date": test_date.isoformat() if test_date else "",
        "Test / source": source_note,
    }
    excel_report = build_excel_report(
        values,
        result,
        lsl,
        usl,
        target,
        cpk_req,
        ppk_req,
        characteristic=characteristic,
        unit=unit,
        metadata=metadata,
    )
    results_df = prepare_results_table(values, lsl, usl, result.mean, result.overall_std, characteristic, unit)
    csv_data = results_df.to_csv(index=False).encode("utf-8")

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", characteristic.lower()).strip("_") or "capability"
    e1, e2 = st.columns(2)
    e1.download_button(
        "Download capability report (.xlsx)",
        data=excel_report,
        file_name=f"{safe_name}_capability_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    e2.download_button(
        "Download cleaned measurements (.csv)",
        data=csv_data,
        file_name=f"{safe_name}_measurements.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("The Excel report includes the project/test metadata, specification, capability results and cleaned measurement table.")

st.divider()
st.caption(
    "Engineering note: capability indices are most meaningful for a stable process. The app does not impose a fixed sample count; "
    "the selected sigma method and distribution assumptions should match the measurement process and your quality standard."
)
