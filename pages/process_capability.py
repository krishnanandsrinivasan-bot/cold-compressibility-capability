from __future__ import annotations

from datetime import date
from io import BytesIO
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from capability import (
    calculate_capability,
    clean_numeric,
    distribution_chart,
    parse_pasted_values,
    prepare_results_table,
    scenario_chart,
    sequence_chart,
)
from ui import hero
from report_pdf import build_capability_pdf


hero(
    "Process Capability",
    "PPAP/process-batch Cpk and cumulative serial-production Ppk in one simple workflow.",
)

study_type = st.segmented_control(
    "Study type",
    ["Process Batch / PPAP", "Serial Production / Cumulative"],
    default="Process Batch / PPAP",
    selection_mode="single",
)
is_serial = study_type == "Serial Production / Cumulative"

if is_serial:
    st.info(
        "**Serial Production → Ppk ≥ 1.33.** Ppk is calculated from the overall variation of all selected comparable measurements. "
        "Use the batch selector to decide which PPAP, survey and serial lots belong to the cumulative history."
    )
else:
    st.info(
        "**Process Batch / PPAP → Cpk ≥ 1.67.** Cpk uses the short-term within-process variation estimated from sequential measurements. "
        "Keep the measurements in production/test order when possible."
    )

# --------------------------- Test information ---------------------------
st.markdown("### Test information")
i1 = st.columns([1.2, 1, 1, 1])
project = i1[0].text_input("Project", placeholder="e.g. VS20 / VAN.EA")
supplier = i1[1].text_input("Supplier", placeholder="e.g. TMD / ITT")
material = i1[2].text_input("Material / Grade", placeholder="e.g. PA4821 V30")
scope_or_batch = i1[3].text_input("History / Scope" if is_serial else "Batch / Lot", placeholder="e.g. PPAP + Survey 1-5" if is_serial else "e.g. PPAP batch")

i2 = st.columns([1.4, .65, .9, 1.05])
characteristic = (i2[0].text_input("Characteristic", value="Cold Compressibility").strip() or "Measurement")
unit = i2[1].text_input("Unit", value="µm").strip()
evaluation_date = i2[2].date_input("Evaluation date", value=date.today())
source_note = i2[3].text_input("Test / source", placeholder="e.g. supplier measurement file")
unit_label = f" ({unit})" if unit else ""
unit_suffix = f" {unit}" if unit else ""

# --------------------------- Setup ---------------------------
with st.sidebar:
    st.divider()
    st.subheader("Specification")
    spec_mode = st.radio("Limit definition", ["Nominal ± tolerance", "Direct LSL / USL"], index=0)
    if spec_mode == "Nominal ± tolerance":
        target = st.number_input(f"Nominal / Target{unit_label}", value=100.0, step=1.0)
        tolerance = st.number_input(f"Tolerance ±{unit_label}", value=25.0, min_value=0.001, step=1.0)
        lsl, usl = float(target - tolerance), float(target + tolerance)
        st.caption(f"LSL {lsl:g}{unit_suffix} · USL {usl:g}{unit_suffix}")
    else:
        lsl = st.number_input(f"LSL{unit_label}", value=75.0, step=1.0)
        usl = st.number_input(f"USL{unit_label}", value=125.0, step=1.0)
        target = st.number_input(f"Target{unit_label}", value=float((lsl + usl) / 2), step=1.0)

    st.divider()
    st.subheader("Acceptance")
    if is_serial:
        requirement = st.number_input("Ppk requirement", value=1.33, min_value=0.0, step=0.01, format="%.2f")
        st.caption("Serial Production · cumulative long-term performance")
    else:
        requirement = st.number_input("Cpk requirement", value=1.67, min_value=0.0, step=0.01, format="%.2f")
        st.caption("Process Batch / PPAP · short-term capability")

# --------------------------- Input ---------------------------
st.markdown("### Load measurement data")
input_mode = st.segmented_control(
    "Input method",
    ["Excel / CSV", "Paste values", "Manual table", "Demo data"],
    default="Excel / CSV",
    selection_mode="single",
)

values = pd.Series(dtype=float)
batch_labels: pd.Series | None = None
removed = 0
message = ""

if input_mode == "Excel / CSV":
    uploaded = st.file_uploader("Upload measurement file", type=["xlsx", "xls", "csv", "txt"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith((".xlsx", ".xls")):
                raw = uploaded.getvalue()
                xls = pd.ExcelFile(BytesIO(raw))
                sheet = st.selectbox("Sheet", xls.sheet_names)
                df = pd.read_excel(BytesIO(raw), sheet_name=sheet)
            else:
                df = pd.read_csv(BytesIO(uploaded.getvalue()), sep=None, engine="python")

            if df.empty:
                st.warning("The selected file/sheet is empty.")
            else:
                numeric_counts = {c: pd.to_numeric(df[c], errors="coerce").notna().sum() for c in df.columns}
                best = max(numeric_counts, key=numeric_counts.get)
                measure_col = st.selectbox("Measurement column", list(df.columns), index=list(df.columns).index(best))
                numeric = pd.to_numeric(df[measure_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                valid = numeric.notna()
                removed = int((~valid).sum())
                working = pd.DataFrame({"Measurement": numeric[valid].astype(float).reset_index(drop=True)})

                if is_serial:
                    other_cols = [c for c in df.columns if c != measure_col]
                    batch_col = st.selectbox(
                        "Batch / Lot column",
                        [None] + other_cols,
                        format_func=lambda x: "— No batch column —" if x is None else str(x),
                        help="Recommended for cumulative serial monitoring. The batch order in the file is preserved.",
                    )
                    if batch_col is not None:
                        labels = df.loc[valid, batch_col].reset_index(drop=True)
                        labels = labels.where(labels.notna(), "Unlabelled").astype(str).str.strip().replace("", "Unlabelled")
                        working["Batch / Lot"] = labels
                        ordered_batches = list(pd.unique(labels))
                        selected_batches = st.multiselect("Batches included in cumulative Ppk", ordered_batches, default=ordered_batches)
                        working = working[working["Batch / Lot"].isin(selected_batches)].reset_index(drop=True)
                        batch_labels = working["Batch / Lot"].copy()
                    else:
                        batch_labels = pd.Series(["Cumulative dataset"] * len(working), dtype="object")
                values = working["Measurement"].reset_index(drop=True)
                message = f"Loaded {len(values):,} numeric values from ‘{measure_col}’."
                with st.expander("Preview imported file"):
                    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")

elif input_mode == "Paste values":
    pasted = st.text_area("Paste a column or list of values", height=210, placeholder="98\n101\n103\n99\n...\n\nGerman decimal comma is accepted: 99,5")
    values, ignored = parse_pasted_values(pasted)
    removed = len(ignored)
    if pasted.strip():
        message = f"Parsed {len(values):,} numeric values."
        if ignored:
            st.warning(f"Ignored {len(ignored)} non-numeric token(s).")
    if is_serial and len(values):
        with st.expander("Optional batch labels"):
            labels_text = st.text_area("One batch / lot label per measurement, same order", key="serial_labels_v211")
            if labels_text.strip():
                labels = [x.strip() or "Unlabelled" for x in labels_text.replace("\r", "").split("\n")]
                if len(labels) == len(values):
                    batch_labels = pd.Series(labels, dtype="object")
                else:
                    st.warning(f"{len(labels)} batch labels for {len(values)} measurements. Labels are ignored until counts match.")
            if batch_labels is None:
                batch_labels = pd.Series(["Cumulative dataset"] * len(values), dtype="object")

elif input_mode == "Manual table":
    if is_serial:
        key = "manual_serial_v211"
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame({"Batch / Lot": [""] * 15, "Measurement": [None] * 15})
        edited = st.data_editor(
            st.session_state[key], num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={"Measurement": st.column_config.NumberColumn(f"{characteristic}{unit_label}", format="%.3f")},
            key="manual_serial_editor_v211",
        )
        st.session_state[key] = edited
        numeric = pd.to_numeric(edited["Measurement"], errors="coerce")
        valid = numeric.notna()
        values = numeric[valid].astype(float).reset_index(drop=True)
        labels = edited.loc[valid, "Batch / Lot"].fillna("").astype(str).str.strip().replace("", "Unlabelled")
        batch_labels = labels.reset_index(drop=True)
        removed = int((~valid).sum())
    else:
        key = "manual_ppap_v211"
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame({"Measurement": [None] * 15})
        edited = st.data_editor(
            st.session_state[key], num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={"Measurement": st.column_config.NumberColumn(f"{characteristic}{unit_label}", format="%.3f")},
            key="manual_ppap_editor_v211",
        )
        st.session_state[key] = edited
        values, removed = clean_numeric(edited["Measurement"].tolist())
    message = f"Using {len(values):,} manually entered values."

else:
    rng = np.random.default_rng(42)
    if is_serial:
        names = ["PPAP", "Survey 1", "Survey 2", "SOP Lot 01", "SOP Lot 02"]
        means = [99.0, 99.7, 100.6, 101.1, 100.4]
        chunks = [np.round(rng.normal(m, 4.8, 60), 2) for m in means]
        values = pd.Series(np.concatenate(chunks), dtype=float)
        batch_labels = pd.Series(np.repeat(names, 60), dtype="object")
        message = "Loaded a five-batch demo history with 300 measurements."
    else:
        sigma = max((float(usl) - float(lsl)) / 11.0, 0.01)
        values = pd.Series(np.round(rng.normal(float(target), sigma, 100), 2))
        message = "Loaded a 100-measurement PPAP demo dataset."

if message:
    st.success(message)
if removed and input_mode != "Paste values":
    st.caption(f"Ignored {removed:,} blank or non-numeric cell(s).")

if not (math.isfinite(float(lsl)) and math.isfinite(float(usl))) or lsl >= usl:
    st.error("USL must be greater than LSL.")
    st.stop()
if len(values) < 2:
    st.info("Load or enter at least two valid measurements. There is no fixed upper sample-size limit.")
    st.stop()
if not is_serial and len(values) < 3:
    st.info("PPAP Cpk needs at least three sequential measurements for the within-process estimate.")
    st.stop()

method = "Direct STDEV.S" if is_serial else "I-MR (within sigma = MR̄ / 1.128)"
try:
    result = calculate_capability(values, float(lsl), float(usl), method=method)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

primary_name = "Ppk" if is_serial else "Cpk"
primary_value = result.ppk if is_serial else result.cpk
analysis_std = result.overall_std if is_serial else result.within_std
passed = bool(math.isfinite(primary_value) and primary_value >= requirement)

# --------------------------- Summary ---------------------------
st.markdown("### Capability summary")
summary = st.columns(6)
summary[0].metric("Measurements (N)", f"{result.n:,}")
summary[1].metric("Batches included" if is_serial and batch_labels is not None else "Mean", f"{batch_labels.nunique():,}" if is_serial and batch_labels is not None else f"{result.mean:.2f}{unit_suffix}")
summary[2].metric("Mean" if is_serial else "Within σ", f"{result.mean:.2f}{unit_suffix}" if is_serial else f"{result.within_std:.3f}{unit_suffix}")
summary[3].metric("Overall σ" if is_serial else "Specification width", f"{result.overall_std:.3f}{unit_suffix}" if is_serial else f"{usl-lsl:.2f}{unit_suffix}")
summary[4].metric(primary_name, f"{primary_value:.3f}" if math.isfinite(primary_value) else "—")
summary[5].metric("Outside specification", f"{result.outside_count} ({result.outside_percent:.2f}%)")

css_class = "pdt-status-pass" if passed else "pdt-status-fail"
st.markdown(
    f'<div class="{css_class}">{study_type}: {primary_name} {primary_value:.3f} — {"PASS" if passed else "FAIL"} &nbsp;·&nbsp; requirement ≥ {requirement:.2f}</div>',
    unsafe_allow_html=True,
)

# --------------------------- Batch history ---------------------------
def batch_history_table(labels: pd.Series, data: pd.Series) -> pd.DataFrame:
    work = pd.DataFrame({"Batch / Lot": labels.astype(str).reset_index(drop=True), "Measurement": data.reset_index(drop=True)})
    cumulative: list[float] = []
    rows: list[dict] = []
    for label in pd.unique(work["Batch / Lot"]):
        batch_vals = work.loc[work["Batch / Lot"] == label, "Measurement"].astype(float).tolist()
        cumulative.extend(batch_vals)
        r = calculate_capability(cumulative, float(lsl), float(usl), method="Direct STDEV.S") if len(cumulative) >= 2 else None
        rows.append({
            "Batch / Lot": label,
            "Batch N": len(batch_vals),
            "Batch mean": float(np.mean(batch_vals)),
            "Cumulative N": len(cumulative),
            "Cumulative mean": float(np.mean(cumulative)),
            "Cumulative Ppk": np.nan if r is None else r.ppk,
        })
    return pd.DataFrame(rows)

history = batch_history_table(batch_labels, values) if is_serial and batch_labels is not None and len(batch_labels) == len(values) else None

# --------------------------- Tabs ---------------------------
if is_serial:
    tab_data, tab_dist, tab_hist, tab_detail, tab_what, tab_export = st.tabs(["Data & Sequence", "Distribution", "Batch History", "Capability Detail", "What-if", "Export"])
else:
    tab_data, tab_dist, tab_detail, tab_what, tab_export = st.tabs(["Data & Sequence", "Distribution", "Capability Detail", "What-if", "Export"])
    tab_hist = None

with tab_data:
    st.plotly_chart(sequence_chart(values, lsl, usl, target, result.mean, characteristic, unit), use_container_width=True)
    results_df = prepare_results_table(values, lsl, usl, result.mean, result.overall_std, characteristic, unit)
    if is_serial and batch_labels is not None and len(batch_labels) == len(results_df):
        results_df.insert(1, "Batch / Lot", batch_labels.reset_index(drop=True))
    st.dataframe(results_df, use_container_width=True, hide_index=True)

with tab_dist:
    st.plotly_chart(distribution_chart(values, lsl, usl, target, result.mean, result.overall_std, characteristic, unit), use_container_width=True)
    width = 6 * (result.overall_std if is_serial else result.within_std)
    spec_width = usl - lsl
    c = st.columns(4)
    c[0].metric("6σ width", f"{width:.2f}{unit_suffix}")
    c[1].metric("Specification width", f"{spec_width:.2f}{unit_suffix}")
    c[2].metric("6σ / spec width", f"{100*width/spec_width:.1f}%")
    c[3].metric("Actual outside spec", f"{result.outside_count} ({result.outside_percent:.2f}%)")

if is_serial and tab_hist is not None:
    with tab_hist:
        st.markdown("#### Cumulative Ppk by batch")
        st.caption("Each point recalculates Ppk after adding the next selected batch. The final point equals the current cumulative Ppk.")
        if history is None or history.empty:
            st.info("Add batch labels to see cumulative Ppk progression.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)
            fig = go.Figure(go.Scatter(x=history["Batch / Lot"], y=history["Cumulative Ppk"], mode="lines+markers", name="Cumulative Ppk", line=dict(width=3), marker=dict(size=9)))
            fig.add_hline(y=requirement, line_dash="dash", line_color="#d62728", annotation_text=f"Requirement {requirement:.2f}")
            fig.update_layout(title="Cumulative Ppk Progression", xaxis_title="Batch / Lot in imported order", yaxis_title="Ppk", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.warning("Combine batches only when they represent the same comparable process, characteristic and specification.")

with tab_detail:
    cols = st.columns(4)
    if is_serial:
        cols[0].metric("Pp", f"{result.pp:.3f}")
        cols[1].metric("PPL", f"{result.ppl:.3f}")
        cols[2].metric("PPU", f"{result.ppu:.3f}")
        cols[3].metric("Ppk", f"{result.ppk:.3f}")
    else:
        cols[0].metric("Cp", f"{result.cp:.3f}")
        cols[1].metric("CPL", f"{result.cpl:.3f}")
        cols[2].metric("CPU", f"{result.cpu:.3f}")
        cols[3].metric("Cpk", f"{result.cpk:.3f}")

    st.markdown("#### Engineering interpretation")
    nearest = min(result.mean - lsl, usl - result.mean)
    sigma_distance = nearest / analysis_std if analysis_std > 0 else float("inf")
    if is_serial:
        st.write(f"The selected cumulative history contains **{result.n:,} measurements**. Mean = **{result.mean:.2f}{unit_suffix}**, overall σ = **{result.overall_std:.3f}{unit_suffix}**, and the nearest specification limit is **{sigma_distance:.2f} overall σ** from the mean.")
    else:
        st.write(f"This process batch contains **{result.n:,} measurements**. Mean = **{result.mean:.2f}{unit_suffix}**, within σ = **{result.within_std:.3f}{unit_suffix}**, and the nearest specification limit is **{sigma_distance:.2f} within σ** from the mean.")

    with st.expander(f"How {primary_name} is calculated"):
        if is_serial:
            st.latex(r"P_{pk}=\min\left(\frac{USL-\bar{x}}{3s_{overall}},\frac{\bar{x}-LSL}{3s_{overall}}\right)")
            st.info("Serial Production uses STDEV.S of every included measurement in the cumulative dataset.")
        else:
            st.latex(r"\sigma_{within}=\frac{\overline{MR}}{1.128}")
            st.latex(r"C_{pk}=\min\left(\frac{USL-\bar{x}}{3\sigma_{within}},\frac{\bar{x}-LSL}{3\sigma_{within}}\right)")
            st.info("PPAP/process-batch Cpk uses an I-MR estimate of short-term within-process variation. Measurement order therefore matters.")

with tab_what:
    w = st.columns([1, 1, 1, 2])
    scenario_mean = w[0].number_input(f"Scenario mean{unit_label}", value=float(result.mean), step=0.1, format="%.3f")
    scenario_std = w[1].number_input(f"Scenario {'overall σ' if is_serial else 'within σ'}{unit_label}", value=max(float(analysis_std), 0.001), min_value=0.001, step=0.1, format="%.3f")
    scenario_value = min((scenario_mean-lsl)/(3*scenario_std), (usl-scenario_mean)/(3*scenario_std))
    w[2].metric(f"Scenario {primary_name}", f"{scenario_value:.3f}", delta=f"{scenario_value-primary_value:+.3f} vs current")
    ppm = (norm.cdf(lsl, loc=scenario_mean, scale=scenario_std) + 1 - norm.cdf(usl, loc=scenario_mean, scale=scenario_std)) * 1_000_000
    with w[3]:
        q1, q2 = st.columns(2)
        q1.metric("Predicted outside", f"{ppm:,.0f} ppm")
        max_sigma = min(scenario_mean-lsl, usl-scenario_mean)/(3*requirement) if requirement > 0 else float("nan")
        q2.metric(f"Max σ for {primary_name} {requirement:.2f}", "—" if not math.isfinite(max_sigma) or max_sigma <= 0 else f"{max_sigma:.3f}{unit_suffix}")
    st.plotly_chart(scenario_chart(lsl, usl, target, scenario_mean, scenario_std, characteristic, unit), use_container_width=True)

with tab_export:
    def build_report() -> bytes:
        metadata_rows = [
            ["Study type", study_type], ["Project", project], ["Supplier", supplier], ["Material / Grade", material],
            ["History / Scope" if is_serial else "Batch / Lot", scope_or_batch], ["Evaluation date", evaluation_date.isoformat()],
            ["Test / source", source_note], ["Characteristic", characteristic], ["Unit", unit], ["N", result.n],
            ["Target", target], ["LSL", lsl], ["USL", usl], ["Mean", result.mean],
            ["Overall sigma (STDEV.S)", result.overall_std],
        ]
        if is_serial:
            metadata_rows += [["Ppk", result.ppk], ["Ppk requirement", requirement], ["Status", "PASS" if passed else "FAIL"]]
        else:
            metadata_rows += [["Within sigma (I-MR)", result.within_std], ["Cpk", result.cpk], ["Cpk requirement", requirement], ["Status", "PASS" if passed else "FAIL"]]
        summary_df = pd.DataFrame(metadata_rows, columns=["Metric", "Value"])
        out = BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            results_df.to_excel(writer, index=False, sheet_name="Measurements")
            if is_serial and history is not None and not history.empty:
                history.to_excel(writer, index=False, sheet_name="Batch History")
            wb = writer.book
            fmt = wb.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "white"})
            for ws in writer.sheets.values():
                ws.set_row(0, None, fmt)
                ws.freeze_panes(1, 0)
                ws.set_column(0, 0, 30)
                ws.set_column(1, 8, 20)
        return out.getvalue()

    pdf_metadata = {
        "Project": project,
        "Supplier": supplier,
        "Material / Grade": material,
        "Batch / Scope": scope_or_batch,
        "Evaluation date": evaluation_date.isoformat(),
        "Test / source": source_note,
    }
    pdf_report = build_capability_pdf(
        values=values,
        result=result,
        lsl=float(lsl),
        usl=float(usl),
        target=float(target),
        requirement=float(requirement),
        characteristic=characteristic,
        unit=unit,
        study_type=study_type,
        is_serial=is_serial,
        passed=passed,
        metadata=pdf_metadata,
        history=history,
    )

    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", characteristic.lower()).strip("_") or "capability"
    slug = "serial_ppk" if is_serial else "ppap_cpk"
    e = st.columns(3)
    e[0].download_button("Download Excel report (.xlsx)", data=build_report(), file_name=f"{safe}_{slug}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    e[1].download_button("Download professional report (.pdf)", data=pdf_report, file_name=f"{safe}_{slug}_report.pdf", mime="application/pdf", use_container_width=True, type="primary")
    e[2].download_button("Download cleaned data (.csv)", data=results_df.to_csv(index=False).encode("utf-8"), file_name=f"{safe}_{slug}_measurements.csv", mime="text/csv", use_container_width=True)
    st.caption(f"PDF is the presentation-ready engineering report; Excel contains the detailed calculation data. Acceptance: {primary_name} >= {requirement:.2f}.")

st.divider()
if is_serial:
    st.caption("Serial note: cumulative Ppk treats all selected measurements as one overall dataset. Do not combine data across a specification, material or process change that should be evaluated separately.")
else:
    st.caption("PPAP note: Cpk uses the moving range of sequential individual measurements to estimate within-process variation, so measurement order matters.")
