from __future__ import annotations

from io import BytesIO
import json
import math
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ui import hero


hero(
    "LDS Friction Lab",
    "Supplier-development workspace for friction batches, process traceability, physical properties, dyno results, comparisons, gaps and actions.",
)

st.info(
    "LDS data is kept in the current Streamlit session only. Export a workspace backup before closing the session. "
    "This avoids writing supplier/project data into the public GitHub repository."
)

BATCH_COLUMNS = [
    "Batch / Lot", "Supplier", "Material / Grade", "Underlayer", "Sample level",
    "Manufacturing stage", "Pad drawing", "Backplate drawing", "Production date",
    "Process status", "4M change", "Process deviation", "Reference batch", "Notes",
]
PHYSICAL_COLUMNS = [
    "Batch / Lot", "Test", "Result", "Unit", "LSL", "USL", "Target", "Sample count",
    "Status", "Test date", "Source / Report", "Notes",
]
DYNO_COLUMNS = [
    "Batch / Lot", "Test", "Run", "Metric", "Result", "Unit", "Min", "Max",
    "Status", "Disc / Rotor", "Test date", "Source / Report", "Notes",
]
ACTION_COLUMNS = [
    "Area", "Batch / Lot", "Action / Gap", "Owner", "Due date", "Priority", "Status", "Notes",
]

PHYSICAL_TESTS = [
    "Cold Compressibility (K3)", "Hot Compressibility", "Heat Transfer", "Shear Strength",
    "Peel Resistance", "Underlayer Distribution", "Natural Frequency", "Density",
    "Pad Thickness", "Backplate Thickness", "T-Pull", "Corrosion", "XRF", "pH",
    "TGA", "Moisture", "Sticking / Adhesion", "Other",
]
DYNO_TESTS = [
    "AK-Master", "AMS / HAMS", "Static μ", "Wear", "Mojacar", "EPB Holding Torque",
    "Rust Removal", "WLTP / Brake Emission", "High-Temperature Outgassing", "Krauss",
    "ECE R13", "Salt / Water Spray", "Stiction Corrosion", "DTV / BTV", "Flaming", "MPU", "Other",
]


def empty_df(columns):
    return pd.DataFrame(columns=columns)


def ensure_workspace():
    defaults = {
        "lds_batches": empty_df(BATCH_COLUMNS),
        "lds_physicals": empty_df(PHYSICAL_COLUMNS),
        "lds_dyno": empty_df(DYNO_COLUMNS),
        "lds_actions": empty_df(ACTION_COLUMNS),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy()


def clean_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df is None:
        return empty_df(columns)
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for col in columns:
        if col not in out.columns:
            out[col] = None
    out = out[columns]
    out = out.replace({np.nan: None})
    return out.reset_index(drop=True)


def to_num(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            v = v.replace(",", ".")
            match = re.search(r"[-+]?\d+(?:\.\d+)?", v)
            if not match:
                return None
            return float(match.group())
        return float(value)
    except Exception:
        return None


def auto_status(result, lsl=None, usl=None):
    if result is None or str(result).strip() == "":
        return "OPEN"
    r = to_num(result)
    lo = to_num(lsl)
    hi = to_num(usl)
    if r is not None:
        if lo is not None and r < lo:
            return "FAIL"
        if hi is not None and r > hi:
            return "FAIL"
        if lo is not None or hi is not None:
            return "PASS"
        return "RESULT"
    txt = str(result).strip().lower()
    if any(x in txt for x in ["nok", "fail", "not ok", "outside"]):
        return "FAIL"
    if any(x in txt for x in ["ok", "pass", "within"]):
        return "PASS"
    return "RESULT"


def recalc_status(df: pd.DataFrame, lower_col: str, upper_col: str):
    out = df.copy()
    if "Status" not in out:
        out["Status"] = None
    statuses = []
    for _, row in out.iterrows():
        current = str(row.get("Status") or "").strip().upper()
        if current in {"OPEN", "PASS", "FAIL", "RESULT", "N.A.", "NA", "CARRYOVER"}:
            if current in {"PASS", "FAIL"} and row.get("Result") not in [None, ""]:
                statuses.append(auto_status(row.get("Result"), row.get(lower_col), row.get(upper_col)))
            else:
                statuses.append(current)
        else:
            statuses.append(auto_status(row.get("Result"), row.get(lower_col), row.get(upper_col)))
    out["Status"] = statuses
    return out


def classify_sheet(name: str, df: pd.DataFrame) -> str:
    n = name.lower()
    cols = " ".join(str(c).lower() for c in df.columns)
    if "action" in n or "owner" in cols or "due date" in cols:
        return "Actions"
    if "dyno" in n or any(k in cols for k in ["rotor", "disc", "run", "ak-master", "ak master"]):
        return "Dyno"
    if "phys" in n or any(k in cols for k in ["lsl", "usl", "sample count", "heat transfer", "compressibility"]):
        return "Physicals"
    if "batch" in n or any(k in cols for k in ["supplier", "material", "underlayer", "4m"]):
        return "Batches"
    return "Unmapped"


def map_columns(df: pd.DataFrame, target_columns: list[str], prefix: str):
    st.caption("Map source columns to LDS fields. Unmapped fields stay blank.")
    source_cols = [None] + list(df.columns)
    aliases = {
        "Batch / Lot": ["batch", "lot", "batch no", "batch number"],
        "Supplier": ["supplier"],
        "Material / Grade": ["material", "grade", "friction material"],
        "Underlayer": ["underlayer", "ul"],
        "Sample level": ["sample", "level"],
        "Test": ["test", "characteristic", "parameter"],
        "Metric": ["metric", "parameter"],
        "Result": ["result", "value", "measured"],
        "Unit": ["unit"],
        "LSL": ["lsl", "min", "lower"],
        "USL": ["usl", "max", "upper"],
        "Min": ["min", "lower"],
        "Max": ["max", "upper"],
        "Target": ["target", "nominal"],
        "Sample count": ["sample count", "n", "qty", "quantity"],
        "Run": ["run", "repeat"],
        "Source / Report": ["source", "report", "reference", "ref"],
        "Disc / Rotor": ["disc", "rotor"],
        "Owner": ["owner", "responsible"],
        "Due date": ["due", "deadline"],
        "Action / Gap": ["action", "gap", "open point"],
        "Priority": ["priority"],
        "Status": ["status"],
        "Notes": ["note", "comment", "remark"],
    }
    lower_lookup = {str(c).strip().lower(): c for c in df.columns}
    defaults = {}
    for tgt in target_columns:
        for a in aliases.get(tgt, []):
            match = next((orig for low, orig in lower_lookup.items() if a == low or a in low), None)
            if match is not None:
                defaults[tgt] = match
                break
    mapped = {}
    cols = st.columns(3)
    for i, tgt in enumerate(target_columns):
        default_value = defaults.get(tgt)
        idx = source_cols.index(default_value) if default_value in source_cols else 0
        mapped[tgt] = cols[i % 3].selectbox(tgt, source_cols, index=idx, key=f"{prefix}_{i}_{tgt}")
    out = pd.DataFrame(index=df.index)
    for tgt in target_columns:
        src = mapped[tgt]
        out[tgt] = df[src] if src is not None else None
    return clean_frame(out, target_columns)


def workspace_excel_bytes():
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        st.session_state.lds_batches.to_excel(writer, index=False, sheet_name="Batches")
        st.session_state.lds_physicals.to_excel(writer, index=False, sheet_name="Physicals")
        st.session_state.lds_dyno.to_excel(writer, index=False, sheet_name="Dyno")
        st.session_state.lds_actions.to_excel(writer, index=False, sheet_name="Actions")
        wb = writer.book
        header = wb.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "white"})
        for ws in writer.sheets.values():
            ws.set_row(0, None, header)
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(ws.dim_colmax or 1, 1), 20)
    return out.getvalue()


def demo_workspace():
    st.session_state.lds_batches = pd.DataFrame([
        ["TW0676", "TMD", "PA4830", "7CM", "Preselection", "Prototype", "0204N60880", "0204N60891", "", "Development", "No", "", "", "Reference preselection batch"],
        ["TW0677", "TMD", "PA4830", "7CM", "Preselection", "Prototype", "0204N60880", "0204N60891", "", "Development", "No", "", "TW0676", "Second preselection batch"],
    ], columns=BATCH_COLUMNS)
    st.session_state.lds_physicals = pd.DataFrame([
        ["TW0676", "Cold Compressibility (K3)", 101, "µm", 75, 125, 100, 16, "PASS", "", "Demo", ""],
        ["TW0676", "Heat Transfer", 198, "°C", None, 200, None, 4, "PASS", "", "Demo", ""],
        ["TW0676", "Peel Resistance", None, "N/mm", None, None, None, None, "OPEN", "", "", "Not yet tested"],
        ["TW0677", "Cold Compressibility (K3)", 104, "µm", 75, 125, 100, 16, "PASS", "", "Demo", ""],
        ["TW0677", "Natural Frequency", None, "Hz", None, None, None, None, "OPEN", "", "", "Missing evidence"],
    ], columns=PHYSICAL_COLUMNS)
    st.session_state.lds_dyno = pd.DataFrame([
        ["TW0676", "AK-Master", 1, "Minimum µ", 0.34, "", 0.33, None, "PASS", "", "", "Demo", ""],
        ["TW0676", "Wear", 1, "Pad wear", 1.2, "mm", None, 1.5, "PASS", "", "", "Demo", ""],
        ["TW0677", "AK-Master", 1, "Minimum µ", 0.32, "", 0.33, None, "FAIL", "", "", "Demo", "Below minimum criterion"],
        ["TW0677", "EPB Holding Torque", 1, "Holding torque", None, "Nm", None, None, "OPEN", "", "", "", "Result pending"],
        ["TW0677", "WLTP / Brake Emission", 1, "PM", None, "mg/km", None, None, "OPEN", "", "", "", "Not yet tested"],
    ], columns=DYNO_COLUMNS)
    st.session_state.lds_actions = pd.DataFrame([
        ["Physicals", "TW0676", "Complete peel resistance", "", "", "Medium", "Open", ""],
        ["Physicals", "TW0677", "Provide natural-frequency evidence", "", "", "Medium", "Open", ""],
        ["Dyno", "TW0677", "Review AK-Master minimum µ result", "", "", "High", "Open", ""],
        ["Dyno", "TW0677", "Complete EPB holding torque", "", "", "High", "Open", ""],
    ], columns=ACTION_COLUMNS)


ensure_workspace()

with st.sidebar:
    st.divider()
    st.subheader("LDS workspace")
    if st.button("Load demo workspace", use_container_width=True):
        demo_workspace()
        st.rerun()
    if st.button("Clear workspace", use_container_width=True):
        st.session_state.lds_batches = empty_df(BATCH_COLUMNS)
        st.session_state.lds_physicals = empty_df(PHYSICAL_COLUMNS)
        st.session_state.lds_dyno = empty_df(DYNO_COLUMNS)
        st.session_state.lds_actions = empty_df(ACTION_COLUMNS)
        st.rerun()
    st.download_button(
        "Export LDS backup (.xlsx)",
        workspace_excel_bytes(),
        "LDS_Friction_Lab_Backup.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

batches = st.session_state.lds_batches
phys = recalc_status(st.session_state.lds_physicals, "LSL", "USL")
dyno = recalc_status(st.session_state.lds_dyno, "Min", "Max")
actions = st.session_state.lds_actions
st.session_state.lds_physicals = phys
st.session_state.lds_dyno = dyno

open_phys = int((phys["Status"].astype(str).str.upper() == "OPEN").sum()) if len(phys) else 0
fail_phys = int((phys["Status"].astype(str).str.upper() == "FAIL").sum()) if len(phys) else 0
open_dyno = int((dyno["Status"].astype(str).str.upper() == "OPEN").sum()) if len(dyno) else 0
fail_dyno = int((dyno["Status"].astype(str).str.upper() == "FAIL").sum()) if len(dyno) else 0
open_actions = int(actions["Status"].astype(str).str.lower().isin(["open", "in progress", "ongoing"]).sum()) if len(actions) else 0

m = st.columns(5)
m[0].metric("Batches", len(batches))
m[1].metric("Physical gaps", open_phys, delta=f"{fail_phys} fail" if fail_phys else None, delta_color="inverse")
m[2].metric("Dyno gaps", open_dyno, delta=f"{fail_dyno} fail" if fail_dyno else None, delta_color="inverse")
m[3].metric("Open actions", open_actions)
m[4].metric("Evidence records", len(phys) + len(dyno))

tabs = st.tabs(["Overview", "Batches", "Physicals", "Dyno", "Compare", "Actions", "Import / Export"])

with tabs[0]:
    st.markdown("### Development status")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        status_rows = []
        for label, df in [("Physicals", phys), ("Dyno", dyno)]:
            if len(df):
                counts = df["Status"].fillna("OPEN").astype(str).str.upper().value_counts()
                for status, count in counts.items():
                    status_rows.append({"Area": label, "Status": status, "Count": int(count)})
        if status_rows:
            fig = px.bar(pd.DataFrame(status_rows), x="Area", y="Count", color="Status", barmode="stack", title="Evidence status")
            fig.update_layout(template="plotly_white", legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No LDS records yet. Import a workbook, enter data, or load the demo workspace.")
    with c2:
        st.markdown("#### Engineering attention")
        issues = []
        if fail_phys:
            issues.append(f"**{fail_phys} physical result(s)** outside numeric criteria.")
        if fail_dyno:
            issues.append(f"**{fail_dyno} dyno result(s)** outside numeric criteria.")
        if open_phys:
            issues.append(f"**{open_phys} physical test(s)** still open.")
        if open_dyno:
            issues.append(f"**{open_dyno} dyno result(s)** still open.")
        missing_sources = 0
        for df in [phys, dyno]:
            if len(df):
                missing_sources += int(
                    (df["Result"].notna())
                    & (df["Result"].astype(str).str.strip() != "")
                    & (df["Source / Report"].fillna("").astype(str).str.strip() == "")
                ).sum()
        if missing_sources:
            issues.append(f"**{missing_sources} completed result(s)** have no source/report reference.")
        if len(batches):
            changes = int(batches["4M change"].fillna("").astype(str).str.lower().isin(["yes", "y", "true", "changed"]).sum())
            if changes:
                issues.append(f"**{changes} batch(es)** indicate a 4M/process change.")
        if not issues:
            st.success("No open rule-based issues detected in the current workspace.")
        else:
            for issue in issues:
                st.markdown(f"- {issue}")

    if len(batches):
        st.markdown("### Batch passports")
        cols = ["Batch / Lot", "Supplier", "Material / Grade", "Underlayer", "Sample level", "Manufacturing stage", "Process status", "4M change"]
        st.dataframe(batches[cols], use_container_width=True, hide_index=True)

with tabs[1]:
    st.markdown("### Batch passport & process traceability")
    st.caption("Capture the batch identity and process sameness before interpreting physical or dyno results.")
    edited = st.data_editor(
        batches, num_rows="dynamic", use_container_width=True, hide_index=True, key="lds_batch_editor",
        column_config={
            "Production date": st.column_config.TextColumn(help="Date or supplier production reference"),
            "4M change": st.column_config.SelectboxColumn(options=["No", "Yes", "Unknown"], default="Unknown"),
            "Process status": st.column_config.SelectboxColumn(options=["Development", "Frozen process", "Serial intent", "Released", "Unknown"], default="Development"),
        },
    )
    st.session_state.lds_batches = clean_frame(edited, BATCH_COLUMNS)

with tabs[2]:
    st.markdown("### Physical properties")
    st.caption("Track each supplier batch against limits and keep the source/report reference with the result.")
    f1, f2, f3 = st.columns([1, 1, 2])
    batch_options = ["All"] + sorted({str(x) for x in phys["Batch / Lot"].dropna() if str(x).strip()})
    sel_batch = f1.selectbox("Batch filter", batch_options, key="phys_batch_filter")
    sel_status = f2.selectbox("Status filter", ["All", "OPEN", "PASS", "FAIL", "RESULT", "N.A.", "CARRYOVER"], key="phys_status_filter")
    search = f3.text_input("Search test / notes", key="phys_search")
    shown = phys.copy()
    if sel_batch != "All":
        shown = shown[shown["Batch / Lot"].astype(str) == sel_batch]
    if sel_status != "All":
        shown = shown[shown["Status"].astype(str).str.upper() == sel_status]
    if search.strip():
        s = search.lower()
        mask = shown.astype(str).apply(lambda col: col.str.lower().str.contains(re.escape(s), na=False)).any(axis=1)
        shown = shown[mask]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    with st.expander("Edit full physicals table", expanded=not len(phys)):
        edited = st.data_editor(
            phys, num_rows="dynamic", use_container_width=True, hide_index=True, key="lds_physical_editor",
            column_config={
                "Test": st.column_config.SelectboxColumn(options=PHYSICAL_TESTS),
                "Status": st.column_config.SelectboxColumn(options=["OPEN", "PASS", "FAIL", "RESULT", "N.A.", "CARRYOVER"]),
            },
        )
        st.session_state.lds_physicals = recalc_status(clean_frame(edited, PHYSICAL_COLUMNS), "LSL", "USL")

with tabs[3]:
    st.markdown("### Dyno results & repeats")
    st.caption("Keep repeats, criteria and source references separate so development decisions remain traceable.")
    f1, f2, f3 = st.columns([1, 1, 2])
    batch_options = ["All"] + sorted({str(x) for x in dyno["Batch / Lot"].dropna() if str(x).strip()})
    sel_batch = f1.selectbox("Batch filter", batch_options, key="dyno_batch_filter")
    sel_status = f2.selectbox("Status filter", ["All", "OPEN", "PASS", "FAIL", "RESULT", "N.A.", "CARRYOVER"], key="dyno_status_filter")
    search = f3.text_input("Search test / metric / notes", key="dyno_search")
    shown = dyno.copy()
    if sel_batch != "All":
        shown = shown[shown["Batch / Lot"].astype(str) == sel_batch]
    if sel_status != "All":
        shown = shown[shown["Status"].astype(str).str.upper() == sel_status]
    if search.strip():
        s = search.lower()
        mask = shown.astype(str).apply(lambda col: col.str.lower().str.contains(re.escape(s), na=False)).any(axis=1)
        shown = shown[mask]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    with st.expander("Edit full dyno table", expanded=not len(dyno)):
        edited = st.data_editor(
            dyno, num_rows="dynamic", use_container_width=True, hide_index=True, key="lds_dyno_editor",
            column_config={
                "Test": st.column_config.SelectboxColumn(options=DYNO_TESTS),
                "Status": st.column_config.SelectboxColumn(options=["OPEN", "PASS", "FAIL", "RESULT", "N.A.", "CARRYOVER"]),
            },
        )
        st.session_state.lds_dyno = recalc_status(clean_frame(edited, DYNO_COLUMNS), "Min", "Max")

with tabs[4]:
    st.markdown("### Batch comparison & history")
    all_batches = sorted(set(
        [str(x) for x in batches["Batch / Lot"].dropna()] +
        [str(x) for x in phys["Batch / Lot"].dropna()] +
        [str(x) for x in dyno["Batch / Lot"].dropna()]
    ))
    selected = st.multiselect("Batches to compare", all_batches, default=all_batches[: min(3, len(all_batches))])
    if not selected:
        st.info("Select at least one batch.")
    else:
        p = phys[phys["Batch / Lot"].astype(str).isin(selected)].copy()
        d = dyno[dyno["Batch / Lot"].astype(str).isin(selected)].copy()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Physicals")
            if len(p):
                st.dataframe(p, use_container_width=True, hide_index=True)
                p["Numeric result"] = p["Result"].map(to_num)
                plot = p.dropna(subset=["Numeric result"])
                if len(plot):
                    fig = px.scatter(plot, x="Test", y="Numeric result", color="Batch / Lot", symbol="Status", title="Physical-result comparison")
                    fig.update_layout(template="plotly_white", xaxis_tickangle=-35)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No physical records for selected batches.")
        with c2:
            st.markdown("#### Dyno")
            if len(d):
                st.dataframe(d, use_container_width=True, hide_index=True)
                d["Numeric result"] = d["Result"].map(to_num)
                plot = d.dropna(subset=["Numeric result"]).copy()
                if len(plot):
                    plot["Label"] = plot["Test"].astype(str) + " · " + plot["Metric"].fillna("").astype(str)
                    fig = px.scatter(plot, x="Label", y="Numeric result", color="Batch / Lot", symbol="Status", title="Dyno-result comparison")
                    fig.update_layout(template="plotly_white", xaxis_tickangle=-35)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No dyno records for selected batches.")

with tabs[5]:
    st.markdown("### Action tracker")
    st.caption("Convert missing evidence, failed criteria and supplier follow-ups into traceable actions.")
    edited = st.data_editor(
        actions, num_rows="dynamic", use_container_width=True, hide_index=True, key="lds_action_editor",
        column_config={
            "Priority": st.column_config.SelectboxColumn(options=["High", "Medium", "Low"]),
            "Status": st.column_config.SelectboxColumn(options=["Open", "In progress", "Blocked", "Closed"]),
        },
    )
    st.session_state.lds_actions = clean_frame(edited, ACTION_COLUMNS)

with tabs[6]:
    st.markdown("### Import existing LDS / restore backup")
    uploaded = st.file_uploader("Upload LDS Excel workbook", type=["xlsx", "xls"], key="lds_import_file")
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            xls = pd.ExcelFile(BytesIO(raw))
            st.success(f"Workbook opened: {len(xls.sheet_names)} sheet(s). Blank Excel cells remain blank.")
            overview = []
            loaded = {}
            for sheet in xls.sheet_names:
                df = pd.read_excel(BytesIO(raw), sheet_name=sheet, dtype=object)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                loaded[sheet] = df
                overview.append({"Sheet": sheet, "Rows": len(df), "Columns": len(df.columns), "Detected as": classify_sheet(sheet, df)})
            st.dataframe(pd.DataFrame(overview), use_container_width=True, hide_index=True)

            sheet = st.selectbox("Sheet to import", xls.sheet_names)
            source = loaded[sheet]
            st.dataframe(source.head(100), use_container_width=True, hide_index=True)
            detected = classify_sheet(sheet, source)
            choices = ["Batches", "Physicals", "Dyno", "Actions"]
            target = st.selectbox("Import this sheet as", choices, index=choices.index(detected) if detected in choices else 0)

            target_columns = {
                "Batches": BATCH_COLUMNS, "Physicals": PHYSICAL_COLUMNS, "Dyno": DYNO_COLUMNS, "Actions": ACTION_COLUMNS,
            }[target]
            mapped = map_columns(source, target_columns, f"map_{sheet}_{target}")
            st.markdown("#### Review before import")
            st.dataframe(mapped.head(100), use_container_width=True, hide_index=True)

            mode = st.radio("Import mode", ["Append", "Replace this LDS table"], horizontal=True)
            if st.button(f"Import {target}", type="primary"):
                key = {"Batches": "lds_batches", "Physicals": "lds_physicals", "Dyno": "lds_dyno", "Actions": "lds_actions"}[target]
                if mode == "Append":
                    st.session_state[key] = pd.concat([st.session_state[key], mapped], ignore_index=True)
                else:
                    st.session_state[key] = mapped.reset_index(drop=True)
                if target == "Physicals":
                    st.session_state[key] = recalc_status(st.session_state[key], "LSL", "USL")
                if target == "Dyno":
                    st.session_state[key] = recalc_status(st.session_state[key], "Min", "Max")
                st.success(f"{len(mapped)} row(s) imported into {target}.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not import workbook: {exc}")

    st.divider()
    st.markdown("### Export")
    st.download_button(
        "Download complete LDS workspace (.xlsx)", workspace_excel_bytes(), "LDS_Friction_Lab_Backup.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, type="primary",
    )
    export_json = {
        "batches": st.session_state.lds_batches.to_dict(orient="records"),
        "physicals": st.session_state.lds_physicals.to_dict(orient="records"),
        "dyno": st.session_state.lds_dyno.to_dict(orient="records"),
        "actions": st.session_state.lds_actions.to_dict(orient="records"),
    }
    st.download_button(
        "Download JSON backup", json.dumps(export_json, default=str, indent=2).encode("utf-8"),
        "LDS_Friction_Lab_Backup.json", "application/json", use_container_width=True,
    )

st.divider()
st.caption(
    "LDS Friction Lab is intended for development traceability and supplier review. "
    "Automatic PASS/FAIL applies only when a numeric result and numeric limit are available; engineering judgement remains necessary for qualitative evidence and carry-over decisions."
)
