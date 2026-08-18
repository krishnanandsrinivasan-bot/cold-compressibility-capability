from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm, shapiro, skew, kurtosis
import plotly.graph_objects as go


D2_FOR_MR2 = 1.128


@dataclass
class CapabilityResult:
    n: int
    mean: float
    overall_std: float
    within_std: float
    cp: float
    cpl: float
    cpu: float
    cpk: float
    pp: float
    ppl: float
    ppu: float
    ppk: float
    minimum: float
    maximum: float
    outside_count: int
    below_lsl: int
    above_usl: int
    outside_percent: float
    predicted_ppm: float
    shapiro_p: float | None
    skewness: float | None
    excess_kurtosis: float | None
    method: str

    def to_dict(self):
        return asdict(self)


def clean_numeric(values: Iterable) -> tuple[pd.Series, int]:
    """Convert values to finite floats. Returns cleaned data and count removed."""
    raw = pd.Series(list(values), dtype="object")
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    removed = int(numeric.isna().sum())
    return numeric.dropna().astype(float).reset_index(drop=True), removed


def parse_pasted_values(text: str) -> tuple[pd.Series, list[str]]:
    """Parse copied values while supporting decimal comma and common separators."""
    if not text or not text.strip():
        return pd.Series(dtype=float), []

    tokens: list[str] = []
    ignored: list[str] = []
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]

    # Excel column copy/paste: one value per line, including German decimal comma.
    if len(lines) > 1:
        for line in lines:
            # If a whole row with tabs or semicolons was pasted, accept every cell.
            parts = re.split(r"[\t;]+", line) if ("\t" in line or ";" in line) else [line]
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # A single comma inside one token is treated as decimal comma.
                if re.fullmatch(r"[+-]?\d+\s*,\s*\d+", part):
                    tokens.append(part.replace(" ", "").replace(",", "."))
                else:
                    tokens.extend([x for x in re.split(r"[ ,]+", part) if x])
    else:
        line = lines[0]
        if "\t" in line or ";" in line:
            parts = [x.strip() for x in re.split(r"[\t;]+", line) if x.strip()]
        elif line.count(",") == 1 and re.fullmatch(r"[+-]?\d+\s*,\s*\d+", line):
            parts = [line.replace(" ", "").replace(",", ".")]
        else:
            # One-line lists: commas/spaces are interpreted as separators.
            parts = [x for x in re.split(r"[,\s]+", line) if x]
        tokens.extend(parts)

    parsed: list[float] = []
    for token in tokens:
        token = token.strip().replace("µm", "").replace("um", "")
        # Allow decimal comma when it survived tokenization.
        if token.count(",") == 1 and "." not in token:
            token = token.replace(",", ".")
        try:
            value = float(token)
            if math.isfinite(value):
                parsed.append(value)
            else:
                ignored.append(token)
        except ValueError:
            ignored.append(token)

    return pd.Series(parsed, dtype=float), ignored


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def calculate_capability(values: Iterable, lsl: float, usl: float, method: str = "Direct STDEV.S") -> CapabilityResult:
    data, _ = clean_numeric(values)
    if len(data) < 2:
        raise ValueError("At least two valid measurements are required.")
    if not (math.isfinite(lsl) and math.isfinite(usl)) or lsl >= usl:
        raise ValueError("USL must be greater than LSL.")

    arr = data.to_numpy(dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    overall_std = float(np.std(arr, ddof=1))

    if method == "I-MR (within sigma = MR̄ / 1.128)":
        if n < 3:
            raise ValueError("I-MR mode needs at least three sequential measurements.")
        mr_bar = float(np.mean(np.abs(np.diff(arr))))
        within_std = mr_bar / D2_FOR_MR2
    else:
        within_std = overall_std
        method = "Direct STDEV.S"

    cpl = _safe_ratio(mean - lsl, 3 * within_std)
    cpu = _safe_ratio(usl - mean, 3 * within_std)
    cpk = min(cpl, cpu) if all(math.isfinite(x) for x in (cpl, cpu)) else float("nan")
    cp = _safe_ratio(usl - lsl, 6 * within_std)

    ppl = _safe_ratio(mean - lsl, 3 * overall_std)
    ppu = _safe_ratio(usl - mean, 3 * overall_std)
    ppk = min(ppl, ppu) if all(math.isfinite(x) for x in (ppl, ppu)) else float("nan")
    pp = _safe_ratio(usl - lsl, 6 * overall_std)

    below = int(np.sum(arr < lsl))
    above = int(np.sum(arr > usl))
    outside = below + above
    outside_percent = 100.0 * outside / n

    if overall_std > 0 and math.isfinite(overall_std):
        p_below = norm.cdf(lsl, loc=mean, scale=overall_std)
        p_above = 1 - norm.cdf(usl, loc=mean, scale=overall_std)
        predicted_ppm = float((p_below + p_above) * 1_000_000)
    else:
        predicted_ppm = 0.0 if lsl <= mean <= usl else 1_000_000.0

    shapiro_p = None
    if 3 <= n <= 5000:
        try:
            shapiro_p = float(shapiro(arr).pvalue)
        except Exception:
            shapiro_p = None

    try:
        skewness = float(skew(arr, bias=False)) if n >= 3 else None
    except Exception:
        skewness = None
    try:
        excess_kurtosis = float(kurtosis(arr, fisher=True, bias=False)) if n >= 4 else None
    except Exception:
        excess_kurtosis = None

    return CapabilityResult(
        n=n,
        mean=mean,
        overall_std=overall_std,
        within_std=within_std,
        cp=cp,
        cpl=cpl,
        cpu=cpu,
        cpk=cpk,
        pp=pp,
        ppl=ppl,
        ppu=ppu,
        ppk=ppk,
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        outside_count=outside,
        below_lsl=below,
        above_usl=above,
        outside_percent=outside_percent,
        predicted_ppm=predicted_ppm,
        shapiro_p=shapiro_p,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        method=method,
    )


def prepare_results_table(values: Iterable, lsl: float, usl: float, mean: float, std: float) -> pd.DataFrame:
    data, _ = clean_numeric(values)
    df = pd.DataFrame({"Pad / Measurement #": np.arange(1, len(data) + 1), "Compressibility (µm)": data})
    df["Status"] = np.select(
        [df["Compressibility (µm)"] < lsl, df["Compressibility (µm)"] > usl],
        ["Below LSL", "Above USL"],
        default="Within spec",
    )
    if std > 0 and math.isfinite(std):
        df["Z from mean"] = (df["Compressibility (µm)"] - mean) / std
    else:
        df["Z from mean"] = np.nan
    return df


def sequence_chart(values: Iterable, lsl: float, usl: float, target: float | None, mean: float) -> go.Figure:
    data, _ = clean_numeric(values)
    x = np.arange(1, len(data) + 1)
    arr = data.to_numpy()
    in_spec = (arr >= lsl) & (arr <= usl)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x[in_spec], y=arr[in_spec], mode="markers", name="Within spec",
        marker=dict(size=8, color="#1f77b4", line=dict(width=0.5, color="white")),
        hovertemplate="Pad %{x}<br>%{y:.2f} µm<extra></extra>",
    ))
    if np.any(~in_spec):
        fig.add_trace(go.Scatter(
            x=x[~in_spec], y=arr[~in_spec], mode="markers", name="Outside spec",
            marker=dict(size=10, color="#d62728", symbol="x"),
            hovertemplate="Pad %{x}<br>%{y:.2f} µm<extra></extra>",
        ))

    fig.add_hline(y=lsl, line_color="#d62728", line_dash="dash", annotation_text=f"LSL {lsl:g}")
    fig.add_hline(y=usl, line_color="#d62728", line_dash="dash", annotation_text=f"USL {usl:g}")
    fig.add_hline(y=mean, line_color="#2ca02c", line_dash="dot", annotation_text=f"Mean {mean:.2f}")
    if target is not None and math.isfinite(target):
        fig.add_hline(y=target, line_color="#7f7f7f", line_dash="dot", annotation_text=f"Target {target:g}")

    fig.update_layout(
        title="Measurement Sequence / Scatter",
        xaxis_title="Measurement number",
        yaxis_title="Cold compressibility (µm)",
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def distribution_chart(values: Iterable, lsl: float, usl: float, target: float | None, mean: float, std: float, title: str = "Distribution & Specification Window") -> go.Figure:
    data, _ = clean_numeric(values)
    arr = data.to_numpy()

    fig = go.Figure()
    bins = max(8, min(40, int(round(math.sqrt(len(arr))))))
    fig.add_trace(go.Histogram(
        x=arr, histnorm="probability density", nbinsx=bins,
        name="Observed data", opacity=0.55, marker_color="#8fb9df",
        hovertemplate="Compressibility %{x:.2f} µm<br>Density %{y:.4f}<extra></extra>",
    ))

    if std > 0 and math.isfinite(std):
        lo = min(float(np.min(arr)), lsl, mean - 4 * std)
        hi = max(float(np.max(arr)), usl, mean + 4 * std)
        xs = np.linspace(lo, hi, 500)
        ys = norm.pdf(xs, loc=mean, scale=std)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Fitted normal", line=dict(color="#1f4e79", width=3)))
        fig.add_vrect(x0=mean - 3 * std, x1=mean + 3 * std, fillcolor="#1f4e79", opacity=0.06, line_width=0, annotation_text="Fitted ±3σ", annotation_position="top left")

    fig.add_vrect(x0=lsl, x1=usl, fillcolor="#2ca02c", opacity=0.05, line_width=0)
    fig.add_vline(x=lsl, line_color="#d62728", line_dash="dash", annotation_text=f"LSL {lsl:g}")
    fig.add_vline(x=usl, line_color="#d62728", line_dash="dash", annotation_text=f"USL {usl:g}")
    fig.add_vline(x=mean, line_color="#2ca02c", line_dash="dot", annotation_text=f"Mean {mean:.2f}")
    if target is not None and math.isfinite(target):
        fig.add_vline(x=target, line_color="#7f7f7f", line_dash="dot", annotation_text=f"Target {target:g}")

    fig.update_layout(
        title=title,
        xaxis_title="Cold compressibility (µm)",
        yaxis_title="Probability density",
        template="plotly_white",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def scenario_chart(lsl: float, usl: float, target: float | None, mean: float, std: float) -> go.Figure:
    if std <= 0:
        std = 0.001
    lo = min(lsl, mean - 4 * std)
    hi = max(usl, mean + 4 * std)
    xs = np.linspace(lo, hi, 500)
    ys = norm.pdf(xs, loc=mean, scale=std)

    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy", name="What-if normal distribution", line=dict(width=3, color="#4c78a8")))
    fig.add_vrect(x0=lsl, x1=usl, fillcolor="#2ca02c", opacity=0.06, line_width=0)
    fig.add_vline(x=lsl, line_color="#d62728", line_dash="dash", annotation_text=f"LSL {lsl:g}")
    fig.add_vline(x=usl, line_color="#d62728", line_dash="dash", annotation_text=f"USL {usl:g}")
    fig.add_vline(x=mean, line_color="#2ca02c", line_dash="dot", annotation_text=f"Scenario mean {mean:.2f}")
    if target is not None and math.isfinite(target):
        fig.add_vline(x=target, line_color="#7f7f7f", line_dash="dot", annotation_text=f"Target {target:g}")
    fig.add_vline(x=mean - 3 * std, line_color="#4c78a8", line_dash="dot", annotation_text="−3σ")
    fig.add_vline(x=mean + 3 * std, line_color="#4c78a8", line_dash="dot", annotation_text="+3σ")
    fig.update_layout(
        title="Capability What-if: Move the Mean or Change the Variation",
        xaxis_title="Cold compressibility (µm)", yaxis_title="Probability density",
        template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def build_excel_report(values: Iterable, result: CapabilityResult, lsl: float, usl: float, target: float | None, cpk_req: float, ppk_req: float) -> bytes:
    data_table = prepare_results_table(values, lsl, usl, result.mean, result.overall_std)
    summary = pd.DataFrame([
        ["Method", result.method],
        ["N", result.n],
        ["Target (µm)", target],
        ["LSL (µm)", lsl],
        ["USL (µm)", usl],
        ["Mean (µm)", result.mean],
        ["Overall STDEV.S (µm)", result.overall_std],
        ["Within sigma (µm)", result.within_std],
        ["Cp", result.cp],
        ["Cpk", result.cpk],
        ["Cpk requirement", cpk_req],
        ["Cpk status", "PASS" if result.cpk >= cpk_req else "FAIL"],
        ["Pp", result.pp],
        ["Ppk", result.ppk],
        ["Ppk requirement", ppk_req],
        ["Ppk status", "PASS" if result.ppk >= ppk_req else "FAIL"],
        ["Outside spec", result.outside_count],
        ["Outside spec (%)", result.outside_percent],
        ["Predicted nonconforming (ppm, normal model)", result.predicted_ppm],
    ], columns=["Metric", "Value"])

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary", startrow=2)
        data_table.to_excel(writer, index=False, sheet_name="Measurements")
        workbook = writer.book
        ws = writer.sheets["Summary"]
        data_ws = writer.sheets["Measurements"]

        header_fmt = workbook.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "white", "border": 0})
        title_fmt = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#1F4E78"})
        num_fmt = workbook.add_format({"num_format": "0.000"})
        pass_fmt = workbook.add_format({"bg_color": "#E2F0D9", "font_color": "#375623"})
        fail_fmt = workbook.add_format({"bg_color": "#FCE4D6", "font_color": "#9C0006"})

        ws.write(0, 0, "Cold Compressibility Capability Report", title_fmt)
        ws.set_column("A:A", 42)
        ws.set_column("B:B", 22)
        ws.set_row(2, None, header_fmt)
        ws.conditional_format("B4:B40", {"type": "text", "criteria": "containing", "value": "PASS", "format": pass_fmt})
        ws.conditional_format("B4:B40", {"type": "text", "criteria": "containing", "value": "FAIL", "format": fail_fmt})

        data_ws.set_column("A:A", 20)
        data_ws.set_column("B:B", 24, num_fmt)
        data_ws.set_column("C:C", 18)
        data_ws.set_column("D:D", 14, num_fmt)
        data_ws.set_row(0, None, header_fmt)
        data_ws.freeze_panes(1, 0)
        data_ws.autofilter(0, 0, len(data_table), len(data_table.columns) - 1)

    return output.getvalue()
