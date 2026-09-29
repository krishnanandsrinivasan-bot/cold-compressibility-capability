from __future__ import annotations

from io import BytesIO
import math
from typing import Mapping, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)


NAVY = colors.HexColor("#183A5A")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GREY = colors.HexColor("#F3F5F7")
MID_GREY = colors.HexColor("#D7DCE1")
TEXT = colors.HexColor("#1F2933")
GREEN = colors.HexColor("#2E7D32")
LIGHT_GREEN = colors.HexColor("#E8F5E9")
RED = colors.HexColor("#B3261E")
LIGHT_RED = colors.HexColor("#FDECEC")
AMBER = colors.HexColor("#9A6700")


def _safe_text(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    return text.replace("≥", ">=").replace("σ", "sigma")


def _fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        if not math.isfinite(float(x)):
            return "-"
    except Exception:
        return _safe_text(x)
    return f"{float(x):.{digits}f}"


def _chart_image(fig, width_mm: float = 178) -> Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_mm * mm
    img.drawHeight = width_mm * mm * ratio
    return img


def _sequence_figure(values, lsl, usl, target, mean, characteristic, unit):
    arr = np.asarray(values, dtype=float)
    x = np.arange(1, len(arr) + 1)
    fig, ax = plt.subplots(figsize=(10.3, 3.8))
    inside = (arr >= lsl) & (arr <= usl)
    ax.scatter(x[inside], arr[inside], s=16, label="Within specification")
    if np.any(~inside):
        ax.scatter(x[~inside], arr[~inside], s=30, marker="x", label="Outside specification")
    ax.axhline(lsl, linestyle="--", linewidth=1.2, label=f"LSL {lsl:g}")
    ax.axhline(usl, linestyle="--", linewidth=1.2, label=f"USL {usl:g}")
    ax.axhline(mean, linestyle=":", linewidth=1.4, label=f"Mean {mean:.2f}")
    if target is not None and math.isfinite(float(target)):
        ax.axhline(target, linestyle="-.", linewidth=1.0, label=f"Target {target:g}")
    ax.set_title("Measurement Sequence / Scatter", loc="left", fontweight="bold")
    ax.set_xlabel("Measurement number")
    ax.set_ylabel(f"{characteristic}{f' ({unit})' if unit else ''}")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _distribution_figure(values, lsl, usl, target, mean, sigma, characteristic, unit, sigma_label):
    arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(10.3, 3.8))
    bins = max(8, min(35, int(round(math.sqrt(len(arr))))))
    ax.hist(arr, bins=bins, density=True, alpha=0.45, label="Observed measurements")
    if sigma > 0 and math.isfinite(float(sigma)):
        lo = min(float(np.min(arr)), lsl, mean - 4 * sigma)
        hi = max(float(np.max(arr)), usl, mean + 4 * sigma)
        xs = np.linspace(lo, hi, 500)
        ax.plot(xs, norm.pdf(xs, loc=mean, scale=sigma), linewidth=2.2, label=f"Normal model ({sigma_label})")
        ax.axvspan(mean - 3 * sigma, mean + 3 * sigma, alpha=0.08, label="+/-3 sigma width")
    ax.axvline(lsl, linestyle="--", linewidth=1.2, label=f"LSL {lsl:g}")
    ax.axvline(usl, linestyle="--", linewidth=1.2, label=f"USL {usl:g}")
    ax.axvline(mean, linestyle=":", linewidth=1.4, label=f"Mean {mean:.2f}")
    if target is not None and math.isfinite(float(target)):
        ax.axvline(target, linestyle="-.", linewidth=1.0, label=f"Target {target:g}")
    ax.set_title("Distribution and Specification Window", loc="left", fontweight="bold")
    ax.set_xlabel(f"{characteristic}{f' ({unit})' if unit else ''}")
    ax.set_ylabel("Probability density")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _history_figure(history: pd.DataFrame, requirement: float):
    fig, ax = plt.subplots(figsize=(10.3, 3.8))
    x = np.arange(len(history))
    y = history["Cumulative Ppk"].astype(float).to_numpy()
    labels = history["Batch / Lot"].astype(str).tolist()
    ax.plot(x, y, marker="o", linewidth=2.2, label="Cumulative Ppk")
    ax.axhline(requirement, linestyle="--", linewidth=1.2, label=f"Requirement {requirement:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Ppk")
    ax.set_title("Cumulative Ppk by Batch / Lot", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def build_capability_pdf(
    *,
    values,
    result,
    lsl: float,
    usl: float,
    target: float | None,
    requirement: float,
    characteristic: str,
    unit: str,
    study_type: str,
    is_serial: bool,
    passed: bool,
    metadata: Mapping[str, Any] | None = None,
    history: pd.DataFrame | None = None,
) -> bytes:
    data = np.asarray(list(values), dtype=float)
    primary_name = "Ppk" if is_serial else "Cpk"
    primary_value = result.ppk if is_serial else result.cpk
    relevant_sigma = result.overall_std if is_serial else result.within_std
    sigma_label = "overall STDEV.S" if is_serial else "within sigma (I-MR)"
    unit_suffix = f" {unit}" if unit else ""

    out = BytesIO()
    doc = SimpleDocTemplate(
        out,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"{characteristic} Process Capability Report",
        author="Pad Development Tools",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitlePDT", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, spaceAfter=4))
    styles.add(ParagraphStyle(name="SubPDT", parent=styles["Normal"], fontSize=9.5, leading=13, textColor=colors.HexColor("#5B6570"), spaceAfter=10))
    styles.add(ParagraphStyle(name="H2PDT", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=6, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyPDT", parent=styles["BodyText"], fontSize=9.2, leading=13, textColor=TEXT))
    styles.add(ParagraphStyle(name="SmallPDT", parent=styles["BodyText"], fontSize=7.8, leading=10, textColor=colors.HexColor("#5B6570")))
    styles.add(ParagraphStyle(name="RightSmall", parent=styles["SmallPDT"], alignment=TA_RIGHT))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(MID_GREY)
        canvas.line(15 * mm, 12 * mm, A4[0] - 15 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(15 * mm, 7.5 * mm, "Pad Development Tools | Process Capability Report")
        canvas.drawRightString(A4[0] - 15 * mm, 7.5 * mm, f"Page {document.page}")
        canvas.restoreState()

    story = []
    story.append(Paragraph("PAD DEVELOPMENT TOOLS", styles["SmallPDT"]))
    story.append(Paragraph("Process Capability Report", styles["TitlePDT"]))
    story.append(Paragraph(f"{_safe_text(characteristic)} | {_safe_text(study_type)}", styles["SubPDT"]))

    status_bg = LIGHT_GREEN if passed else LIGHT_RED
    status_fg = GREEN if passed else RED
    status = "PASS" if passed else "FAIL"
    status_data = [[
        Paragraph(f"<b>{status}</b>", ParagraphStyle("Status", fontName="Helvetica-Bold", fontSize=15, textColor=status_fg, alignment=TA_LEFT)),
        Paragraph(f"<b>{primary_name} {_fmt(primary_value)}</b><br/>Requirement: {primary_name} >= {requirement:.2f}", styles["BodyPDT"]),
        Paragraph(f"N = {result.n:,}<br/>Outside spec = {result.outside_count} ({result.outside_percent:.2f}%)", styles["BodyPDT"]),
    ]]
    status_table = Table(status_data, colWidths=[32 * mm, 75 * mm, 63 * mm], rowHeights=[18 * mm])
    status_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), status_bg),
        ("BOX", (0, 0), (-1, -1), 0.7, status_fg),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(status_table)
    story.append(Spacer(1, 6 * mm))

    meta = metadata or {}
    meta_rows = [
        ["Project", _safe_text(meta.get("Project")), "Supplier", _safe_text(meta.get("Supplier"))],
        ["Material / Grade", _safe_text(meta.get("Material / Grade")), "Batch / Scope", _safe_text(meta.get("Batch / Scope"))],
        ["Evaluation date", _safe_text(meta.get("Evaluation date")), "Test / source", _safe_text(meta.get("Test / source"))],
    ]
    meta_table = Table(meta_rows, colWidths=[29 * mm, 56 * mm, 29 * mm, 56 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GREY),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT_GREY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Capability Summary", styles["H2PDT"]))
    if is_serial:
        metrics = [
            ["Target", f"{_fmt(target, 2)}{unit_suffix}", "LSL", f"{lsl:.2f}{unit_suffix}", "USL", f"{usl:.2f}{unit_suffix}"],
            ["Mean", f"{result.mean:.2f}{unit_suffix}", "Overall sigma", f"{result.overall_std:.3f}{unit_suffix}", "Ppk", _fmt(result.ppk)],
            ["Pp", _fmt(result.pp), "PPL", _fmt(result.ppl), "PPU", _fmt(result.ppu)],
            ["Minimum", f"{result.minimum:.2f}{unit_suffix}", "Maximum", f"{result.maximum:.2f}{unit_suffix}", "Predicted outside", f"{result.predicted_ppm:,.0f} ppm"],
        ]
    else:
        metrics = [
            ["Target", f"{_fmt(target, 2)}{unit_suffix}", "LSL", f"{lsl:.2f}{unit_suffix}", "USL", f"{usl:.2f}{unit_suffix}"],
            ["Mean", f"{result.mean:.2f}{unit_suffix}", "Within sigma", f"{result.within_std:.3f}{unit_suffix}", "Cpk", _fmt(result.cpk)],
            ["Cp", _fmt(result.cp), "CPL", _fmt(result.cpl), "CPU", _fmt(result.cpu)],
            ["Overall STDEV.S", f"{result.overall_std:.3f}{unit_suffix}", "Minimum", f"{result.minimum:.2f}{unit_suffix}", "Maximum", f"{result.maximum:.2f}{unit_suffix}"],
        ]
    metric_table = Table(metrics, colWidths=[27 * mm, 30 * mm] * 3)
    metric_style = [
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for col in [0, 2, 4]:
        metric_style += [("BACKGROUND", (col, 0), (col, -1), LIGHT_BLUE), ("FONTNAME", (col, 0), (col, -1), "Helvetica-Bold")]
    metric_table.setStyle(TableStyle(metric_style))
    story.append(metric_table)
    story.append(Spacer(1, 4 * mm))

    if passed:
        conclusion = f"The evaluated process {'history' if is_serial else 'batch'} meets the defined {primary_name} requirement."
    else:
        conclusion = f"The evaluated process {'history' if is_serial else 'batch'} does not meet the defined {primary_name} requirement."
    if result.outside_count == 0 and not passed:
        conclusion += " All measured parts are within specification, but the process distribution is not sufficiently capable against the acceptance criterion."
    center_offset = result.mean - (float(target) if target is not None else (lsl + usl) / 2)
    conclusion += f" Mean offset from target is {center_offset:+.2f}{unit_suffix}."
    story.append(Table([[Paragraph(f"<b>Engineering interpretation:</b> {conclusion}", styles["BodyPDT"])]], colWidths=[171 * mm], style=[
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7E0") if not passed else colors.HexColor("#EEF7EE")),
        ("BOX", (0, 0), (-1, -1), 0.5, AMBER if not passed else GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Measurement Sequence", styles["H2PDT"]))
    story.append(_chart_image(_sequence_figure(data, lsl, usl, target, result.mean, characteristic, unit)))
    story.append(PageBreak())

    story.append(Paragraph("Distribution and Capability", styles["TitlePDT"]))
    story.append(Paragraph(
        f"Normal model uses {_safe_text(sigma_label)} = {_fmt(relevant_sigma)}{unit_suffix}. "
        f"The shaded +/-3 sigma width is compared with the specification window {lsl:g} to {usl:g}{unit_suffix}.",
        styles["SubPDT"],
    ))
    story.append(_chart_image(_distribution_figure(data, lsl, usl, target, result.mean, relevant_sigma, characteristic, unit, sigma_label)))
    story.append(Spacer(1, 5 * mm))

    spec_width = usl - lsl
    six_sigma = 6 * relevant_sigma
    detail_rows = [
        ["Relevant sigma method", sigma_label, "6 sigma width", f"{six_sigma:.2f}{unit_suffix}"],
        ["Specification width", f"{spec_width:.2f}{unit_suffix}", "6 sigma / spec width", f"{100*six_sigma/spec_width:.1f}%"],
        ["Below LSL", str(result.below_lsl), "Above USL", str(result.above_usl)],
        ["Skewness", _fmt(result.skewness), "Excess kurtosis", _fmt(result.excess_kurtosis)],
    ]
    detail_table = Table(detail_rows, colWidths=[38 * mm, 48 * mm, 38 * mm, 47 * mm])
    detail_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GREY),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT_GREY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 5 * mm))

    method_text = (
        "Serial Production: Ppk is based on the overall sample standard deviation (STDEV.S) of all selected comparable measurements. "
        "Cumulative history should be reset or separated when material, specification, equipment, or process conditions change materially."
        if is_serial else
        "Process Batch / PPAP: Cpk uses the within-process sigma estimated from the moving range of sequential individual measurements (MR-bar / 1.128). "
        "Measurement order therefore matters."
    )
    story.append(Paragraph("Method and Usage Note", styles["H2PDT"]))
    story.append(Paragraph(method_text, styles["BodyPDT"]))
    story.append(Spacer(1, 3 * mm))
    if result.shapiro_p is not None:
        normality = "does not show strong evidence against normality" if result.shapiro_p >= 0.05 else "suggests the data may deviate from normality"
        story.append(Paragraph(f"Normality diagnostic: Shapiro-Wilk p = {result.shapiro_p:.4f}; this {normality}. Capability should be interpreted together with the plots and engineering knowledge.", styles["SmallPDT"]))

    if is_serial and history is not None and not history.empty:
        story.append(PageBreak())
        story.append(Paragraph("Serial Production - Cumulative History", styles["TitlePDT"]))
        story.append(Paragraph("Ppk is recalculated after each selected batch is added in imported order.", styles["SubPDT"]))
        story.append(_chart_image(_history_figure(history, requirement)))
        story.append(Spacer(1, 4 * mm))
        table_data = [["Batch / Lot", "Batch N", "Batch mean", "Cum. N", "Cum. mean", "Cum. Ppk"]]
        for _, row in history.iterrows():
            table_data.append([
                _safe_text(row["Batch / Lot"]),
                str(int(row["Batch N"])),
                f"{float(row['Batch mean']):.2f}",
                str(int(row["Cumulative N"])),
                f"{float(row['Cumulative mean']):.2f}",
                f"{float(row['Cumulative Ppk']):.3f}" if pd.notna(row["Cumulative Ppk"]) else "-",
            ])
        hist_table = Table(table_data, colWidths=[42*mm, 22*mm, 28*mm, 22*mm, 28*mm, 27*mm], repeatRows=1)
        hist_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(hist_table)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
