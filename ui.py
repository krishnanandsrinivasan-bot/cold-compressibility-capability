from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.15rem;
                padding-bottom: 3rem;
                max-width: 1500px;
            }
            [data-testid="stSidebar"] {
                border-right: 1px solid #e7ebef;
            }
            [data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
            }
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e4e9ef;
                border-radius: 12px;
                padding: 13px 15px;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            }
            div[data-testid="stMetricLabel"] {
                font-weight: 650;
            }
            .pdt-hero {
                padding: 24px 28px;
                border-radius: 18px;
                background: linear-gradient(120deg, #102f4f 0%, #1b5d84 62%, #287aa4 100%);
                color: white;
                margin: 0 0 1.15rem 0;
                box-shadow: 0 10px 30px rgba(16, 47, 79, .14);
            }
            .pdt-hero .eyebrow {
                font-size: .78rem;
                text-transform: uppercase;
                letter-spacing: .11em;
                font-weight: 700;
                opacity: .82;
                margin-bottom: .4rem;
            }
            .pdt-hero h1 {
                margin: 0 0 .35rem 0;
                font-size: clamp(1.85rem, 3vw, 2.75rem);
                line-height: 1.08;
            }
            .pdt-hero p {
                margin: 0;
                max-width: 900px;
                opacity: .92;
                font-size: 1rem;
            }
            .pdt-section-title {
                font-size: 1.02rem;
                font-weight: 750;
                margin: .2rem 0 .4rem 0;
            }
            .pdt-card {
                border: 1px solid #e2e8ef;
                border-radius: 14px;
                padding: 16px 17px;
                min-height: 132px;
                background: #ffffff;
                box-shadow: 0 1px 4px rgba(15, 23, 42, .04);
                margin-bottom: .45rem;
            }
            .pdt-card h3 {
                margin: 0 0 .35rem 0;
                font-size: 1rem;
            }
            .pdt-card p {
                margin: 0;
                color: #5c6673;
                font-size: .9rem;
                line-height: 1.38;
            }
            .pdt-badge-live {
                display: inline-block;
                background: #e8f5e9;
                color: #236b2b;
                border: 1px solid #cce8d0;
                border-radius: 999px;
                padding: 3px 8px;
                font-size: .72rem;
                font-weight: 750;
                margin-bottom: .5rem;
            }
            .pdt-badge-roadmap {
                display: inline-block;
                background: #f2f4f7;
                color: #596270;
                border: 1px solid #e1e5ea;
                border-radius: 999px;
                padding: 3px 8px;
                font-size: .72rem;
                font-weight: 750;
                margin-bottom: .5rem;
            }
            .pdt-status-pass {
                background:#e9f7ec;
                color:#205f29;
                border:1px solid #cce8d0;
                padding:10px 14px;
                border-radius:10px;
                font-weight:750;
            }
            .pdt-status-fail {
                background:#fff0f0;
                color:#a32b2b;
                border:1px solid #f1caca;
                padding:10px 14px;
                border-radius:10px;
                font-weight:750;
            }
            .pdt-note {
                color:#5b6573;
                font-size:.9rem;
            }
            .pdt-kicker {
                color: #64748b;
                font-size: .84rem;
                font-weight: 650;
                text-transform: uppercase;
                letter-spacing: .06em;
            }
            .pdt-placeholder {
                border: 1px dashed #cfd8e3;
                border-radius: 14px;
                background: #f9fbfd;
                padding: 22px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, eyebrow: str = "ASTEMO · PAD DEVELOPMENT TOOLS") -> None:
    st.markdown(
        f"""
        <div class="pdt-hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tool_card(title: str, description: str, status: str = "Roadmap") -> None:
    badge_class = "pdt-badge-live" if status.lower() == "live" else "pdt-badge-roadmap"
    st.markdown(
        f"""
        <div class="pdt-card">
            <span class="{badge_class}">{status}</span>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def coming_soon(title: str, purpose: str, planned: list[str]) -> None:
    hero(title, purpose)
    st.info("This module is already reserved in the platform structure. It can be activated later without changing the app architecture.")
    st.markdown("### Planned capabilities")
    for item in planned:
        st.markdown(f"- {item}")
    st.markdown(
        """
        <div class="pdt-placeholder">
            <b>Module status: Roadmap</b><br>
            The navigation and product structure are ready. The engineering logic will be added as the next tools are defined.
        </div>
        """,
        unsafe_allow_html=True,
    )
