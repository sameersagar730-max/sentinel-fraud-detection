"""Visual system: CSS, KPI cards, section headers and formatters."""

from __future__ import annotations

import streamlit as st

from src.config import (
    ACCENT,
    AMBER,
    BORDER,
    FAINT,
    FONT,
    FRAUD,
    LEGIT,
    MODES,
    MUTED,
    PANEL,
    TEXT,
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: Inter, 'Segoe UI', system-ui, sans-serif; }

.stApp {
  background:
    radial-gradient(1200px 600px at 15% -10%, rgba(56,189,248,0.10), transparent 60%),
    radial-gradient(900px 500px at 95% 0%, rgba(167,139,250,0.08), transparent 55%),
    #070D18;
  color: #E8EEF8;
}
header[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

[data-testid="stMainBlockContainer"], .block-container {
  padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1480px;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A1322 0%, #070D18 100%);
  border-right: 1px solid #16233A;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
[data-testid="stSidebar"] hr { border-color: #16233A; }

h1, h2, h3, h4 { color: #E8EEF8; letter-spacing: -0.02em; }
p, li, span, label { color: #C7D4E6; }
a { color: #38BDF8; }

/* ---------- sidebar nav ---------- */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 8px 12px; border-radius: 9px; border: 1px solid transparent;
  transition: background 120ms ease, border-color 120ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #101B2E; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 13.5px; font-weight: 500; color: #B9C7DA; }
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(56,189,248,0.14), rgba(56,189,248,0.03));
  border-color: rgba(56,189,248,0.35);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: #E8EEF8; font-weight: 600; }
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child { display: none; }

/* ---------- cards ---------- */
.kpi {
  position: relative; padding: 16px 18px 15px 18px; border-radius: 14px;
  background: linear-gradient(160deg, rgba(19,31,51,0.92), rgba(11,19,33,0.92));
  border: 1px solid #1D2B43; overflow: hidden; height: 100%;
}
.kpi:before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--accent, #38BDF8);
}
.kpi-label { font-size: 10.5px; letter-spacing: 1.6px; font-weight: 700; color: #8496B0; text-transform: uppercase; }
.kpi-value { font-size: 27px; font-weight: 750; color: #F1F6FE; line-height: 1.15; margin-top: 6px; letter-spacing: -0.03em; }
.kpi-foot { font-size: 11.5px; color: #7E90A8; margin-top: 7px; }
.kpi-foot b { color: var(--accent, #38BDF8); font-weight: 650; }

.eyebrow { font-size: 10.5px; letter-spacing: 2.4px; font-weight: 800; color: #38BDF8; text-transform: uppercase; }
.sec-title { font-size: 25px; font-weight: 750; color: #F1F6FE; margin: 4px 0 2px 0; letter-spacing: -0.03em; }
.sec-sub { font-size: 13px; color: #8496B0; margin-bottom: 6px; }

.panel-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 2px; }
.panel-title { font-size: 13.5px; font-weight: 650; color: #E8EEF8; }
.panel-sub { font-size: 11px; color: #6F8299; }

.badge {
  display:inline-block; padding: 3px 9px; border-radius: 999px; font-size: 10px;
  font-weight: 700; letter-spacing: 0.9px; text-transform: uppercase;
}

.mode-banner {
  display:flex; gap:14px; align-items:flex-start; padding: 14px 18px; border-radius: 13px;
  border: 1px solid var(--edge); background: var(--wash); margin: 4px 0 6px 0;
}
.mode-banner .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--edge); margin-top: 6px; flex: 0 0 9px; box-shadow: 0 0 14px var(--edge); }
.mode-banner .mb-title { font-size: 12.5px; font-weight: 700; color: #E8EEF8; }
.mode-banner .mb-body { font-size: 12px; color: #93A5BC; line-height: 1.55; margin-top: 2px; }

.note {
  border-left: 2px solid #1E3A5F; background: rgba(15,24,41,0.7); border-radius: 0 10px 10px 0;
  padding: 11px 15px; font-size: 12px; color: #93A5BC; line-height: 1.6;
}

/* ---------- streamlit widgets ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(165deg, rgba(18,30,49,0.72), rgba(10,18,31,0.72));
  border: 1px solid #1D2B43 !important; border-radius: 14px;
}
[data-testid="stMetric"] {
  background: linear-gradient(165deg, rgba(18,30,49,0.85), rgba(10,18,31,0.85));
  border: 1px solid #1D2B43; border-radius: 13px; padding: 12px 16px;
}
[data-testid="stMetricLabel"] p { font-size: 10.5px !important; letter-spacing: 1.3px; color: #8496B0 !important; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #F1F6FE; font-weight: 700; }

.stButton > button, .stDownloadButton > button {
  border-radius: 9px; border: 1px solid #25405F; background: #101E33; color: #D9E6F7;
  font-weight: 600; font-size: 12.5px; padding: 7px 14px; transition: all 120ms ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #38BDF8; color: #FFFFFF; background: #12263F;
}
.stButton > button[kind="primary"] { background: #0E7490; border-color: #0E7490; color: white; }
.stButton > button[kind="primary"]:hover { background: #0891B2; border-color: #0891B2; }

[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stDateInput input {
  background: #0C1626 !important; border-color: #22334D !important; color: #E8EEF8 !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #38BDF8; }
.stSlider [data-baseweb="slider"] > div > div { background: #1E3A5F; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #1D2B43; }
.stTabs [data-baseweb="tab"] {
  background: transparent; border-radius: 8px 8px 0 0; padding: 7px 14px;
  font-size: 12.5px; color: #8496B0;
}
.stTabs [aria-selected="true"] { background: rgba(56,189,248,0.08); color: #E8EEF8; }

[data-testid="stDataFrame"] { border: 1px solid #1D2B43; border-radius: 11px; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid #1D2B43; border-radius: 11px; background: rgba(12,22,38,0.55); }
[data-testid="stExpander"] summary { font-size: 12.5px; color: #B9C7DA; }
div[data-testid="stAlert"] { border-radius: 10px; border: 1px solid #1D2B43; }

hr { border-color: #16233A; }
[data-testid="stCaptionContainer"] p { color: #6F8299 !important; font-size: 11px; }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: #0A1220; }
::-webkit-scrollbar-thumb { background: #1E3350; border-radius: 9px; }
::-webkit-scrollbar-thumb:hover { background: #2A466B; }

/* sidebar brand */
.brand-name { font-size: 11px; letter-spacing: 3px; font-weight: 800; color: #38BDF8; }
.brand-title { font-size: 15px; font-weight: 700; color: #E8EEF8; margin-top: 2px; }
.brand-sub { font-size: 11px; color: #6F8299; margin-top: 2px; }
.side-stat { border:1px solid #1D2B43; border-radius: 11px; padding: 10px 12px; background: #0C1526; font-size: 11.5px; line-height: 1.8; color: #A9B9CE; }
.side-stat b { color: #E8EEF8; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- formatting
def fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def fmt_money(value: float) -> str:
    return f"${value:,.0f}"


def fmt_money2(value: float) -> str:
    return f"${value:,.2f}"


def fmt_compact(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}k"
    return f"${value:,.0f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


# ---------------------------------------------------------------- components
def section_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <div class="sec-title">{title}</div>
        <div class="sec-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def panel_head(title: str, sub: str = "") -> None:
    st.markdown(
        f"""<div class="panel-head"><span class="panel-title">{title}</span>
        <span class="panel-sub">{sub}</span></div>""",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[dict]) -> None:
    """Render a row of custom KPI cards. Each item: label, value, foot, accent."""
    cols = st.columns(len(items), gap="small")
    for col, item in zip(cols, items):
        accent = item.get("accent", ACCENT)
        with col:
            st.markdown(
                f"""
                <div class="kpi" style="--accent:{accent}">
                  <div class="kpi-label">{item['label']}</div>
                  <div class="kpi-value">{item['value']}</div>
                  <div class="kpi-foot">{item.get('foot', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def badge(text: str, color: str = ACCENT) -> str:
    return (
        f"<span class='badge' style='color:{color};background:{color}18;"
        f"border:1px solid {color}44'>{text}</span>"
    )


def mode_banner(mode: str) -> None:
    meta = MODES[mode]
    accent = meta["accent"]
    st.markdown(
        f"""
        <div class="mode-banner" style="--edge:{accent};--wash:{accent}0D">
          <div class="dot"></div>
          <div>
            <div class="mb-title">{meta['eyebrow']} · {meta['headline']}</div>
            <div class="mb-body">{meta['blurb']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def spacer(height: int = 10) -> None:
    st.markdown(f"<div style='height:{height}px'></div>", unsafe_allow_html=True)
