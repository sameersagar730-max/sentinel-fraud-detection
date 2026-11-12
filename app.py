"""Sentinel — Financial Fraud Intelligence dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import sections
from src.config import APP_NAME, APP_TITLE, DATASET, MODES, PAGES
from src.data import load_dataset, with_target
from src.model import train_all
from src.ui import fmt_int, fmt_pct, inject_css

st.set_page_config(
    page_title=f"{APP_NAME} · {APP_TITLE}",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_LABELS = {key: f"{i + 1:02d}   {label}" for i, (key, label, _) in enumerate(PAGES)}
PAGE_SUBS = {key: sub for key, _, sub in PAGES}


# ---------------------------------------------------------------- sidebar
def _reset_filters() -> None:
    for key in list(st.session_state):
        if key.startswith("f_"):
            del st.session_state[key]
    st.rerun()


def sidebar(df: pd.DataFrame) -> tuple[str, str, dict, bool]:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:2px 0 14px 0">
              <div class="brand-name">SENTINEL</div>
              <div class="brand-title">{DATASET['label']}</div>
              <div class="brand-sub">fraud intelligence workspace</div>
            </div>
            <div class="side-stat">
              <b>{len(df):,}</b> transactions ·
              <b style="color:#F87171">{fmt_pct(df['is_fraud'].mean())}</b> shipped fraud<br>
              <b>{df['user_id'].nunique():,}</b> users · <b>{df['location'].nunique()}</b> cities ·
              14 parameters<br>
              <span style="color:#6F8299">{DATASET['origin']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("ANALYSIS MODE")
        mode = st.radio(
            "Analysis mode",
            list(MODES.keys()),
            format_func=lambda k: MODES[k]["label"],
            label_visibility="collapsed",
            key="mode",
        )
        st.caption(MODES[mode]["eyebrow"].title())

        st.markdown("---")
        st.caption("NAVIGATION")
        page = st.radio(
            "Page",
            [key for key, _, _ in PAGES],
            format_func=lambda k: PAGE_LABELS[k],
            label_visibility="collapsed",
            key="page",
        )
        st.caption(PAGE_SUBS[page])

        st.markdown("---")
        st.caption("MODEL SETTINGS")
        use_smote = st.checkbox(
            "SMOTE oversampling",
            value=True,
            key="use_smote",
            help="Interpolate minority points on the training fold only. When off, class weights are used instead.",
        )

        st.markdown("---")
        st.caption("FILTERS")

        start, end = df["timestamp"].min().date(), df["timestamp"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(start, end),
            min_value=start,
            max_value=end,
            key="f_dates",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            d0, d1 = date_range
        else:
            d0, d1 = start, end

        lo, hi = float(df["amount"].min()), float(df["amount"].max())
        amount_range = st.slider(
            "Transaction amount",
            min_value=round(lo, 2),
            max_value=round(hi, 2),
            value=(round(lo, 2), round(hi, 2)),
            step=max(round((hi - lo) / 250, 2), 0.01),
            key="f_amount",
        )

        txn_types = st.multiselect("Transaction type", sorted(df["txn_type"].unique()), key="f_type")
        devices = st.multiselect("Device type", sorted(df["device"].unique()), key="f_device")
        merchants = st.multiselect("Merchant category", sorted(df["merchant"].unique()), key="f_merchant")
        cities = st.multiselect("City", sorted(df["location"].unique()), key="f_city")
        cards = st.multiselect("Card type", sorted(df["card_type"].unique()), key="f_card")

        with st.expander("Behavioural filters"):
            prev_fraud = st.checkbox("Prior fraudulent activity only", key="f_prev")
            young = st.checkbox("Cards younger than 12 months", key="f_young")
            high_velocity = st.checkbox("11+ transactions per day", key="f_velocity")

        fraud_only = st.checkbox("Fraud-labelled records only", key="f_fraud")
        st.button("Reset filters", width="stretch", on_click=_reset_filters)

    return mode, page, {
        "d0": d0,
        "d1": d1,
        "amount": amount_range,
        "txn_type": txn_types,
        "device": devices,
        "merchant": merchants,
        "city": cities,
        "card": cards,
        "prev_fraud": prev_fraud,
        "young": young,
        "velocity": high_velocity,
        "fraud": fraud_only,
    }, use_smote


def apply_filters(df: pd.DataFrame, opts: dict, with_dates: bool = True) -> pd.DataFrame:
    out = df
    if with_dates:
        out = out[
            (out["timestamp"] >= pd.Timestamp(opts["d0"]))
            & (out["timestamp"] < pd.Timestamp(opts["d1"]) + pd.Timedelta(days=1))
        ]
    out = out[(out["amount"] >= opts["amount"][0]) & (out["amount"] <= opts["amount"][1])]
    for key, col in (
        ("txn_type", "txn_type"),
        ("device", "device"),
        ("merchant", "merchant"),
        ("city", "location"),
        ("card", "card_type"),
    ):
        if opts[key]:
            out = out[out[col].isin(opts[key])]
    if opts["prev_fraud"]:
        out = out[out["prev_fraud"] == 1]
    if opts["young"]:
        out = out[out["card_age"] < 12]
    if opts["velocity"]:
        out = out[out["daily_count"] >= 11]
    if opts["fraud"]:
        out = out[out["is_fraud"] == 1]
    return out


def previous_window(full: pd.DataFrame, opts: dict) -> pd.DataFrame:
    """Equally sized window immediately before the selected date range."""
    start = pd.Timestamp(opts["d0"])
    end = pd.Timestamp(opts["d1"]) + pd.Timedelta(days=1)
    span = end - start
    prev_end = start - pd.Timedelta(seconds=1)
    prev_start = prev_end - span
    base = apply_filters(full, opts, with_dates=False)
    return base[(base["timestamp"] >= prev_start) & (base["timestamp"] <= prev_end)]


# ---------------------------------------------------------------- page
def main() -> None:
    inject_css()
    full = load_dataset()
    mode, page, opts, use_smote = sidebar(full)

    filtered = apply_filters(full, opts)
    df = with_target(filtered, mode)

    sections.hero(mode)

    if df.empty:
        st.warning("No transactions match the current filters. Widen the date or amount range.")
        return

    if page == "overview":
        sections.overview(df, with_target(previous_window(full, opts), mode), mode)

    elif page == "explorer":
        sections.explorer(df, mode)

    elif page == "patterns":
        sections.patterns(df, mode)

    elif page == "models":
        with st.spinner("Training nine models on 50,000 records — first load only…"):
            results = train_all(mode, use_smote)
        sections.model_lab(df, results, mode, use_smote)

    elif page == "scorer":
        sections.scorer(df, mode, use_smote)

    elif page == "alerts":
        sections.alerts(df, mode, use_smote)

    elif page == "report":
        with st.spinner("Training models for the report — first load only…"):
            results = train_all(mode, use_smote)
        sections.report(df, results, mode)

    st.markdown(
        f"<div style='margin-top:34px;padding-top:14px;border-top:1px solid #16233A;"
        f"font-size:10.5px;color:#5D7089;display:flex;justify-content:space-between'>"
        f"<span>SENTINEL · {DATASET['label']} · {fmt_int(len(df))} filtered records</span>"
        f"<span>Streamlit · scikit-learn · Plotly · deterministic seeds</span></div>",
        unsafe_allow_html=True,
    )


main()
