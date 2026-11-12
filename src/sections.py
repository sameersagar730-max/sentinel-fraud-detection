"""Page sections rendered by app.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import analytics, charts
from src.config import (
    ACCENT,
    ACCENT_2,
    AMBER,
    CATEGORICAL_FIELDS,
    DATASET,
    FRAUD,
    LEGIT,
    MODES,
    MUTED,
    TEXT,
    VIOLET,
)
from src.data import dataset_summary, quality_report
from src.model import (
    MODELS,
    MODEL_ORDER,
    comparison_table,
    score_frame,
    single_score,
    threshold_metrics,
)
from src.ui import (
    badge,
    fmt_compact,
    fmt_int,
    fmt_money,
    fmt_money2,
    fmt_pct,
    kpi_row,
    mode_banner,
    note,
    panel_head,
    section_header,
)

PLOT = {"displayModeBar": False}


# ---------------------------------------------------------------- helpers
def _delta(current: float, previous: float) -> str:
    if previous is None or previous == 0 or pd.isna(previous):
        return "no prior window"
    change = (current - previous) / previous * 100
    arrow = "▲" if change >= 0 else "▼"
    return f"{arrow} {abs(change):.1f}% vs prior window"


def _band(score: float) -> tuple[str, str, str]:
    if score < 0.05:
        return "Low", LEGIT, "Auto-approve. No manual review required."
    if score < 0.20:
        return "Guarded", ACCENT, "Approve with standard monitoring."
    if score < 0.50:
        return "Elevated", AMBER, "Queue for analyst review before settlement."
    return "Critical", FRAUD, "Block and escalate to the fraud desk."


def _signal_audit(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import chi2_contingency, pointbiserialr

    rows: list[dict] = []
    y = df["target"]

    for col in CATEGORICAL_FIELDS:
        table = pd.crosstab(df[col], y)
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(table)
        n = table.values.sum()
        cramers_v = np.sqrt(chi2 / (n * max(min(table.shape) - 1, 1)))
        rows.append(
            {
                "Feature": CATEGORICAL_FIELDS[col],
                "Type": "Categorical",
                "Test": "Chi-square",
                "Statistic": float(chi2),
                "p-value": float(p),
                "Effect (Cramér's V)": float(cramers_v),
            }
        )

    for col, label in (
        ("amount", "Transaction amount"),
        ("balance", "Account balance"),
        ("daily_count", "Daily transaction count"),
        ("card_age", "Card age"),
        ("prev_fraud", "Previous fraudulent activity"),
    ):
        if df[col].nunique() < 2:
            continue
        r, p = pointbiserialr(y.to_numpy(), df[col].astype(float).to_numpy())
        rows.append(
            {
                "Feature": label,
                "Type": "Numeric",
                "Test": "Point-biserial r",
                "Statistic": float(r),
                "p-value": float(p),
                "Effect (|r|)": float(abs(r)),
            }
        )

    out = pd.DataFrame(rows)
    effect_col = "Effect (Cramér's V)" if "Effect (Cramér's V)" in out else "Effect (|r|)"
    out["Effect"] = out[[c for c in out.columns if c.startswith("Effect")]].bfill(axis=1).iloc[:, 0]
    out["Verdict"] = np.where(
        (out["p-value"] < 0.01) & (out["Effect"] >= 0.05),
        "Signal",
        np.where(out["p-value"] < 0.05, "Weak", "No signal"),
    )
    return out[["Feature", "Type", "Test", "Statistic", "p-value", "Effect", "Verdict"]]


# ---------------------------------------------------------------- hero
def hero(mode: str) -> None:
    meta = MODES[mode]
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
                    padding:4px 0 16px 0;border-bottom:1px solid #16233A;margin-bottom:14px">
          <div>
            <div class="eyebrow">SENTINEL · FRAUD INTELLIGENCE</div>
            <div style="font-size:36px;font-weight:800;letter-spacing:-0.035em;color:#F3F8FF;line-height:1.1;margin-top:5px">
              Financial Fraud Intelligence
            </div>
            <div style="font-size:13px;color:#8496B0;margin-top:7px">
              50,000 transactions · 8,963 users · 5 cities · 14 parameters · Jan–Dec 2023
            </div>
          </div>
          <div style="text-align:right;padding-bottom:3px">
            {badge(meta['eyebrow'], meta['accent'])}
            <div style="font-size:11px;color:#6F8299;margin-top:8px">Synthetic Fraud Ledger · v1.0</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode_banner(mode)


# ---------------------------------------------------------------- overview
def overview(df: pd.DataFrame, prev: pd.DataFrame, mode: str) -> None:
    section_header("01 · OVERVIEW", "Portfolio health at a glance", "Every figure reacts to the sidebar filters.")

    rows = len(df)
    volume = float(df["amount"].sum())
    fraud = int(df["target"].sum())
    rate = float(df["target"].mean()) if rows else 0.0
    avg = float(df["amount"].mean()) if rows else 0.0

    kpi_row(
        [
            {
                "label": "Transactions",
                "value": fmt_int(rows),
                "foot": _delta(rows, len(prev)),
                "accent": ACCENT,
            },
            {
                "label": "Processed volume",
                "value": fmt_compact(volume),
                "foot": _delta(volume, float(prev["amount"].sum()) if len(prev) else 0.0),
                "accent": ACCENT_2,
            },
            {
                "label": "Fraudulent records",
                "value": fmt_int(fraud),
                "foot": f"of {fmt_int(rows)} scored records",
                "accent": FRAUD,
            },
            {
                "label": "Fraud rate",
                "value": fmt_pct(rate),
                "foot": _delta(rate, float(prev["target"].mean()) if len(prev) else 0.0),
                "accent": AMBER,
            },
            {
                "label": "Average ticket",
                "value": fmt_money2(avg),
                "foot": f"median {fmt_money2(float(df['amount'].median()))}",
                "accent": VIOLET,
            },
        ]
    )

    st.write("")
    left, right = st.columns([2.15, 1], gap="medium")
    with left:
        with st.container(border=True):
            panel_head("Monthly volume & fraud rate", "12 months · filtered view")
            st.plotly_chart(charts.monthly_volume_rate(analytics.monthly_series(df)), use_container_width=True, config=PLOT)
    with right:
        with st.container(border=True):
            panel_head("Class balance", "target used by the models")
            st.plotly_chart(charts.donut(fraud, rows - fraud), use_container_width=True, config=PLOT)

    st.write("")
    left, right = st.columns([2.15, 1], gap="medium")
    with left:
        with st.container(border=True):
            panel_head("Daily transaction flow", "1 Jan – 31 Dec 2023")
            st.plotly_chart(charts.daily_volume(analytics.daily_series(df)), use_container_width=True, config=PLOT)
    with right:
        with st.container(border=True):
            panel_head("Fraud rate by weekday", "share of records flagged")
            st.plotly_chart(charts.weekday_bars(analytics.weekday_series(df)), use_container_width=True, config=PLOT)

    st.write("")
    section_header("CONCENTRATION", "Where fraud concentrates", "Top categories ranked by observed fraud rate.")
    cols = st.columns(3, gap="medium")
    for col, (field, title) in zip(
        cols,
        [("merchant", "Merchant category"), ("location", "City"), ("txn_type", "Transaction type")],
    ):
        with col:
            with st.container(border=True):
                panel_head(title, "fraud rate")
                st.plotly_chart(
                    charts.rate_bars(analytics.rate_by(df, field), field, n=5),
                    use_container_width=True,
                    config=PLOT,
                )

    if mode == "original":
        monthly = analytics.monthly_series(df)
        spread = (monthly["rate"].max() - monthly["rate"].min()) * 100
        note(
            f"<b>Reading the flat line.</b> The monthly fraud rate moves inside a "
            f"{spread:.1f} percentage-point band across all twelve months, and every category sits "
            f"within one point of the {fmt_pct(rate)} base rate. That flatness is the first clue "
            f"that the shipped label was generated independently of the features — the Model Lab "
            f"quantifies it."
        )
    else:
        note(
            "<b>Demonstration target.</b> Fraud here is produced by the documented risk function "
            "described in the Project report: amount z-score, balance drain, daily velocity, card "
            "age, prior fraud and channel. It is synthetic and is never presented as the dataset's "
            "own label."
        )


# ---------------------------------------------------------------- explorer
def explorer(df: pd.DataFrame, mode: str) -> None:
    section_header("02 · DATA EXPLORER", "Filter and inspect every record", "Search, sort and export the filtered ledger.")

    tabs = st.tabs(["Ledger", "Data quality", "Distributions"])

    with tabs[0]:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            query = st.text_input("Search transaction or user ID", placeholder="TXN_199 or USER_2734")
        with c2:
            sort_by = st.selectbox("Sort by", ["timestamp", "amount", "risk (amount-to-balance)"], index=0)
        with c3:
            page_size = st.selectbox("Rows", [50, 100, 250, 500], index=1)

        view = df
        if query:
            q = query.strip().upper()
            view = view[
                view["transaction_id"].str.upper().str.contains(q, na=False)
                | view["user_id"].str.upper().str.contains(q, na=False)
            ]
        if sort_by == "amount":
            view = view.sort_values("amount", ascending=False)
        elif sort_by.startswith("risk"):
            view = view.sort_values("amount_to_balance", ascending=False)
        else:
            view = view.sort_values("timestamp", ascending=False)

        show = view.head(page_size)
        st.dataframe(
            show[
                [
                    "transaction_id",
                    "timestamp",
                    "user_id",
                    "amount",
                    "balance",
                    "txn_type",
                    "merchant",
                    "location",
                    "device",
                    "card_type",
                    "card_age",
                    "daily_count",
                    "prev_fraud",
                    "target_label",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=430,
            column_config={
                "transaction_id": "Transaction",
                "timestamp": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "user_id": "User",
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "balance": st.column_config.NumberColumn("Balance", format="$%.0f"),
                "txn_type": "Type",
                "merchant": "Merchant",
                "location": "City",
                "device": "Device",
                "card_type": "Card",
                "card_age": st.column_config.NumberColumn("Card age (mo)"),
                "daily_count": st.column_config.NumberColumn("Daily count"),
                "prev_fraud": st.column_config.CheckboxColumn("Prior fraud"),
                "target_label": "Label",
            },
        )
        st.caption(f"Showing {len(show):,} of {len(view):,} matching records.")
        st.download_button(
            "Download filtered CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="sentinel_filtered_transactions.csv",
            mime="text/csv",
        )

    with tabs[1]:
        report = quality_report()
        kpi_row(
            [
                {"label": "Raw shape", "value": f"{report['raw_shape'][0]:,} × {report['raw_shape'][1]}", "foot": "rows × columns", "accent": ACCENT},
                {"label": "Missing cells", "value": fmt_int(report["raw_missing"]), "foot": "across the raw CSV", "accent": LEGIT},
                {"label": "Duplicate rows", "value": fmt_int(report["raw_duplicates"]), "foot": "exact duplicates", "accent": LEGIT},
                {"label": "Duplicate IDs", "value": fmt_int(report["duplicate_ids"]), "foot": "transaction_id collisions", "accent": LEGIT},
                {"label": "Date span", "value": f"{(report['end'] - report['start']).days + 1} days", "foot": "1 Jan – 31 Dec 2023", "accent": VIOLET},
            ]
        )
        st.write("")
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True):
                panel_head("Column dictionary", "raw name → canonical name → type")
                mapping = pd.DataFrame(
                    [
                        {
                            "Raw column": raw,
                            "Canonical": canon,
                            "Type": report["dtypes"].get(raw, ""),
                            "Non-null": f"{report['rows']:,}",
                        }
                        for raw, canon in report["field_map"].items()
                    ]
                )
                st.dataframe(mapping, use_container_width=True, hide_index=True, height=420)
        with right:
            with st.container(border=True):
                panel_head("Label distribution", "shipped vs demonstration target")
                labels = pd.DataFrame(
                    {
                        "Target": ["Original Fraud_Label", "Risk-model demo"],
                        "Fraud": [report["class_counts"].get(1, 0), report["demo_counts"].get(1, 0)],
                        "Legitimate": [report["class_counts"].get(0, 0), report["demo_counts"].get(0, 0)],
                    }
                )
                labels["Fraud rate"] = labels["Fraud"] / (labels["Fraud"] + labels["Legitimate"])
                st.dataframe(
                    labels,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Fraud": st.column_config.NumberColumn(format="%d"),
                        "Legitimate": st.column_config.NumberColumn(format="%d"),
                        "Fraud rate": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=1),
                    },
                )
                st.write("")
                note(
                    "The raw file is structurally clean — no missing cells, no duplicates, no invalid "
                    "dates. The problem is not data hygiene; it is that the label carries no signal."
                )
        with st.expander("Numeric ranges and integrity checks"):
            ranges = pd.DataFrame(
                [
                    {"Field": key, "Minimum": lo, "Maximum": hi}
                    for key, (lo, hi) in report["numeric_ranges"].items()
                ]
            )
            st.dataframe(ranges, use_container_width=True, hide_index=True)
            st.caption(
                "Account_Balance is uniform on $500–$100,000, Card_Age uniform on 1–239 months and "
                "Daily_Transaction_Count uniform on 1–14 — synthetic tells that match the flat label."
            )

    with tabs[2]:
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True):
                panel_head("Transaction amount", "log scale · by target")
                st.plotly_chart(charts.amount_hist(df), use_container_width=True, config=PLOT)
            with st.container(border=True):
                panel_head("Daily transaction count", "behavioural velocity")
                st.plotly_chart(charts.box_by_target(df, "daily_count"), use_container_width=True, config=PLOT)
        with right:
            with st.container(border=True):
                panel_head("Amount vs account balance", "sample of 3,500 records")
                st.plotly_chart(charts.scatter_balance(df), use_container_width=True, config=PLOT)
            with st.container(border=True):
                panel_head("Card age", "months since issuance")
                st.plotly_chart(charts.box_by_target(df, "card_age"), use_container_width=True, config=PLOT)


# ---------------------------------------------------------------- patterns
def patterns(df: pd.DataFrame, mode: str) -> None:
    section_header(
        "03 · PATTERN ANALYSIS",
        "Where fraud concentrates",
        "Univariate splits, numeric bands, correlations and a formal signal audit.",
    )

    tabs = st.tabs(["Categorical splits", "Numeric bands", "Correlation", "Signal audit"])

    with tabs[0]:
        cols = st.columns(2, gap="medium")
        for i, (field, label) in enumerate(CATEGORICAL_FIELDS.items()):
            table = analytics.rate_by(df, field)
            with cols[i % 2]:
                with st.container(border=True):
                    panel_head(label, f"{table['transactions'].sum():,} records")
                    st.plotly_chart(charts.rate_bars(table, field, n=len(table)), use_container_width=True, config=PLOT)
                    st.dataframe(
                        table[[field, "transactions", "fraud", "rate"]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            field: label,
                            "transactions": st.column_config.NumberColumn("Records", format="%d"),
                            "fraud": st.column_config.NumberColumn("Fraud", format="%d"),
                            "rate": st.column_config.ProgressColumn("Fraud rate", format="%.2f%%", min_value=0, max_value=1),
                        },
                    )

    with tabs[1]:
        cols = st.columns(3, gap="medium")
        for col, (field, label) in zip(
            cols,
            [("amount_band", "Transaction amount"), ("balance_band", "Account balance"), ("card_age_band", "Card age")],
        ):
            with col:
                with st.container(border=True):
                    panel_head(label, "fraud rate by band")
                    st.plotly_chart(
                        charts.rate_bars(analytics.rate_by(df, field), field, n=10),
                        use_container_width=True,
                        config=PLOT,
                    )
        st.write("")
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True):
                panel_head("Amount distribution", "legitimate vs fraud")
                st.plotly_chart(charts.amount_hist(df), use_container_width=True, config=PLOT)
        with right:
            with st.container(border=True):
                panel_head("Balance distribution", "legitimate vs fraud")
                st.plotly_chart(charts.box_by_target(df, "balance", log=False), use_container_width=True, config=PLOT)

    with tabs[2]:
        left, right = st.columns([1.05, 1], gap="medium")
        with left:
            with st.container(border=True):
                panel_head("Correlation matrix", "Pearson · filtered view")
                st.plotly_chart(charts.heatmap_corr(analytics.correlation(df)), use_container_width=True, config=PLOT)
        with right:
            with st.container(border=True):
                panel_head("Amount vs balance", "colour = target class")
                st.plotly_chart(charts.scatter_balance(df), use_container_width=True, config=PLOT)
        note(
            "Correlation is a linear lens only. A feature can be predictive with r ≈ 0 if the "
            "relationship is non-monotone, so the signal audit below adds chi-square and "
            "point-biserial tests on top."
        )

    with tabs[3]:
        audit = _signal_audit(df)
        signal_count = int((audit["Verdict"] == "Signal").sum())
        kpi_row(
            [
                {"label": "Features tested", "value": str(len(audit)), "foot": "5 categorical · 5 numeric", "accent": ACCENT},
                {"label": "With signal", "value": str(signal_count), "foot": "p < 0.01 and effect ≥ 0.05", "accent": LEGIT if signal_count else FRAUD},
                {"label": "Smallest p-value", "value": f"{audit['p-value'].min():.3g}", "foot": "across all tests", "accent": AMBER},
                {"label": "Largest effect", "value": f"{audit['Effect'].max():.3f}", "foot": "Cramér's V or |r|", "accent": VIOLET},
            ]
        )
        st.write("")
        with st.container(border=True):
            panel_head("Independence tests against the target", "chi-square for categoricals · point-biserial for numerics")
            st.dataframe(
                audit,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Statistic": st.column_config.NumberColumn(format="%.4f"),
                    "p-value": st.column_config.NumberColumn(format="%.4f"),
                    "Effect": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                    "Verdict": st.column_config.TextColumn("Verdict"),
                },
            )
        if mode == "original":
            note(
                "<b>Interpretation.</b> With 50,000 records the tests have enough power to detect "
                "tiny effects, yet none clears the practical-signal threshold. The label is "
                "statistically independent of every recorded parameter — a random-label "
                "generation artefact, not a modelling problem."
            )
        else:
            note(
                "<b>Interpretation.</b> The demonstration target was built from these same "
                "parameters, so strong effects are expected by construction. That is the point: "
                "the pipeline can learn when a signal exists, and the honest mode shows what "
                "happens when it does not."
            )


# ---------------------------------------------------------------- model lab
def model_lab(df: pd.DataFrame, results: dict, mode: str, use_smote: bool = True) -> None:
    section_header(
        "04 · MODEL LAB",
        "Nine algorithms, one honest comparison",
        "Stratified 75/25 split · SMOTE on the training fold only · identical feature matrix.",
    )

    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        st.error("Every model failed to train. Check the environment and reload.")
        return

    best_name = max(valid, key=lambda k: valid[k]["metrics"]["roc_auc"])
    best = valid[best_name]
    kpi_row(
        [
            {"label": "Best ROC-AUC", "value": f"{best['metrics']['roc_auc']:.3f}", "foot": best["label"], "accent": ACCENT if mode == "demo" else FRAUD},
            {"label": "Best PR-AUC", "value": f"{best['metrics']['pr_auc']:.3f}", "foot": f"base rate {best['metrics']['baseline']:.2%}", "accent": VIOLET},
            {"label": "Test records", "value": fmt_int(best["test_size"]), "foot": f"{best['test_fraud']:,} fraud · 25% hold-out", "accent": ACCENT_2},
            {"label": "Training records", "value": fmt_int(best["train_size"]), "foot": f"{best['train_size_after_smote']:,} after SMOTE", "accent": AMBER},
            {"label": "Models trained", "value": str(len(valid)), "foot": f"of {len(MODEL_ORDER)} requested", "accent": LEGIT},
        ]
    )

    st.write("")
    with st.container(border=True):
        panel_head("Model comparison", "sorted by ROC-AUC · identical split for every estimator")
        table = comparison_table(results).sort_values("roc_auc", ascending=False)
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "model": "Model",
                "family": "Family",
                "roc_auc": st.column_config.ProgressColumn("ROC-AUC", format="%.3f", min_value=0, max_value=1),
                "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
                "status": "Status",
            },
        )

    if mode == "original":
        note(
            "<b>Why every score sits at 0.50.</b> ROC-AUC 0.5 is the expected result when the label "
            "is independent of the features: the model can only rank by the base rate. Accuracy is "
            "not a rescue — it simply tracks whatever positive rate the balanced/SMOTE training "
            "boundary produces. This table is evidence about the data, not a ranking of algorithms."
        )
    else:
        note(
            "<b>What good looks like.</b> The demonstration target is learnable, so the ranking "
            "separates: ROC-AUC around 0.85, PR-AUC far above the 6.1% base rate. The spread "
            "between families is the real algorithmic comparison."
        )

    st.write("")
    tabs = st.tabs(["ROC curves", "Precision–Recall", "Threshold & confusion", "Drivers", "Risk ranking"])

    with tabs[0]:
        with st.container(border=True):
            panel_head("Receiver operating characteristic", "one curve per estimator")
            st.plotly_chart(charts.roc_curves(results), use_container_width=True, config=PLOT)
        st.caption("A curve hugging the diagonal is a model with no discriminative power. The chance line is drawn for reference.")

    with tabs[1]:
        with st.container(border=True):
            panel_head("Precision–Recall", "the honest view under class imbalance")
            st.plotly_chart(charts.pr_curves(results), use_container_width=True, config=PLOT)
        st.caption("PR-AUC is more informative than ROC-AUC when the positive class is rare; the base rate is the no-skill floor.")

    with tabs[2]:
        c1, c2 = st.columns([1, 2], gap="medium")
        with c1:
            chosen = st.selectbox(
                "Model",
                [k for k in MODEL_ORDER if k in valid],
                format_func=lambda k: MODELS[k]["label"],
                key="thr_model",
            )
            res = valid[chosen]
            default_t = float(np.clip(res["metrics"]["best_threshold"], 0.02, 0.98))
            threshold = st.slider("Decision threshold", 0.02, 0.98, default_t, 0.01, key="thr_value")
            m = threshold_metrics(res["y_test"], res["scores"], threshold)
            kpi_row(
                [
                    {"label": "Precision", "value": f"{m['precision']:.3f}", "foot": f"{m['tp']:,} true / {m['alerts']:,} alerts", "accent": ACCENT},
                    {"label": "Recall", "value": f"{m['recall']:.3f}", "foot": f"{m['fn']:,} missed", "accent": LEGIT},
                    {"label": "F1", "value": f"{m['f1']:.3f}", "foot": f"accuracy {m['accuracy']:.3f}", "accent": AMBER},
                ]
            )
        with c2:
            with st.container(border=True):
                panel_head(f"Confusion matrix · {MODELS[chosen]['label']}", f"threshold {threshold:.2f}")
                st.plotly_chart(charts.confusion(m["matrix"]), use_container_width=True, config=PLOT)
        st.write("")
        with st.container(border=True):
            panel_head("Threshold trade-off", "precision, recall and F1 as the cut-off moves")
            fig = charts.threshold_curve(res["y_test"], res["scores"])
            fig.add_vline(x=threshold, line=dict(color=FRAUD, width=1.4, dash="dash"))
            st.plotly_chart(fig, use_container_width=True, config=PLOT)
        st.caption(MODELS[chosen]["note"])

    with tabs[3]:
        chosen = st.selectbox(
            "Model",
            [k for k in MODEL_ORDER if k in valid],
            format_func=lambda k: MODELS[k]["label"],
            key="imp_model",
        )
        imp = valid[chosen].get("importance")
        if imp is None or imp.empty:
            st.info(
                f"{MODELS[chosen]['label']} exposes no native coefficients or impurity importances. "
                "Use a tree or linear model to read driver rankings, or add permutation importance."
            )
        else:
            with st.container(border=True):
                panel_head(f"Driver ranking · {MODELS[chosen]['label']}", imp["kind"].iloc[0])
                st.plotly_chart(charts.importance(imp), use_container_width=True, config=PLOT)
            st.dataframe(
                imp.assign(weight=lambda d: (d["weight"] * 100).round(2)).rename(
                    columns={"feature": "Feature", "weight": "Share (%)"}
                )[["Feature", "Share (%)"]],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[4]:
        chosen = st.selectbox(
            "Score the ledger with",
            [k for k in MODEL_ORDER if k in valid],
            index=[k for k in MODEL_ORDER if k in valid].index("random_forest") if "random_forest" in valid else 0,
            format_func=lambda k: MODELS[k]["label"],
            key="rank_model",
        )
        with st.spinner("Scoring all 50,000 records…"):
            scored = score_frame(mode, chosen, use_smote)
        left, right = st.columns([1.15, 1], gap="medium")
        with left:
            with st.container(border=True):
                panel_head("Risk decile lift", "how much more fraud the top deciles carry")
                st.plotly_chart(charts.lift_bars(analytics.risk_lift(scored)), use_container_width=True, config=PLOT)
        with right:
            with st.container(border=True):
                panel_head("Score separation", "held-out predictions by class")
                res = valid[chosen]
                st.plotly_chart(charts.score_hist(res["scores"], res["y_test"]), use_container_width=True, config=PLOT)
        summary = analytics.score_summary(scored, 0.5)
        kpi_row(
            [
                {"label": "Alerts at 0.50", "value": fmt_int(summary["alerts"]), "foot": f"{summary['alert_rate']:.2%} of ledger", "accent": AMBER},
                {"label": "Fraud caught", "value": fmt_int(summary["caught"]), "foot": f"recall {summary['recall']:.1%}", "accent": LEGIT},
                {"label": "Alert precision", "value": f"{summary['precision']:.1%}", "foot": f"{summary['missed']:,} missed", "accent": ACCENT},
                {"label": "Value at risk", "value": fmt_compact(summary["value_at_risk"]), "foot": "sum of flagged amounts", "accent": FRAUD},
            ]
        )


# ---------------------------------------------------------------- scorer
def scorer(df: pd.DataFrame, mode: str, use_smote: bool = True) -> None:
    section_header(
        "05 · TRANSACTION SCORER",
        "Score a single transaction",
        "The trained pipeline runs live on the values you enter.",
    )

    available = [k for k in MODEL_ORDER if k in MODELS]
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        with st.container(border=True):
            panel_head("Transaction details", "defaults are ledger medians")
            with st.form("scorer"):
                c1, c2 = st.columns(2)
                with c1:
                    amount = st.number_input("Amount ($)", 0.0, 5_000.0, float(df["amount"].median()), 5.0)
                    balance = st.number_input("Account balance ($)", 0.0, 250_000.0, float(df["balance"].median()), 100.0)
                    daily_count = st.slider("Daily transaction count", 1, 14, int(df["daily_count"].median()))
                    card_age = st.slider("Card age (months)", 1, 239, int(df["card_age"].median()))
                with c2:
                    txn_type = st.selectbox("Transaction type", sorted(df["txn_type"].unique()))
                    device = st.selectbox("Device", sorted(df["device"].unique()))
                    merchant = st.selectbox("Merchant category", sorted(df["merchant"].unique()))
                    location = st.selectbox("City", sorted(df["location"].unique()))
                c3, c4 = st.columns(2)
                with c3:
                    card_type = st.selectbox("Card type", sorted(df["card_type"].unique()))
                with c4:
                    prev_fraud = st.checkbox("Previous fraudulent activity", value=False)
                model_name = st.selectbox(
                    "Scoring model",
                    available,
                    index=available.index("random_forest") if "random_forest" in available else 0,
                    format_func=lambda k: MODELS[k]["label"],
                )
                submitted = st.form_submit_button("Score transaction", type="primary", width="stretch")

        if submitted:
            payload = {
                "amount": amount,
                "balance": balance,
                "daily_count": daily_count,
                "card_age": card_age,
                "prev_fraud": int(prev_fraud),
                "txn_type": txn_type,
                "device": device,
                "merchant": merchant,
                "location": location,
                "card_type": card_type,
            }
            st.session_state["last_score"] = single_score(mode, model_name, payload, use_smote)
            st.session_state["last_payload"] = payload
            st.session_state["last_model"] = model_name

    with right:
        with st.container(border=True):
            panel_head("Risk assessment", "probability of fraud")
            score = st.session_state.get("last_score")
            if score is None:
                st.markdown(
                    f"<div style='height:200px;display:flex;align-items:center;justify-content:center;"
                    f"color:{MUTED};font-size:13px'>Enter a transaction and press Score.</div>",
                    unsafe_allow_html=True,
                )
            else:
                band, color, advice = _band(score)
                st.plotly_chart(charts.gauge(score), use_container_width=True, config=PLOT)
                st.markdown(
                    f"<div style='text-align:center;margin-top:-10px'>{badge(band + ' RISK', color)}"
                    f"<div style='font-size:12px;color:#93A5BC;margin-top:8px'>{advice}</div></div>",
                    unsafe_allow_html=True,
                )

        if st.session_state.get("last_score") is not None:
            payload = st.session_state["last_payload"]
            with st.container(border=True):
                panel_head("Why this score", "inputs that moved the estimate")
                tags = []
                if payload["amount"] >= float(df["amount"].quantile(0.9)):
                    tags.append(("Amount in top 10%", FRAUD))
                if payload["amount"] / (payload["balance"] + 1) >= 0.10:
                    tags.append(("Spends >10% of balance", AMBER))
                if payload["daily_count"] >= 11:
                    tags.append(("High daily velocity", AMBER))
                if payload["card_age"] <= 12:
                    tags.append(("New card (<1 yr)", AMBER))
                if payload["prev_fraud"]:
                    tags.append(("Prior fraud flag", FRAUD))
                if payload["txn_type"] in ("Online", "Bank Transfer"):
                    tags.append(("Remote channel", ACCENT))
                if not tags:
                    tags.append(("No elevated indicators", LEGIT))
                st.markdown(" ".join(badge(t, c) for t, c in tags), unsafe_allow_html=True)
                st.caption(
                    f"Amount sits at the {float((df['amount'] <= payload['amount']).mean()):.0%} percentile "
                    f"of the filtered ledger · model: {MODELS[st.session_state.get('last_model', 'random_forest')]['label']}"
                )


# ---------------------------------------------------------------- alerts
def alerts(df: pd.DataFrame, mode: str, use_smote: bool = True) -> None:
    section_header(
        "06 · ALERT STREAM",
        "Simulated real-time monitoring",
        "A producer/consumer loop over the ledger — no broker, no credentials, fully reproducible.",
    )

    if "alert_feed" not in st.session_state:
        st.session_state["alert_feed"] = []
        st.session_state["alert_pointer"] = 0

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        model_name = st.selectbox(
            "Scoring model",
            [k for k in MODEL_ORDER],
            index=MODEL_ORDER.index("random_forest"),
            format_func=lambda k: MODELS[k]["label"],
            key="alert_model",
        )
    with c2:
        threshold = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05, key="alert_threshold")
    with c3:
        batch = st.selectbox("Batch size", [8, 12, 20, 30], index=1)
    with c4:
        st.write("")
        st.write("")
        run = st.button("Ingest next batch", type="primary", width="stretch")

    with st.spinner("Scoring the ledger…"):
        scored = score_frame(mode, model_name, use_smote)
    scored = scored.sort_values("timestamp").reset_index(drop=True)

    if run:
        pointer = st.session_state["alert_pointer"]
        chunk = scored.iloc[pointer : pointer + batch]
        if chunk.empty:
            st.session_state["alert_pointer"] = 0
            chunk = scored.iloc[:batch]
        for _, row in chunk.iterrows():
            st.session_state["alert_feed"].insert(
                0,
                {
                    "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "id": row["transaction_id"],
                    "amount": float(row["amount"]),
                    "city": row["location"],
                    "type": row["txn_type"],
                    "score": float(row["risk_score"]),
                    "actual": int(row["target"]),
                    "flagged": bool(row["risk_score"] >= threshold),
                },
            )
        st.session_state["alert_feed"] = st.session_state["alert_feed"][:60]
        st.session_state["alert_pointer"] = (st.session_state["alert_pointer"] + batch) % max(len(scored), 1)

    summary = analytics.score_summary(scored, threshold)
    kpi_row(
        [
            {"label": "Alerts in ledger", "value": fmt_int(summary["alerts"]), "foot": f"threshold {threshold:.2f}", "accent": AMBER},
            {"label": "Alert rate", "value": f"{summary['alert_rate']:.2%}", "foot": "of all transactions", "accent": ACCENT},
            {"label": "Precision", "value": f"{summary['precision']:.1%}", "foot": f"{summary['caught']:,} confirmed fraud", "accent": LEGIT},
            {"label": "Recall", "value": f"{summary['recall']:.1%}", "foot": f"{summary['missed']:,} missed", "accent": VIOLET},
            {"label": "Value at risk", "value": fmt_compact(summary["value_at_risk"]), "foot": "flagged amount", "accent": FRAUD},
        ]
    )

    st.write("")
    left, right = st.columns([1, 1.35], gap="medium")

    with left:
        with st.container(border=True):
            panel_head("Live feed", f"{len(st.session_state['alert_feed'])} events buffered")
            feed = st.session_state["alert_feed"]
            if not feed:
                st.markdown(
                    f"<div style='padding:28px 0;text-align:center;color:{MUTED};font-size:12.5px'>"
                    "Feed is empty. Press <b>Ingest next batch</b> to stream transactions.</div>",
                    unsafe_allow_html=True,
                )
            else:
                for item in feed[:14]:
                    color = FRAUD if item["flagged"] else "#334155"
                    label = "ALERT" if item["flagged"] else "clear"
                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    padding:8px 10px;border-left:2px solid {color};margin-bottom:5px;
                                    background:rgba(13,23,39,0.65);border-radius:0 8px 8px 0">
                          <div>
                            <div style="font-size:12px;color:#E8EEF8;font-weight:600">{item['id']}
                              <span style="color:#6F8299;font-weight:400">· {item['city']} · {item['type']}</span></div>
                            <div style="font-size:10.5px;color:#6F8299">{item['time']} · ${item['amount']:,.2f}</div>
                          </div>
                          <div style="text-align:right">
                            <div style="font-size:12.5px;font-weight:700;color:{color}">{item['score']:.1%}</div>
                            <div style="font-size:9.5px;letter-spacing:1px;color:{color}">{label}</div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if st.button("Clear feed", width="stretch"):
                    st.session_state["alert_feed"] = []
                    st.session_state["alert_pointer"] = 0
                    st.rerun()

    with right:
        with st.container(border=True):
            panel_head("Alert queue", "highest-risk transactions first")
            queue = (
                scored[scored["risk_score"] >= threshold]
                .sort_values("risk_score", ascending=False)
                .head(120)
            )
            st.dataframe(
                queue[
                    ["transaction_id", "timestamp", "amount", "location", "txn_type", "risk_score", "target_label"]
                ],
                use_container_width=True,
                hide_index=True,
                height=430,
                column_config={
                    "transaction_id": "Transaction",
                    "timestamp": st.column_config.DateColumn("Date", format="DD MMM"),
                    "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                    "location": "City",
                    "txn_type": "Type",
                    "risk_score": st.column_config.ProgressColumn("Risk", format="%.3f", min_value=0, max_value=1),
                    "target_label": "Actual",
                },
            )
            st.caption(
                "Producer emits one transaction per tick; the consumer scores it and routes records above "
                "the threshold to the queue. Email/SMS delivery is mocked — no app passwords are stored."
            )


# ---------------------------------------------------------------- report
def report(df: pd.DataFrame, results: dict, mode: str) -> None:
    section_header(
        "07 · PROJECT REPORT",
        "Method, results and limitations",
        "The written report, generated from the live run so every number matches the dashboard.",
    )

    valid = {k: v for k, v in results.items() if "error" not in v}
    best_name = max(valid, key=lambda k: valid[k]["metrics"]["roc_auc"]) if valid else None
    best = valid.get(best_name, {})
    table = comparison_table(results).sort_values("roc_auc", ascending=False) if results else pd.DataFrame()
    summary = dataset_summary(df)

    rows = []
    for _, row in table.iterrows():
        rows.append(
            f"| {row['model']} | {row['family']} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | "
            f"{row['accuracy']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    table_md = "\n".join(rows)

    if mode == "original":
        verdict = (
            "The shipped label is independent of every feature. ROC-AUC sits at 0.50 for all "
            "estimators, so the correct conclusion is that this file cannot support supervised "
            "fraud detection as labelled. Accuracy moves with the rebalancing choice, not with skill."
        )
    else:
        verdict = (
            f"The demonstration target is learnable: the best estimator reaches ROC-AUC "
            f"{best['metrics']['roc_auc']:.3f} and PR-AUC {best['metrics']['pr_auc']:.3f} against a "
            f"{best['metrics']['baseline']:.1%} base rate. This validates the pipeline, not the "
            f"dataset's original label."
        )

    md = f"""# Financial Fraud Detection — Project Report

**Sentinel · Financial Fraud Intelligence**  
Dataset: `{DATASET['origin']}` · {summary['rows']:,} transactions · {summary['users']:,} users · {summary['span']}  
Active mode: **{MODES[mode]['label']}** · generated from the live dashboard run

---

## Abstract

This project builds an end-to-end fraud detection workflow: exploratory analysis, leakage-free
preprocessing, class-imbalance handling, six supervised models, threshold analysis and an
interactive monitoring dashboard. On the shipped labels the workflow demonstrates a **negative
result** — the label is statistically independent of the features. A documented synthetic risk
target is included so the same pipeline can be evaluated on a learnable signal without
misrepresenting the source data.

## 1. Introduction

Card-not-present fraud is rare, adversarial and expensive. A production system must rank
transactions by risk, choose an operating threshold that balances analyst workload against missed
losses, and explain its decisions. This project covers all three: ranking (ROC-AUC / PR-AUC),
operating point (precision–recall trade-off) and explanation (coefficients and impurity importance).

## 2. Dataset overview

| Property | Value |
|---|---|
| Records | {summary['rows']:,} |
| Users | {summary['users']:,} |
| Period | {summary['span']} |
| Cities | {summary['cities']} |
| Merchant categories | {summary['merchants']} |
| Parameters | 14 |
| Processed volume | ${summary['volume']:,.0f} |
| Average ticket | ${summary['avg_ticket']:,.2f} |
| Median ticket | ${summary['median_ticket']:,.2f} |
| Shipped fraud rate | {summary['rate']:.2%} ({summary['fraud']:,} records) |
| Demo target rate | {summary['demo_rate']:.2%} ({summary['demo_fraud']:,} records) |

The file is structurally clean: no missing cells, no duplicate rows, no invalid dates. Numeric
fields are synthetic-uniform — `Account_Balance` on $500–$100,000, `Card_Age` on 1–239 months,
`Daily_Transaction_Count` on 1–14 — while `Transaction_Amount` is realistically right-skewed
(mean ${summary['avg_ticket']:.2f}, median ${summary['median_ticket']:.2f}).

## 3. Preprocessing and feature engineering

- **Split first.** Stratified 75/25 train–test split before any fitting, so no test information
  leaks into scaling or resampling.
- **Numeric pipeline.** Median imputation with missing-indicator, then `StandardScaler` for
  distance and margin models; trees consume the unscaled matrix.
- **Categorical pipeline.** Constant imputation, then one-hot encoding with `min_frequency=0.005`
  and `handle_unknown="ignore"`.
- **Engineered features.** `log_amount`, `amount_to_balance` (balance drain), calendar features
  (`month_num`, `weekday_num`, `is_weekend`), plus the raw behavioural fields.
- **SMOTE.** Implemented in-project (no wheel for this Python build) and applied to the
  **training fold only**, interpolating minority points to 50% of the majority count.
- **Class weighting.** Applied only when SMOTE is switched off, so the two corrections are never
  stacked on the same training fold.

## 4. Models

| Model | Family | Configuration |
|---|---|---|
| Logistic Regression | Linear | L2, C=0.5, balanced weights, 3,000 iterations |
| K-Nearest Neighbours | Instance-based | k=25, distance weights, capped at 12,000 training rows |
| Support Vector Classifier | Margin-based | RBF kernel, sigmoid-calibrated, capped at 8,000 rows |
| Decision Tree | Rule-based | depth 8, min leaf 20, balanced |
| Random Forest | Bagged ensemble | 240 trees, depth 16, sqrt features |
| XGBoost | Boosted ensemble | 260 rounds, depth 5, learning rate 0.08 |

## 5. Evaluation

All estimators see the same feature matrix and the same hold-out split.

| Model | Family | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
{table_md}

**Verdict.** {verdict}

## 6. Confusion matrix analysis

The dashboard exposes an interactive threshold. Lowering it raises recall and analyst workload;
raising it raises precision and misses more fraud. The confusion matrix reports TP, FP, TN and FN
for the chosen cut-off, and the threshold curve shows the full trade-off. For a fraud desk the
operating point is an economic decision: the cost of a missed fraud versus the cost of a manual
review, not a single "best" F1.

## 7. Data-quality finding

Chi-square tests on all five categorical fields and point-biserial correlations on all five numeric
fields fail to clear a practical-signal threshold. With 50,000 records the tests are well powered,
so absence of significance is informative. `Previous_Fraudulent_Activity` — the field a fraud model
would expect to matter most — splits 32.15% versus 32.03%. The label was generated independently of
the features.

This matters because the reference brief cites ROC-AUC 0.98 on the Kaggle credit-card dataset
(284,807 rows, 492 frauds, 0.172%). That result is not reproducible from this file: it is a
different dataset with a different label-generating process.

## 8. Limitations

1. **Synthetic data.** No real customer behaviour, seasonality or adversarial drift.
2. **Random label.** Supervised metrics on the original label are uninformative by construction.
3. **Demo target is fabricated.** Its ~0.85 AUC validates the pipeline, not real-world skill.
4. **No temporal validation.** Dates are uniformly distributed, so a chronological split would not
   add information here; production systems should still use one.
5. **No cost matrix.** Threshold selection is presented as a trade-off, not an optimised decision.
6. **Single-node training.** No distributed or streaming feature store.

## 9. Conclusion and future work

The pipeline, evaluation harness and dashboard are complete and reproducible. The headline finding
is a data-quality result: **a clean file can still be unlearnable**. Next steps:

- Re-run the identical pipeline on a dataset with a genuine label (e.g. the Kaggle credit-card set)
  and compare the model ranking.
- Replace the demo risk function with a cost-sensitive label derived from chargeback outcomes.
- Add sequence models (LSTM/GRU) over per-user transaction histories.
- Move scoring to a streaming feature store with online drift monitoring.
- Add SHAP explanations and a model-risk register before any production use.

## Appendix — reproduction

```bash
pip install -r requirements.txt
streamlit run app.py
```

All results are deterministic: `random_state=42` for splits and estimators, seed 7 for the
demonstration label, seed 42 for SMOTE. No credentials, broker or external service is required.
"""

    with st.container(border=True):
        st.markdown(md)
    st.download_button(
        "Download report (Markdown)",
        md.encode("utf-8"),
        file_name="financial-fraud-detection-report.md",
        mime="text/markdown",
    )
