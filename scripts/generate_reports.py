"""Regenerate the static report artifacts in reports/.

Run from the project root:  python scripts/generate_reports.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset  # noqa: E402
from src.model import MODELS, MODEL_ORDER, comparison_table, train  # noqa: E402

REPORTS = ROOT / "reports"
LEDGER = ROOT / "data" / "transactions.csv"


def _md_table(frame: pd.DataFrame) -> str:
    """Markdown table without requiring the optional `tabulate` dependency."""
    rounded = frame.copy()
    for col in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[col]):
            rounded[col] = rounded[col].map(lambda v: f"{v:.3f}")
    head = "| " + " | ".join(rounded.columns) + " |"
    rule = "| " + " | ".join("---" for _ in rounded.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rounded.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load_dataset()

    frames = []
    for mode in ("original", "demo"):
        started = time.perf_counter()
        results = {}
        for name in MODEL_ORDER:
            t0 = time.perf_counter()
            results[name] = train(mode, name, True)
            print(f"  {mode:<8} {MODELS[name]['label']:<26} {time.perf_counter() - t0:6.1f}s", flush=True)
        table = comparison_table(results)
        table.insert(0, "mode", mode)
        frames.append(table)
        print(f"{mode}: {len(results)} models in {time.perf_counter() - started:.1f}s", flush=True)

    comparison = pd.concat(frames, ignore_index=True)
    out = REPORTS / "model_comparison.csv"
    comparison.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}")

    # Label-signal evidence: fraud rate per category is flat when the label is
    # independent of the features.
    rows = []
    for col in ["txn_type", "device", "location", "merchant", "card_type"]:
        rates = df.groupby(col)["is_fraud"].agg(["mean", "size"]).reset_index()
        rates.columns = ["value", "fraud_rate", "records"]
        rates.insert(0, "feature", col)
        rows.append(rates)
    pd.concat(rows, ignore_index=True).to_csv(REPORTS / "label_signal_by_category.csv", index=False)
    print("wrote reports/label_signal_by_category.csv")

    demo = comparison[comparison["mode"] == "demo"].copy()
    orig = comparison[comparison["mode"] == "original"].copy()
    best = demo.sort_values("roc_auc", ascending=False).iloc[0]
    cols = ["model", "roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
    lines = [
        "# Financial Fraud Detection - Project Summary",
        "",
        "Sentinel / Financial Fraud Intelligence",
        "",
        f"- Dataset: `{LEDGER.name}` - {len(df):,} transactions, "
        f"{df['user_id'].nunique():,} users, "
        f"{df['timestamp'].min():%d %b %Y} to {df['timestamp'].max():%d %b %Y}",
        f"- Shipped fraud rate: {df['is_fraud'].mean():.2%} "
        f"({int(df['is_fraud'].sum()):,} of {len(df):,})",
        "- Split: stratified 75/25, SMOTE applied to the training fold only.",
        "",
        "## Headline result",
        "",
        f"On the shipped `Fraud_Label`, every model lands at ROC-AUC ~0.50 "
        f"(range {orig['roc_auc'].min():.3f} to {orig['roc_auc'].max():.3f}). "
        "That is the correct outcome for a label generated independently of the features, "
        "and it is reported as a data-quality finding rather than a modelling failure.",
        "",
        f"In the documented risk-model demo, the best estimator is "
        f"**{best['model']}** with ROC-AUC {best['roc_auc']:.3f}, PR-AUC {best['pr_auc']:.3f} "
        f"and F1 {best['f1']:.3f} at the 0.5 threshold.",
        "",
        "## Model comparison - risk-model demo",
        "",
        "```",
        _md_table(demo[cols].sort_values("roc_auc", ascending=False)),
        "```",
        "",
        "## Model comparison - shipped labels",
        "",
        "```",
        _md_table(orig[cols].sort_values("roc_auc", ascending=False)),
        "```",
        "",
        "## Method",
        "",
        "1. Exploratory analysis: class balance, amount distributions, fraud rate by channel.",
        "2. Leakage-free preprocessing: ColumnTransformer (median impute + scale for numeric,",
        "   most-frequent impute + one-hot for categorical) fitted on the training fold only.",
        "3. Class imbalance: SMOTE on the training fold, never on the hold-out.",
        "4. Nine estimators: logistic regression (L1), KNN, SVC (RBF), decision tree (depth-limited),",
        "   random forest, XGBoost, isolation forest, MLP autoencoder, and a soft-voting ensemble.",
        "5. Threshold analysis and confusion matrices on the untouched hold-out.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "streamlit run app.py",
        "python scripts/generate_reports.py",
        "```",
        "",
        "## Limitations",
        "",
        "- The shipped label is independent of the features; no model can rank it above chance.",
        "- The demo target is synthetic and exists to exercise the scoring and alerting paths.",
        "- Email/SMS delivery is mocked; no credentials are stored in the repository.",
        "",
    ]
    (REPORTS / "project-summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote reports/project-summary.md")


if __name__ == "__main__":
    main()




