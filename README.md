# Sentinel - Financial Fraud Intelligence

An end-to-end fraud detection project: exploratory analysis, leakage-free preprocessing,
class-imbalance handling, nine estimators, threshold analysis, live transaction scoring,
and a simulated real-time alert stream - all in one Streamlit dashboard.

Built from `synthetic_fraud_dataset1.csv` (50,000 transactions, 8,963 users, calendar year 2023)
and the accompanying notebook and report.

---

## The honest headline

The shipped `Fraud_Label` is **statistically independent of every recorded feature**.

- Logistic regression on the shipped label: ROC-AUC **0.492**
- Random forest: ROC-AUC **0.497**
- Best of nine estimators: ROC-AUC **0.506**
- Fraud rate is ~0.32 in every category of transaction type, device, city, merchant and card type
- All nine models converge on ROC-AUC ~0.50, which is exactly what theory predicts for a label
  drawn independently of the predictors

The reference report's figures (284,807 transactions, 492 frauds, 0.172% rate, ~0.98 AUC) come
from the well-known Kaggle credit-card dataset, not from this file. Rather than hide the
discrepancy, the app reports it as a **data-quality finding** and proves it in the Model Lab.

To keep the rest of the pipeline meaningful, the app ships a second, clearly-labelled
**Risk-model demo** target: a transparent risk function over amount, balance drain, velocity,
card age, prior fraud and channel. Models learn that target to ROC-AUC ~0.85, which exercises
the scorer, alert stream and report end to end.

| Mode | Target | What it demonstrates |
| --- | --- | --- |
| Original labels | `is_fraud` | The label carries no signal - a data audit result |
| Risk-model demo | `demo_fraud` | The pipeline working on a learnable, documented target |

---

## Pages

| Page | Contents |
| --- | --- |
| 01 Overview | Portfolio KPIs, monthly volume vs fraud rate, class balance, channel mix |
| 02 Data explorer | Filterable ledger, column profiles, quality checks, raw-record inspector |
| 03 Pattern analysis | Fraud rate by channel/city/merchant, amount bands, velocity, card age |
| 04 Model lab | Nine estimators, ROC/PR curves, threshold sweep, confusion matrix, drivers |
| 05 Transaction scorer | Live scoring of a single transaction through the trained pipeline |
| 06 Alert stream | Producer/consumer simulation with threshold routing and a risk queue |
| 07 Project report | Method, results and limitations generated from the live run |

Every page reacts to the sidebar filters (date range, city, channel, amount) and the mode toggle.

## Models

| Estimator | Notes |
| --- | --- |
| Logistic Regression | L1 penalty, scaled numeric features |
| K-Nearest Neighbours | distance-weighted, k tuned for speed |
| Support Vector Classifier | RBF kernel, probability calibration, capped training sample |
| Decision Tree | depth-limited to control variance |
| Random Forest | 300 trees, balanced subsample |
| XGBoost | gradient boosting, histogram tree method |
| Isolation Forest | unsupervised anomaly detector, label never used in fitting |
| AutoEncoder | MLP reconstruction error on legitimate rows only |
| Soft-Voting Ensemble | equal-weight average of the five supervised members |

Preprocessing is a `ColumnTransformer` (median impute + scale for numeric, most-frequent impute +
one-hot for categorical) fitted on the training fold only. SMOTE is applied to the training fold
only, never to the hold-out. Every estimator sees the identical stratified 75/25 split.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first Model Lab visit trains all nine estimators (about 60-90 seconds); results are cached
for the rest of the session. Other pages load immediately.

## Regenerate the report artifacts

```bash
python scripts/generate_reports.py
```

Writes:

- `reports/model_comparison.csv` - metrics for all nine models in both modes
- `reports/label_signal_by_category.csv` - fraud rate per category (the independence evidence)
- `reports/project-summary.md` - narrative summary with the comparison tables

## Project structure

```
fraud-detection/
  app.py                  Streamlit entry point, sidebar, page dispatch
  requirements.txt
  .streamlit/config.toml  dark theme
  data/transactions.csv   working ledger (canonical column names)
  src/
    config.py             palette, dataset metadata, modes, column labels
    data.py               loading, canonicalisation, filters, demo target, quality checks
    analytics.py          aggregations behind the charts
    model.py              preprocessing, nine estimators, metrics, scoring
    charts.py             Plotly figure builders
    sections.py           page renderers
    ui.py                 CSS, KPI cards, badges, formatters
  scripts/
    generate_reports.py   headless report generation
  reports/                generated artifacts
```

## Reproducibility

- Deterministic seeds (`random_state=42`) for the split, SMOTE, and every estimator
- Identical feature matrix and split for all models
- No network calls, no credentials, no external services
- Email/SMS delivery in the alert stream is simulated and clearly labelled as such

## Limitations

- The shipped label is independent of the features; no model can rank it above chance.
- The demo target is synthetic. It exists to demonstrate the pipeline, not to claim real-world lift.
- The ledger is synthetic and small; production fraud systems need far more data, entity history
  and drift monitoring.
- Unsupervised detectors are included for comparison, not as production-ready anomaly systems.
