"""Central configuration: palette, dataset metadata, modes and column labels."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "Sentinel"
APP_TITLE = "Financial Fraud Intelligence"
APP_SUBTITLE = "Transaction risk monitoring, behavioural analysis and model evaluation"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

# ---------------------------------------------------------------- palette
BG = "#070D18"
PANEL = "#0E1727"
PANEL_ALT = "#131F33"
BORDER = "#1D2B43"
TEXT = "#E8EEF8"
MUTED = "#8496B0"
FAINT = "#5D7089"
ACCENT = "#38BDF8"
ACCENT_2 = "#22D3EE"
FRAUD = "#F87171"
FRAUD_SOFT = "#FCA5A5"
LEGIT = "#34D399"
AMBER = "#FBBF24"
VIOLET = "#A78BFA"
PINK = "#F472B6"

PALETTE = [ACCENT, VIOLET, LEGIT, AMBER, FRAUD, ACCENT_2, PINK, "#94A3B8"]
FRAUD_SCALE = [[0.0, "#0E1727"], [0.35, "#7F1D1D"], [0.7, "#DC2626"], [1.0, "#FCA5A5"]]
RISK_SCALE = [[0.0, "#0E1727"], [0.4, "#1E3A8A"], [0.7, "#0EA5E9"], [1.0, "#FBBF24"]]
DIVERGING = [[0.0, "#38BDF8"], [0.5, "#0E1727"], [1.0, "#F87171"]]

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

# ---------------------------------------------------------------- dataset
DATASET = {
    "key": "synthetic_fraud",
    "label": "Synthetic Fraud Ledger",
    "short": "Fraud Ledger",
    "file": "transactions.csv",
    "origin": "synthetic_fraud_dataset1.csv",
    "rows": 50_000,
    "columns": 14,
    "users": 8_963,
    "period": "01 Jan 2023 – 31 Dec 2023",
    "cities": 5,
    "volume": 4_970_550.61,
    "fraud_rate": 0.32134,
    "fraud_count": 16_067,
    "note": (
        "Fifty thousand card transactions across 8,963 users, five cities and five "
        "merchant categories. The shipped Fraud_Label is statistically independent of "
        "every recorded parameter — the Model Lab proves it and offers a documented "
        "risk-based demonstration target alongside the original labels."
    ),
}

CURRENCY = "$"

# ---------------------------------------------------------------- modes
MODES = {
    "original": {
        "label": "Original labels",
        "short": "Original",
        "target": "is_fraud",
        "eyebrow": "HONEST ANALYSIS",
        "headline": "The shipped Fraud_Label carries no learnable signal",
        "blurb": (
            "Models are trained on the dataset's own Fraud_Label. Every model lands at "
            "ROC-AUC ≈ 0.50, which is the correct result: the label was generated "
            "independently of the features. This is a data-quality finding, not a model failure."
        ),
        "accent": FRAUD,
    },
    "demo": {
        "label": "Risk-model demo",
        "short": "Risk demo",
        "target": "demo_fraud",
        "eyebrow": "SYNTHETIC DEMONSTRATION",
        "headline": "A documented risk target shows the pipeline working end to end",
        "blurb": (
            "A transparent risk function (amount, balance drain, velocity, card age, prior "
            "fraud, channel) generates a synthetic 6.1% fraud target. Models learn it to "
            "ROC-AUC ≈ 0.85. This target is fabricated for demonstration and is labelled as such everywhere."
        ),
        "accent": ACCENT,
    },
}

# ---------------------------------------------------------------- columns
RAW_COLUMNS = [
    "Transaction_ID",
    "User_ID",
    "Transaction_Amount",
    "Transaction_Type",
    "Date",
    "Account_Balance",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Card_Type",
    "Card_Age",
    "Fraud_Label",
]

CANONICAL = [
    "transaction_id",
    "user_id",
    "timestamp",
    "amount",
    "balance",
    "txn_type",
    "device",
    "location",
    "merchant",
    "prev_fraud",
    "daily_count",
    "card_type",
    "card_age",
    "is_fraud",
]

PARAMETERS = {
    "transaction_id": "Transaction ID",
    "user_id": "User ID",
    "timestamp": "Transaction date",
    "amount": "Transaction amount",
    "balance": "Account balance",
    "txn_type": "Transaction type",
    "device": "Device type",
    "location": "Location",
    "merchant": "Merchant category",
    "prev_fraud": "Previous fraudulent activity",
    "daily_count": "Daily transaction count",
    "card_type": "Card type",
    "card_age": "Card age (months)",
    "is_fraud": "Fraudulent",
}

CATEGORICAL_FIELDS = {
    "txn_type": "Transaction type",
    "device": "Device type",
    "location": "Location",
    "merchant": "Merchant category",
    "card_type": "Card type",
}

NUMERIC_FIELDS = {
    "amount": "Transaction amount",
    "balance": "Account balance",
    "daily_count": "Daily transaction count",
    "card_age": "Card age (months)",
    "prev_fraud": "Previous fraudulent activity",
}

PAGES = [
    ("overview", "Overview", "Portfolio health at a glance"),
    ("explorer", "Data explorer", "Filter and inspect every record"),
    ("patterns", "Pattern analysis", "Where fraud concentrates"),
    ("models", "Model lab", "Nine algorithms, one honest comparison"),
    ("scorer", "Transaction scorer", "Score a single transaction"),
    ("alerts", "Alert stream", "Simulated real-time monitoring"),
    ("report", "Project report", "Method, results and limitations"),
]
