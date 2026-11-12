"""Load, clean, enrich and label the fraud ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import CANONICAL, DATA_DIR, DATASET, MODES, RAW_COLUMNS

FIELD_MAP = dict(zip(RAW_COLUMNS, CANONICAL))

# Coefficients of the documented demonstration risk function. Every term is
# visible to the reader in the Project report; nothing is hidden in a black box.
DEMO_INTERCEPT = -5.4
DEMO_WEIGHTS = {
    "amount_z": 1.05,
    "velocity": 1.00,
    "young_card": 1.40,
    "prior_fraud": 1.90,
    "balance_drain": 1.20,
    "txn_type": 1.00,
    "merchant": 1.00,
    "device": 1.00,
}
TYPE_RISK = {"Online": 0.65, "Bank Transfer": 0.55, "ATM Withdrawal": 0.25, "POS": 0.00}
MERCHANT_RISK = {
    "Travel": 0.35,
    "Electronics": 0.35,
    "Restaurants": 0.10,
    "Clothing": 0.10,
    "Groceries": 0.00,
}
DEVICE_RISK = {"Laptop": 0.20, "Tablet": 0.10, "Mobile": 0.00}


def _parse_dates(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    for fmt in ("%d %B %Y", "%d %b %Y", "%d-%m-%Y", "%Y-%m-%d"):
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.98:
            return parsed
    return pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def demo_probability(df: pd.DataFrame) -> np.ndarray:
    """Documented risk score in [0, 1] used to build the demonstration target."""
    amount = df["amount"].to_numpy(dtype=float)
    balance = df["balance"].to_numpy(dtype=float)
    count = df["daily_count"].to_numpy(dtype=float)
    age = df["card_age"].to_numpy(dtype=float)
    prior = df["prev_fraud"].to_numpy(dtype=float)

    amount_z = np.clip((amount - 99.4) / 98.7, -1.2, 4.0)
    drain = np.clip(amount / (balance + 1.0), 0.0, 0.35) * 5.0
    velocity = (count - 7.5) / 4.0
    young = np.clip(1.0 - age / 240.0, 0.0, 1.0)

    type_risk = df["txn_type"].map(TYPE_RISK).fillna(0.0).to_numpy(dtype=float)
    merch_risk = df["merchant"].map(MERCHANT_RISK).fillna(0.0).to_numpy(dtype=float)
    device_risk = df["device"].map(DEVICE_RISK).fillna(0.0).to_numpy(dtype=float)

    logit = (
        DEMO_INTERCEPT
        + DEMO_WEIGHTS["amount_z"] * amount_z
        + DEMO_WEIGHTS["velocity"] * velocity
        + DEMO_WEIGHTS["young_card"] * young
        + DEMO_WEIGHTS["prior_fraud"] * prior
        + DEMO_WEIGHTS["balance_drain"] * drain
        + DEMO_WEIGHTS["txn_type"] * type_risk
        + DEMO_WEIGHTS["merchant"] * merch_risk
        + DEMO_WEIGHTS["device"] * device_risk
    )
    return _sigmoid(logit)


def raw_frame() -> pd.DataFrame:
    """The CSV exactly as shipped (used by the data-integrity panel)."""
    return pd.read_csv(DATA_DIR / DATASET["file"])


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load, normalise and enrich the ledger into one canonical frame."""
    raw = raw_frame()
    out = pd.DataFrame(index=raw.index)
    out["transaction_id"] = raw["Transaction_ID"].astype(str)
    out["user_id"] = raw["User_ID"].astype(str)
    out["timestamp"] = _parse_dates(raw["Date"])
    out["amount"] = pd.to_numeric(raw["Transaction_Amount"], errors="coerce")
    out["balance"] = pd.to_numeric(raw["Account_Balance"], errors="coerce")
    out["txn_type"] = raw["Transaction_Type"].astype(str).str.strip()
    out["device"] = raw["Device_Type"].astype(str).str.strip()
    out["location"] = raw["Location"].astype(str).str.strip()
    out["merchant"] = raw["Merchant_Category"].astype(str).str.strip()
    out["prev_fraud"] = pd.to_numeric(raw["Previous_Fraudulent_Activity"], errors="coerce").fillna(0).astype(int)
    out["daily_count"] = pd.to_numeric(raw["Daily_Transaction_Count"], errors="coerce")
    out["card_type"] = raw["Card_Type"].astype(str).str.strip()
    out["card_age"] = pd.to_numeric(raw["Card_Age"], errors="coerce")
    out["is_fraud"] = pd.to_numeric(raw["Fraud_Label"], errors="coerce").fillna(0).astype(int)

    out = out[CANONICAL].copy()
    out = out.dropna(subset=["timestamp", "amount", "balance"]).reset_index(drop=True)
    out["amount"] = out["amount"].clip(lower=0)
    out["balance"] = out["balance"].clip(lower=0)

    out["date"] = out["timestamp"].dt.normalize()
    out["month"] = out["timestamp"].dt.to_period("M").dt.to_timestamp()
    out["month_num"] = out["timestamp"].dt.month
    out["weekday"] = out["timestamp"].dt.day_name()
    out["weekday_num"] = out["timestamp"].dt.weekday
    out["is_weekend"] = (out["weekday_num"] >= 5).astype(int)
    out["amount_to_balance"] = out["amount"] / (out["balance"] + 1.0)

    out["amount_band"] = pd.cut(
        out["amount"],
        bins=[-0.01, 25, 50, 100, 250, 500, np.inf],
        labels=["<$25", "$25–50", "$50–100", "$100–250", "$250–500", "$500+"],
    )
    out["balance_band"] = pd.cut(
        out["balance"],
        bins=[-0.01, 10_000, 25_000, 50_000, 75_000, np.inf],
        labels=["<$10k", "$10–25k", "$25–50k", "$50–75k", "$75k+"],
    )
    out["card_age_band"] = pd.cut(
        out["card_age"],
        bins=[-0.01, 12, 36, 60, 120, 180, np.inf],
        labels=["<1 yr", "1–3 yr", "3–5 yr", "5–10 yr", "10–15 yr", "15 yr+"],
    )

    out["demo_prob"] = demo_probability(out)
    rng = np.random.default_rng(7)
    out["demo_fraud"] = rng.binomial(1, out["demo_prob"].to_numpy())

    out["fraud_label"] = np.where(out["is_fraud"] == 1, "Fraud", "Legitimate")
    out["demo_label"] = np.where(out["demo_fraud"] == 1, "Fraud", "Legitimate")
    return out


def with_target(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return a copy carrying `target` (0/1) and `target_label` for the chosen mode."""
    target = MODES[mode]["target"]
    out = df.copy()
    out["target"] = out[target].astype(int)
    out["target_label"] = np.where(out["target"] == 1, "Fraud", "Legitimate")
    return out


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "users": int(df["user_id"].nunique()),
        "fraud": int(df["is_fraud"].sum()),
        "rate": float(df["is_fraud"].mean()),
        "volume": float(df["amount"].sum()),
        "avg_ticket": float(df["amount"].mean()),
        "median_ticket": float(df["amount"].median()),
        "span": f"{df['timestamp'].min():%d %b %Y} – {df['timestamp'].max():%d %b %Y}",
        "cities": int(df["location"].nunique()),
        "merchants": int(df["merchant"].nunique()),
        "demo_fraud": int(df["demo_fraud"].sum()),
        "demo_rate": float(df["demo_fraud"].mean()),
    }


def quality_report() -> dict:
    raw = raw_frame()
    df = load_dataset()
    missing = raw.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return {
        "raw_shape": raw.shape,
        "raw_missing": int(raw.isna().sum().sum()),
        "raw_duplicates": int(raw.duplicated().sum()),
        "duplicate_ids": int(df["transaction_id"].duplicated().sum()),
        "missing": missing.to_dict(),
        "class_counts": df["is_fraud"].value_counts().to_dict(),
        "demo_counts": df["demo_fraud"].value_counts().to_dict(),
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "columns": list(raw.columns),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "field_map": FIELD_MAP,
        "rows": len(df),
        "numeric_ranges": {
            c: (float(df[c].min()), float(df[c].max()))
            for c in ["amount", "balance", "daily_count", "card_age", "prev_fraud"]
        },
    }
