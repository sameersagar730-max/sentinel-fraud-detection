"""Aggregations shared by the dashboard sections."""

from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("month")
        .agg(transactions=("transaction_id", "size"), volume=("amount", "sum"), fraud=("target", "sum"))
        .reset_index()
    )
    out["rate"] = out["fraud"] / out["transactions"].clip(lower=1)
    return out


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("date")
        .agg(transactions=("transaction_id", "size"), volume=("amount", "sum"), fraud=("target", "sum"))
        .reset_index()
    )
    out["rate"] = out["fraud"] / out["transactions"].clip(lower=1)
    return out


def rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = (
        df.groupby(col, observed=True)
        .agg(transactions=("transaction_id", "size"), fraud=("target", "sum"), volume=("amount", "sum"))
        .reset_index()
    )
    out["rate"] = out["fraud"] / out["transactions"].clip(lower=1)
    out["share"] = out["transactions"] / max(len(df), 1)
    return out.sort_values("rate", ascending=False)


def weekday_series(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    out = rate_by(df, "weekday")
    out["weekday"] = pd.Categorical(out["weekday"], categories=order, ordered=True)
    return out.sort_values("weekday")


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["amount", "balance", "daily_count", "card_age", "prev_fraud", "target"]
    labels = {
        "amount": "Amount",
        "balance": "Balance",
        "daily_count": "Daily count",
        "card_age": "Card age",
        "prev_fraud": "Prior fraud",
        "target": "Fraud",
    }
    return df[cols].rename(columns=labels).corr(numeric_only=True)


def risk_lift(scored: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    out = scored.copy()
    out["decile"] = pd.qcut(out["risk_score"].rank(method="first"), bins, labels=False) + 1
    agg = (
        out.groupby("decile")
        .agg(transactions=("transaction_id", "size"), fraud=("target", "sum"), avg_score=("risk_score", "mean"))
        .reset_index()
    )
    agg["rate"] = agg["fraud"] / agg["transactions"].clip(lower=1)
    overall = max(out["target"].mean(), 1e-9)
    agg["lift"] = agg["rate"] / overall
    return agg


def concentration(df: pd.DataFrame, col: str, top: int = 4) -> pd.DataFrame:
    table = rate_by(df, col)
    table["fraud_share"] = table["fraud"] / max(table["fraud"].sum(), 1)
    return table.head(top)


def score_summary(scored: pd.DataFrame, threshold: float) -> dict:
    alerts = scored[scored["risk_score"] >= threshold]
    caught = int(alerts["target"].sum())
    total_fraud = int(scored["target"].sum())
    return {
        "alerts": int(len(alerts)),
        "alert_rate": float(len(alerts) / max(len(scored), 1)),
        "caught": caught,
        "recall": float(caught / max(total_fraud, 1)),
        "precision": float(caught / max(len(alerts), 1)),
        "value_at_risk": float(alerts["amount"].sum()),
        "missed": int(total_fraud - caught),
    }


def numeric_profile(df: pd.DataFrame, col: str) -> dict:
    s = df[col].astype(float)
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p95": float(np.percentile(s, 95)),
        "max": float(s.max()),
        "min": float(s.min()),
    }
