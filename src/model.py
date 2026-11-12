"""Feature engineering and model training for the fraud lab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODES
from src.data import load_dataset

NUMERIC = [
    "amount",
    "log_amount",
    "balance",
    "amount_to_balance",
    "daily_count",
    "card_age",
    "prev_fraud",
    "month_num",
    "weekday_num",
    "is_weekend",
]
CATEGORICAL = ["txn_type", "device", "location", "merchant", "card_type"]

MODELS = {
    "logistic": {
        "label": "Logistic Regression",
        "short": "Logistic",
        "family": "Linear",
        "scale": True,
        "max_train": None,
        "note": "Interpretable linear baseline; coefficients read directly as log-odds.",
    },
    "knn": {
        "label": "K-Nearest Neighbours",
        "short": "KNN",
        "family": "Instance-based",
        "scale": True,
        "max_train": 12_000,
        "note": "Distance-based scoring; sensitive to scaling and local density.",
    },
    "svc": {
        "label": "Support Vector Classifier",
        "short": "SVC",
        "family": "Margin-based",
        "scale": True,
        "max_train": 8_000,
        "note": "RBF margin classifier, sigmoid-calibrated so scores are probabilities.",
    },
    "decision_tree": {
        "label": "Decision Tree",
        "short": "Tree",
        "family": "Rule-based",
        "scale": False,
        "max_train": None,
        "note": "Depth-limited rule tree; easy to explain, prone to instability.",
    },
    "random_forest": {
        "label": "Random Forest",
        "short": "Forest",
        "family": "Ensemble",
        "scale": False,
        "max_train": None,
        "note": "Averaged decorrelated trees; the general-purpose leader on tabular data.",
    },
    "xgboost": {
        "label": "XGBoost",
        "short": "XGBoost",
        "family": "Boosted ensemble",
        "scale": False,
        "max_train": None,
        "note": "Gradient-boosted trees; strong ranking quality on structured features.",
    },
    "isolation_forest": {
        "label": "Isolation Forest",
        "short": "IsoForest",
        "family": "Anomaly detection",
        "scale": True,
        "max_train": None,
        "note": "Unsupervised: isolates outliers by random partitioning. Uses labels only for scoring, never for fitting.",
    },
    "autoencoder": {
        "label": "AutoEncoder",
        "short": "AutoEnc",
        "family": "Neural network",
        "scale": True,
        "max_train": 15_000,
        "note": "Reconstruction autoencoder trained on legitimate rows only; high reconstruction error signals fraud.",
    },
    "ensemble": {
        "label": "Soft-Voting Ensemble",
        "short": "Ensemble",
        "family": "Ensemble",
        "scale": True,
        "max_train": 10_000,
        "note": "Averages calibrated probabilities from Logistic, Tree, Forest, XGBoost and SVC.",
    },
}

MODEL_ORDER = [
    "logistic",
    "knn",
    "svc",
    "decision_tree",
    "random_forest",
    "xgboost",
    "isolation_forest",
    "autoencoder",
    "ensemble",
]

# Models that do not consume the label during fitting.
UNSUPERVISED = {"isolation_forest", "autoencoder"}

# Soft-voting members: the five supervised families in the notebook's ensemble.
ENSEMBLE_MEMBERS = ["logistic", "decision_tree", "random_forest", "xgboost", "svc"]


# ---------------------------------------------------------------- features
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Shared feature engineering for a ledger slice or a single row."""
    X = pd.DataFrame(index=df.index)
    amount = df["amount"].astype(float)
    X["amount"] = amount
    X["log_amount"] = np.log1p(amount.clip(lower=0))
    X["balance"] = df["balance"].astype(float)
    X["amount_to_balance"] = amount / (df["balance"].astype(float) + 1.0)
    X["daily_count"] = df["daily_count"].astype(float)
    X["card_age"] = df["card_age"].astype(float)
    X["prev_fraud"] = df["prev_fraud"].astype(float)
    X["month_num"] = df["timestamp"].dt.month.astype(float)
    X["weekday_num"] = df["timestamp"].dt.weekday.astype(float)
    X["is_weekend"] = (df["timestamp"].dt.weekday >= 5).astype(float)

    for col in CATEGORICAL:
        X[col] = df[col].fillna("Unknown").astype(str)

    return X[NUMERIC + CATEGORICAL]


def build_features(df: pd.DataFrame, mode: str = "original") -> tuple[pd.DataFrame, pd.Series]:
    """Build the shared feature matrix and the target for the chosen mode."""
    y = df[MODES[mode]["target"]].astype(int)
    return _engineer(df), y


def feature_row(payload: dict) -> pd.DataFrame:
    """One-row frame for the interactive scorer, shaped like build_features output."""
    ts = pd.Timestamp(payload.get("timestamp", "2023-06-15"))
    row = {
        "amount": float(payload["amount"]),
        "balance": float(payload["balance"]),
        "daily_count": float(payload["daily_count"]),
        "card_age": float(payload["card_age"]),
        "prev_fraud": int(payload["prev_fraud"]),
        "txn_type": str(payload["txn_type"]),
        "device": str(payload["device"]),
        "location": str(payload["location"]),
        "merchant": str(payload["merchant"]),
        "card_type": str(payload["card_type"]),
        "timestamp": ts,
    }
    return _engineer(pd.DataFrame([row]))


# ---------------------------------------------------------------- preprocessing
def build_preprocessor(scale: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=0.005,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(name: str, balanced: bool = True):
    """Build an estimator. `balanced=False` is used when SMOTE already rebalances."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    weight = "balanced" if balanced else None
    if name == "logistic":
        return LogisticRegression(class_weight=weight, max_iter=3000, C=0.5, solver="lbfgs")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    if name == "svc":
        return CalibratedClassifierCV(
            SVC(C=1.0, gamma="scale", class_weight=weight, random_state=42),
            cv=3,
            method="sigmoid",
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight=weight, random_state=42
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=240,
            max_depth=16,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample" if balanced else None,
            n_jobs=-1,
            random_state=42,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=260,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
            scale_pos_weight=2.1 if balanced else 1.0,
        )
    if name == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            n_jobs=-1,
            random_state=42,
        )
    if name == "autoencoder":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=(8, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=80,
            early_stopping=True,
            n_iter_no_change=6,
            validation_fraction=0.12,
            random_state=42,
        )
    if name == "ensemble":
        raise ValueError("The ensemble is assembled from fitted members in train(), not built here.")
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------- SMOTE
def smote_balance(
    X: np.ndarray,
    y: np.ndarray,
    ratio: float = 0.5,
    k: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal SMOTE: interpolate new minority points until minority = ratio × majority.

    Implemented in-project because imbalanced-learn has no wheel for this Python
    build. Identical algorithm to the library default, minus the extra features.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    minority = X[y == 1]
    n_majority = int((y == 0).sum())
    n_target = int(round(ratio * n_majority))
    n_new = n_target - len(minority)
    if n_new <= 0 or len(minority) < 2:
        return X, y

    k = min(k, len(minority) - 1)
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority)
    base_idx = rng.integers(0, len(minority), n_new)
    neighbours = nn.kneighbors(minority[base_idx], return_distance=False)[:, 1:]
    partner = neighbours[np.arange(n_new), rng.integers(0, k, n_new)]
    gap = rng.random((n_new, X.shape[1]))
    synthetic = minority[base_idx] + gap * (minority[partner] - minority[base_idx])

    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_new, dtype=int)])


# ---------------------------------------------------------------- training
class FittedModel:
    """Preprocessor + estimator pair with a DataFrame-friendly scoring API.

    Three scoring modes keep one interface for very different estimators:
      prob           - supervised classifiers expose predict_proba directly
      anomaly        - Isolation Forest: negated decision_function, squashed to [0, 1]
      reconstruction - AutoEncoder: mean squared reconstruction error, squashed to [0, 1]
    """

    def __init__(self, pre, clf, kind: str = "prob", center: float = 0.0, spread: float = 1.0):
        self.pre = pre
        self.clf = clf
        self.kind = kind
        self.center = float(center)
        self.spread = float(spread) if spread else 1.0

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.pre.transform(X)
        if self.kind == "prob":
            return self.clf.predict_proba(Z)[:, 1]
        if self.kind == "anomaly":
            return -self.clf.decision_function(Z)
        if self.kind == "reconstruction":
            Z = np.asarray(Z, dtype=float)
            recon = np.asarray(self.clf.predict(Z), dtype=float)
            return np.mean((Z - recon) ** 2, axis=1)
        raise ValueError(f"Unknown scoring kind: {self.kind}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.raw_score(X)
        if self.kind == "prob":
            return raw
        z = (raw - self.center) / self.spread
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class AveragingModel:
    """Equal-weight soft voting over already-fitted member pipelines."""

    kind = "prob"

    def __init__(self, members: list):
        self.members = members

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.members:
            raise ValueError("Ensemble has no fitted members")
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


def _stratified_cap(y: pd.Series, idx: np.ndarray, cap: int, seed: int = 42) -> np.ndarray:
    if cap is None or len(idx) <= cap:
        return idx
    from sklearn.model_selection import train_test_split

    keep, _ = train_test_split(
        idx, train_size=cap, stratify=y.loc[idx], random_state=seed
    )
    return np.asarray(keep)


def _aggregate_importance(pre, clf, categorical: list[str]) -> pd.DataFrame | None:
    names = list(pre.get_feature_names_out())
    if hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_, dtype=float)
        kind = "impurity importance"
    elif hasattr(clf, "coef_"):
        values = np.abs(np.asarray(clf.coef_[0], dtype=float))
        kind = "absolute coefficient"
    else:
        return None

    cat_sorted = sorted(categorical, key=len, reverse=True)
    rows = []
    for name, value in zip(names, values):
        base = name
        for col in cat_sorted:
            if name.startswith(col + "_"):
                base = col
                break
        rows.append((base, float(value)))

    agg = pd.DataFrame(rows, columns=["feature", "weight"]).groupby("feature", as_index=False)["weight"].sum()
    total = agg["weight"].sum()
    agg["weight"] = agg["weight"] / total if total else 0.0
    agg["kind"] = kind
    return agg.sort_values("weight", ascending=False).reset_index(drop=True)


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (scores >= 0.5).astype(int)
    prec, rec, pr_thr = precision_recall_curve(y_true, scores)
    f1_curve = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best_f1 = float(f1_curve.max())
    best_at = float(pr_thr[int(np.argmax(f1_curve[:-1]))]) if len(pr_thr) else 0.5

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "best_f1": best_f1,
        "best_threshold": best_at,
        "baseline": float(y_true.mean()),
        "matrix": confusion_matrix(y_true, pred, labels=[0, 1]),
    }


@st.cache_resource(show_spinner=False)
def train(mode: str, model_name: str, use_smote: bool) -> dict:
    """Fit one model for one mode (cached across reruns)."""
    from sklearn.model_selection import train_test_split

    df = load_dataset()
    X, y = build_features(df, mode)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, stratify=y.to_numpy(), random_state=42
    )

    cap = MODELS[model_name]["max_train"]
    train_idx = _stratified_cap(y, train_idx, cap)
    subsampled = cap is not None and len(train_idx) == cap

    # The ensemble reuses the cached member fits instead of refitting five models.
    # Equal-weight averaging of member probabilities is exactly soft voting.
    if model_name == "ensemble":
        parts = [train(mode, member, use_smote) for member in ENSEMBLE_MEMBERS]
        parts = [p for p in parts if "error" not in p]
        y_test = y.iloc[test_idx].to_numpy()
        model = AveragingModel([p["model"] for p in parts])
        scores = model.predict_proba(X.iloc[test_idx])
        return {
            "model_name": model_name,
            "mode": mode,
            "label": MODELS[model_name]["label"],
            "family": MODELS[model_name]["family"],
            "note": MODELS[model_name]["note"],
            "model": model,
            "scores": scores,
            "y_test": y_test,
            "test_index": np.asarray(test_idx),
            "metrics": _metrics(y_test, scores),
            "importance": None,
            "use_smote": use_smote,
            "unsupervised": False,
            "train_size": int(len(train_idx)),
            "train_size_after_smote": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "train_fraud": int(y.iloc[train_idx].sum()),
            "test_fraud": int(y_test.sum()),
            "subsampled": False,
            "members": [p["label"] for p in parts],
            "train_span": (df.loc[train_idx, "timestamp"].min(), df.loc[train_idx, "timestamp"].max()),
            "test_span": (df.loc[test_idx, "timestamp"].min(), df.loc[test_idx, "timestamp"].max()),
        }

    pre = build_preprocessor(MODELS[model_name]["scale"])
    X_train = pre.fit_transform(X.iloc[train_idx])
    X_test = pre.transform(X.iloc[test_idx])
    y_train = y.iloc[train_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    # Unsupervised detectors learn the shape of the data, not the label, so
    # oversampling the minority class would only corrupt what they fit.
    if use_smote and model_name not in UNSUPERVISED:
        X_train, y_train = smote_balance(X_train, y_train, ratio=0.5, k=5, seed=42)

    clf = make_estimator(model_name, balanced=not use_smote)
    kind, center, spread = "prob", 0.0, 1.0

    if model_name == "autoencoder":
        normal = np.asarray(X_train)[np.asarray(y_train) == 0]
        clf.fit(normal, normal)
        recon = np.asarray(clf.predict(normal), dtype=float)
        err = np.mean((normal - recon) ** 2, axis=1)
        kind = "reconstruction"
        center = float(np.percentile(err, 95))
        spread = float(err.std()) or 1.0
    elif model_name == "isolation_forest":
        clf.fit(X_train)
        raw = -clf.decision_function(X_train)
        kind = "anomaly"
        center = float(np.percentile(raw, 95))
        spread = float(raw.std()) or 1.0
    else:
        clf.fit(X_train, y_train)

    model = FittedModel(pre, clf, kind=kind, center=center, spread=spread)
    scores = model.predict_proba(X.iloc[test_idx])
    metrics = _metrics(y_test, scores)
    importance = _aggregate_importance(pre, clf, CATEGORICAL)

    return {
        "model_name": model_name,
        "mode": mode,
        "label": MODELS[model_name]["label"],
        "family": MODELS[model_name]["family"],
        "note": MODELS[model_name]["note"],
        "model": model,
        "scores": scores,
        "y_test": y_test,
        "test_index": np.asarray(test_idx),
        "metrics": metrics,
        "importance": importance,
        "use_smote": use_smote and model_name not in UNSUPERVISED,
        "unsupervised": model_name in UNSUPERVISED,
        "train_size": int(len(train_idx)),
        "train_size_after_smote": int(len(y_train)),
        "test_size": int(len(test_idx)),
        "train_fraud": int(y.iloc[train_idx].sum()),
        "test_fraud": int(y_test.sum()),
        "subsampled": subsampled,
        "train_span": (df.loc[train_idx, "timestamp"].min(), df.loc[train_idx, "timestamp"].max()),
        "test_span": (df.loc[test_idx, "timestamp"].min(), df.loc[test_idx, "timestamp"].max()),
    }


@st.cache_resource(show_spinner=False)
def train_all(mode: str, use_smote: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in MODEL_ORDER:
        try:
            out[name] = train(mode, name, use_smote)
        except Exception as exc:  # one estimator must not break the page
            out[name] = {
                "model_name": name,
                "mode": mode,
                "label": MODELS[name]["label"],
                "family": MODELS[name]["family"],
                "error": str(exc),
            }
    return out


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alerts": int(tp + fp),
        "alert_rate": float(pred.mean()),
        "matrix": confusion_matrix(y_true, pred, labels=[0, 1]),
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        res = results.get(name, {})
        m = res.get("metrics", {})
        rows.append(
            {
                "model": MODELS[name]["label"],
                "family": MODELS[name]["family"],
                "roc_auc": m.get("roc_auc", np.nan),
                "pr_auc": m.get("pr_auc", np.nan),
                "accuracy": m.get("accuracy", np.nan),
                "precision": m.get("precision", np.nan),
                "recall": m.get("recall", np.nan),
                "f1": m.get("f1", np.nan),
                "brier": m.get("brier", np.nan),
                "status": (
                    "failed"
                    if "error" in res
                    else "unsupervised"
                    if res.get("unsupervised")
                    else "subsampled"
                    if res.get("subsampled")
                    else "ok"
                ),
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def score_frame(mode: str, model_name: str, use_smote: bool = True) -> pd.DataFrame:
    """Attach the model's risk score and the active target to every row."""
    res = train(mode, model_name, use_smote)
    base = load_dataset()
    X, _ = build_features(base, mode)
    out = base.copy()
    out["risk_score"] = res["model"].predict_proba(X)
    out["risk_band"] = pd.cut(
        out["risk_score"],
        bins=[-0.001, 0.05, 0.20, 0.50, 1.001],
        labels=["Low", "Guarded", "Elevated", "Critical"],
    ).astype(str)
    out["target"] = out[MODES[mode]["target"]].astype(int)
    out["target_label"] = out["target"].map({1: "Fraud", 0: "Legit"})
    return out


def single_score(mode: str, model_name: str, payload: dict, use_smote: bool = True) -> float:
    res = train(mode, model_name, use_smote)
    row = feature_row(payload)
    return float(res["model"].predict_proba(row)[0])
