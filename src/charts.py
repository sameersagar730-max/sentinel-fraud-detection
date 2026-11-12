"""Plotly figures for the Sentinel dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import (
    ACCENT,
    ACCENT_2,
    AMBER,
    BORDER,
    DIVERGING,
    FONT,
    FRAUD,
    FRAUD_SOFT,
    LEGIT,
    MUTED,
    PALETTE,
    PANEL_ALT,
    TEXT,
    VIOLET,
)


def _style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=12),
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=MUTED),
        ),
        hoverlabel=dict(bgcolor=PANEL_ALT, bordercolor=BORDER, font=dict(color=TEXT, family=FONT)),
        bargap=0.28,
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.07)",
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.07)",
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
    )
    return fig


def empty(message: str = "No records match the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=MUTED, size=13),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=240, legend=False)


def monthly_volume_rate(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=monthly["month"],
        y=monthly["volume"],
        name="Volume",
        marker_color=ACCENT,
        marker_opacity=0.55,
        hovertemplate="%{x|%b %Y}<br>Volume $%{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=monthly["month"],
        y=monthly["rate"] * 100,
        name="Fraud rate",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=FRAUD, width=2.4),
        marker=dict(size=6, color=FRAUD, line=dict(width=1, color="#1a0f14")),
        hovertemplate="%{x|%b %Y}<br>Fraud rate %{y:.2f}%<extra></extra>",
    )
    fig.update_layout(
        yaxis=dict(title="Volume ($)", tickformat="~s"),
        yaxis2=dict(
            title="Fraud rate (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, max(monthly["rate"].max() * 100 * 1.35, 5)],
            tickfont=dict(color=FRAUD, size=11),
        ),
    )
    return _style(fig, height=330)


def daily_volume(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=daily["date"],
        y=daily["transactions"],
        mode="lines",
        line=dict(color=ACCENT, width=1.4),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.10)",
        name="Transactions",
        hovertemplate="%{x|%d %b %Y}<br>%{y} transactions<extra></extra>",
    )
    return _style(fig, height=240, legend=False)


def donut(fraud: int, legit: int) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[legit, fraud],
            hole=0.72,
            marker=dict(colors=[LEGIT, FRAUD], line=dict(color="#070D18", width=2)),
            textinfo="none",
            hovertemplate="%{label}<br>%{value:,} (%{percent})<extra></extra>",
            sort=False,
        )
    )
    fig.add_annotation(
        text=f"<b style='font-size:22px'>{fraud / max(fraud + legit, 1) * 100:.1f}%</b><br><span style='font-size:10px;color:{MUTED}'>FRAUD SHARE</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color=TEXT),
    )
    return _style(fig, height=260, legend=True)


def amount_hist(df: pd.DataFrame, log: bool = True) -> go.Figure:
    legit = df.loc[df["target"] == 0, "amount"]
    fraud = df.loc[df["target"] == 1, "amount"]
    fig = go.Figure()
    fig.add_histogram(
        x=legit,
        name="Legitimate",
        marker_color=LEGIT,
        opacity=0.55,
        nbinsx=60,
        hovertemplate="Amount $%{x:,.0f}<br>%{y} legitimate<extra></extra>",
    )
    fig.add_histogram(
        x=fraud,
        name="Fraud",
        marker_color=FRAUD,
        opacity=0.72,
        nbinsx=60,
        hovertemplate="Amount $%{x:,.0f}<br>%{y} fraud<extra></extra>",
    )
    fig.update_layout(barmode="overlay")
    if log:
        fig.update_xaxes(type="log", title="Transaction amount ($, log scale)")
    else:
        fig.update_xaxes(title="Transaction amount ($)")
    fig.update_yaxes(title="Transactions")
    return _style(fig, height=320)


def rate_bars(table: pd.DataFrame, label_col: str, n: int = 8) -> go.Figure:
    data = table.head(n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["rate"] * 100,
            y=data[label_col].astype(str),
            orientation="h",
            marker=dict(
                color=data["rate"] * 100,
                colorscale=[[0, "#134E4A"], [0.5, AMBER], [1, FRAUD]],
                showscale=False,
                line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in data["rate"] * 100],
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            customdata=np.stack([data["transactions"], data["fraud"]], axis=-1),
            hovertemplate="%{y}<br>Fraud rate %{x:.2f}%<br>%{customdata[0]:,} transactions · %{customdata[1]:,} fraud<extra></extra>",
        )
    )
    fig.update_xaxes(title="Fraud rate (%)", range=[0, max(data["rate"].max() * 100 * 1.3, 5)])
    fig.update_yaxes(title="")
    return _style(fig, height=max(240, 34 * len(data) + 40), legend=False)


def box_by_target(df: pd.DataFrame, col: str, log: bool = False) -> go.Figure:
    fig = go.Figure()
    for label, color, flag in (("Legitimate", LEGIT, 0), ("Fraud", FRAUD, 1)):
        values = df.loc[df["target"] == flag, col]
        fig.add_box(
            y=values,
            name=label,
            marker_color=color,
            line=dict(width=1.2, color=color),
            fillcolor=color,
            opacity=0.55,
            boxpoints=False,
        )
    if log:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title=col.replace("_", " ").title())
    return _style(fig, height=300)


def heatmap_corr(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=DIVERGING,
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, color=TEXT),
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(
                thickness=8,
                len=0.8,
                outlinewidth=0,
                tickfont=dict(color=MUTED, size=10),
            ),
        )
    )
    return _style(fig, height=360, legend=False)


def scatter_balance(df: pd.DataFrame, max_points: int = 3500) -> go.Figure:
    sample = df.sample(min(max_points, len(df)), random_state=42) if len(df) > max_points else df
    fig = go.Figure()
    for label, color, flag in (("Legitimate", LEGIT, 0), ("Fraud", FRAUD, 1)):
        part = sample[sample["target"] == flag]
        fig.add_scatter(
            x=part["amount"],
            y=part["balance"],
            mode="markers",
            name=label,
            marker=dict(color=color, size=4, opacity=0.55, line=dict(width=0)),
            hovertemplate="Amount $%{x:,.2f}<br>Balance $%{y:,.0f}<extra></extra>",
        )
    fig.update_xaxes(type="log", title="Transaction amount ($, log)")
    fig.update_yaxes(title="Account balance ($)")
    return _style(fig, height=330)


def roc_curves(results: dict) -> go.Figure:
    from sklearn.metrics import roc_curve

    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color="#334155", width=1, dash="dash"),
        name="Chance",
        hoverinfo="skip",
    )
    for i, (name, res) in enumerate(results.items()):
        if "error" in res:
            continue
        fpr, tpr, _ = roc_curve(res["y_test"], res["scores"])
        auc = res["metrics"]["roc_auc"]
        fig.add_scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{res['label']} · {auc:.3f}",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
        )
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    return _style(fig, height=380)


def pr_curves(results: dict) -> go.Figure:
    from sklearn.metrics import precision_recall_curve

    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        if "error" in res:
            continue
        precision, recall, _ = precision_recall_curve(res["y_test"], res["scores"])
        ap = res["metrics"]["pr_auc"]
        fig.add_scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"{res['label']} · {ap:.3f}",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
        )
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _style(fig, height=380)


def confusion(matrix: np.ndarray) -> go.Figure:
    labels = ["Legitimate", "Fraud"]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            colorscale=[[0, "#0E1727"], [0.5, "#164E63"], [1, ACCENT]],
            text=[[f"{v:,}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=18, color=TEXT),
            hovertemplate="%{y}<br>%{x}<br>%{z:,} records<extra></extra>",
            showscale=False,
        )
    )
    return _style(fig, height=300, legend=False)


def importance(imp: pd.DataFrame, n: int = 11) -> go.Figure:
    data = imp.head(n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["weight"] * 100,
            y=data["feature"].str.replace("_", " ").str.title(),
            orientation="h",
            marker=dict(
                color=data["weight"],
                colorscale=[[0, "#1E3A8A"], [1, ACCENT]],
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in data["weight"] * 100],
            textposition="outside",
            textfont=dict(color=MUTED, size=10),
            hovertemplate="%{y}<br>%{x:.2f}% of total importance<extra></extra>",
        )
    )
    fig.update_xaxes(title="Relative importance (%)")
    fig.update_yaxes(title="")
    return _style(fig, height=max(260, 30 * len(data) + 40), legend=False)


def threshold_curve(y_true: np.ndarray, scores: np.ndarray) -> go.Figure:
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.linspace(0.05, 0.95, 46)
    precision, recall, f1 = [], [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        precision.append(precision_score(y_true, pred, zero_division=0))
        recall.append(recall_score(y_true, pred, zero_division=0))
        f1.append(f1_score(y_true, pred, zero_division=0))

    fig = go.Figure()
    for values, name, color in (
        (precision, "Precision", ACCENT),
        (recall, "Recall", LEGIT),
        (f1, "F1", AMBER),
    ):
        fig.add_scatter(
            x=thresholds,
            y=values,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.2),
            hovertemplate=f"{name} %{{y:.3f}} at threshold %{{x:.2f}}<extra></extra>",
        )
    fig.update_xaxes(title="Decision threshold")
    fig.update_yaxes(title="Score", range=[0, 1.02])
    return _style(fig, height=320)


def score_hist(scores: np.ndarray, y_true: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(
        x=scores[y_true == 0],
        name="Legitimate",
        marker_color=LEGIT,
        opacity=0.55,
        nbinsx=45,
    )
    fig.add_histogram(
        x=scores[y_true == 1],
        name="Fraud",
        marker_color=FRAUD,
        opacity=0.75,
        nbinsx=45,
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Predicted risk score", range=[0, 1])
    fig.update_yaxes(title="Records")
    return _style(fig, height=300)


def gauge(score: float) -> go.Figure:
    pct = score * 100
    color = LEGIT if pct < 20 else (AMBER if pct < 55 else FRAUD)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number=dict(suffix="%", font=dict(size=44, color=TEXT, family=FONT)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=BORDER, tickfont=dict(color=MUTED, size=10)),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 20], color="rgba(52,211,153,0.10)"),
                    dict(range=[20, 55], color="rgba(251,191,36,0.10)"),
                    dict(range=[55, 100], color="rgba(248,113,113,0.10)"),
                ],
                threshold=dict(line=dict(color=FRAUD_SOFT, width=2), thickness=0.75, value=55),
            ),
        )
    )
    return _style(fig, height=240, legend=False)


def lift_bars(lift: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[f"D{int(d)}" for d in lift["decile"]],
            y=lift["lift"],
            marker=dict(
                color=lift["lift"],
                colorscale=[[0, "#1E3A8A"], [0.5, ACCENT], [1, FRAUD]],
                showscale=False,
            ),
            text=[f"{v:.1f}×" for v in lift["lift"]],
            textposition="outside",
            textfont=dict(color=MUTED, size=10),
            hovertemplate="Decile %{x}<br>Lift %{y:.2f}× · fraud rate %{customdata:.2%}<extra></extra>",
            customdata=lift["rate"],
        )
    )
    fig.add_hline(y=1.0, line=dict(color="#475569", width=1, dash="dash"))
    fig.update_xaxes(title="Risk decile (D10 = highest risk)")
    fig.update_yaxes(title="Lift over base rate")
    return _style(fig, height=300, legend=False)


def weekday_bars(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=table["weekday"].astype(str).str[:3],
            y=table["rate"] * 100,
            marker=dict(
                color=table["rate"] * 100,
                colorscale=[[0, "#134E4A"], [1, FRAUD]],
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in table["rate"] * 100],
            textposition="outside",
            textfont=dict(color=MUTED, size=10),
            hovertemplate="%{x}<br>Fraud rate %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_yaxes(title="Fraud rate (%)", range=[0, max(table["rate"].max() * 100 * 1.3, 5)])
    return _style(fig, height=270, legend=False)
