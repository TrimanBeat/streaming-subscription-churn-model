from __future__ import annotations

import pandas as pd


def get_segment_options(df: pd.DataFrame) -> list[str]:
    candidates = [
        "subscription_type",
        "customer_service_inquiries",
        "region",
        "risk_segment_rule",
        "age_group",
        "state_code",
    ]
    return [col for col in candidates if col in df.columns]


def build_segment_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Construye un resumen por segmento ordenado por churn descendente."""
    agg_map = {"churned": "mean"}

    optional_metrics = {
        "weekly_hours": "mean",
        "song_skip_rate": "mean",
        "num_subscription_pauses": "mean",
        "engagement_index": "mean",
    }

    use_cols = [group_col, "churned"] + [c for c in optional_metrics if c in df.columns]
    working = df[use_cols].copy()

    for col, func in optional_metrics.items():
        if col in working.columns:
            agg_map[col] = func

    summary = (
        working
        .groupby(group_col, dropna=False)
        .agg(agg_map)
        .reset_index()
        .rename(columns={
            group_col: "segmento",
            "churned": "churn_rate",
            "weekly_hours": "weekly_hours_media",
            "song_skip_rate": "skip_rate_media",
            "num_subscription_pauses": "pauses_media",
            "engagement_index": "engagement_media",
        })
        .sort_values("churn_rate", ascending=False)
    )

    counts = (
        working
        .groupby(group_col, dropna=False)
        .size()
        .reset_index(name="n_clientes")
        .rename(columns={group_col: "segmento"})
    )

    summary = summary.merge(counts, on="segmento", how="left")

    ordered_cols = [
        "segmento",
        "n_clientes",
        "churn_rate",
        "weekly_hours_media",
        "skip_rate_media",
        "pauses_media",
        "engagement_media",
    ]
    ordered_cols = [c for c in ordered_cols if c in summary.columns]
    summary = summary[ordered_cols]

    for col in summary.columns:
        if col != "segmento":
            summary[col] = summary[col].round(3)

    return summary


def get_numeric_plot_candidates(df: pd.DataFrame) -> list[str]:
    candidates = [
        "weekly_hours",
        "song_skip_rate",
        "num_subscription_pauses",
        "age",
        "average_session_length",
        "weekly_unique_songs",
        "weekly_songs_played",
        "engagement_index",
    ]
    return [col for col in candidates if col in df.columns]


def build_churn_rate_by_segment(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)["churned"]
        .mean()
        .reset_index(name="churn_rate")
        .sort_values("churn_rate", ascending=False)
    )
    out["churn_rate"] = out["churn_rate"].round(3)
    return out


def build_numeric_boxplot_df(
    df: pd.DataFrame,
    metric_col: str,
    churn_col: str = "churned",
) -> pd.DataFrame:
    """Devuelve un dataframe simple para boxplots churn vs no churn."""
    if metric_col not in df.columns or churn_col not in df.columns:
        return pd.DataFrame()

    plot_df = df[[metric_col, churn_col]].copy()
    plot_df["grupo"] = plot_df[churn_col].map({0: "No churn", 1: "Churn"})
    return plot_df


def metric_columns_for_display(df: pd.DataFrame, preferred_cols: list[str]) -> list[str]:
    return [col for col in preferred_cols if col in df.columns]
