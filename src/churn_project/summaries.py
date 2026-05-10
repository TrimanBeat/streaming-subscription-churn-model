from __future__ import annotations

import pandas as pd


def top_risk_customers(
    df: pd.DataFrame,
    top_n: int = 20,
    active_only: bool = False,
    truth_col: str = "y_true",
    proba_col: str = "p_churn",
) -> pd.DataFrame:
    """Ordena por probabilidad de churn y devuelve los top_n."""
    working = df.copy()
    if active_only and truth_col in working.columns:
        working = working[working[truth_col] == 0].copy()

    if proba_col in working.columns:
        working = working.sort_values(proba_col, ascending=False)

    return working.head(top_n)


def metric_columns_for_display(df: pd.DataFrame, preferred_cols: list[str]) -> list[str]:
    """Devuelve solo las columnas preferidas que realmente existen."""
    return [col for col in preferred_cols if col in df.columns]
