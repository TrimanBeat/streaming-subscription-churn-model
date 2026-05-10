from __future__ import annotations

import numpy as np
import pandas as pd


def classify_risk(p: float) -> str:
    if p < 0.33:
        return "Bajo"
    if p < 0.66:
        return "Medio"
    return "Alto"


def risk_color(label: str) -> str:
    mapping = {
        "Bajo": "#2E8B57",
        "Medio": "#E6C200",
        "Alto": "#C0392B",
    }
    return mapping.get(label, "#94A3B8")


def filter_by_risk_level(df: pd.DataFrame, risk_level: str, risk_col: str = "risk_level") -> pd.DataFrame:
    """Filtra por nivel de riesgo. Si se selecciona 'Todos', devuelve todo."""
    if risk_level == "Todos":
        return df.copy()
    if risk_col not in df.columns:
        return df.copy()
    return df[df[risk_col] == risk_level].copy()


def build_manual_input(
    feature_df: pd.DataFrame,
    form_values: dict,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """Construye una fila completa para predicción a partir de los campos del formulario."""
    row = {}

    for col in feature_df.columns:
        if col in form_values:
            row[col] = form_values[col]
        else:
            if col in numeric_cols:
                row[col] = float(feature_df[col].median()) if feature_df[col].notna().any() else 0.0
            else:
                mode_vals = feature_df[col].mode(dropna=True)
                row[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"

    return pd.DataFrame([row])


def get_random_customer(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un cliente aleatorio del dataset ya preparado para predecir."""
    return feature_df.sample(1, random_state=np.random.randint(0, 10_000)).copy()


def predict_proba_single(model, input_df: pd.DataFrame) -> float:
    """Devuelve la probabilidad de la clase positiva para un único registro."""
    return float(model.predict_proba(input_df)[:, 1][0])


def add_simulation_record(
    current_df: pd.DataFrame,
    source_name: str,
    model_name: str,
    input_df: pd.DataFrame,
    p_churn: float,
) -> pd.DataFrame:
    """Añade una simulación al histórico y devuelve el DataFrame actualizado."""
    record = input_df.copy()
    record["source"] = source_name
    record["model_used"] = model_name
    record["p_churn"] = p_churn
    record["risk_level"] = classify_risk(p_churn)

    return pd.concat([current_df, record], ignore_index=True)
