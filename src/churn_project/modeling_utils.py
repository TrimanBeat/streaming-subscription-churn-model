from __future__ import annotations

import ast
from typing import Literal

import pandas as pd

from .simulation_utils import filter_by_risk_level


ModelVisualType = Literal["coefficients", "importance", "dnn"]


def get_dnn_config(best_params_df: pd.DataFrame) -> dict:
    """Extrae la configuración ganadora de la DNN desde el CSV de mejores parámetros."""
    dnn_row = best_params_df[best_params_df["model"] == "DNN Tuned"].iloc[0]

    hidden_units_raw = dnn_row["classifier__model__hidden_units"]
    if isinstance(hidden_units_raw, str):
        hidden_units = ast.literal_eval(hidden_units_raw)
    else:
        hidden_units = hidden_units_raw

    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,)
    elif isinstance(hidden_units, list):
        hidden_units = tuple(hidden_units)

    return {
        "hidden_units": hidden_units,
        "dropout_rate": float(dnn_row["classifier__model__dropout_rate"]),
        "learning_rate": float(dnn_row["classifier__model__learning_rate"]),
        "batch_size": int(dnn_row["classifier__batch_size"]),
        "epochs": int(dnn_row["classifier__epochs"]),
    }


def clean_feature_names(series: pd.Series) -> pd.Series:
    """Limpia prefijos y guiones bajos de nombres de variables para visualización."""
    return (
        series.astype(str)
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )


def select_model_bundle(
    selected_model: str,
    logreg_tuned_metrics: pd.DataFrame,
    logreg_tuned_preds: pd.DataFrame,
    logreg_tuned_cm_pct: pd.DataFrame,
    rf_tuned_metrics: pd.DataFrame,
    rf_tuned_preds: pd.DataFrame,
    rf_tuned_cm_pct: pd.DataFrame,
    dnn_tuned_metrics: pd.DataFrame,
    dnn_tuned_preds: pd.DataFrame,
    dnn_tuned_cm_pct: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, ModelVisualType]:
    """Devuelve métricas, predicciones, matriz y tipo de visualización según modelo seleccionado."""
    if selected_model == "Logistic Regression":
        return (
            logreg_tuned_metrics.iloc[0],
            logreg_tuned_preds,
            logreg_tuned_cm_pct,
            "coefficients",
        )

    if selected_model == "Random Forest":
        return (
            rf_tuned_metrics.iloc[0],
            rf_tuned_preds,
            rf_tuned_cm_pct,
            "importance",
        )

    return (
        dnn_tuned_metrics.iloc[0],
        dnn_tuned_preds,
        dnn_tuned_cm_pct,
        "dnn",
    )


def filter_risk_table(
    preds_df: pd.DataFrame,
    risk_level: str = "Todos",
    active_only: bool = True,
    truth_col: str = "y_true",
    risk_col: str = "risk_level",
    proba_col: str = "p_churn",
) -> pd.DataFrame:
    """Filtra y ordena una tabla de riesgo por nivel de riesgo y opcionalmente por clientes activos."""
    working = preds_df.copy()

    if active_only and truth_col in working.columns:
        working = working[working[truth_col] == 0].copy()

    if risk_col not in working.columns and proba_col in working.columns:
        def classify(p):
            if p < 0.33:
                return "Bajo"
            if p < 0.66:
                return "Medio"
            return "Alto"
        working[risk_col] = working[proba_col].apply(classify)

    working = filter_by_risk_level(working, risk_level, risk_col=risk_col)

    if proba_col in working.columns:
        working = working.sort_values(proba_col, ascending=False)

    return working


def build_dnn_architecture_table(dnn_config: dict) -> pd.DataFrame:
    """Construye una tabla sencilla para representar la arquitectura de la DNN."""
    arch_rows = [{"Capa": "Entrada", "Configuración": "Features preprocesadas"}]

    for i, units in enumerate(dnn_config["hidden_units"], start=1):
        arch_rows.append({
            "Capa": f"Dense {i}",
            "Configuración": f"{units} neuronas + ReLU"
        })
        arch_rows.append({
            "Capa": f"Dropout {i}",
            "Configuración": f"rate = {dnn_config['dropout_rate']}"
        })

    arch_rows.append({
        "Capa": "Salida",
        "Configuración": "1 neurona + Sigmoid"
    })

    return pd.DataFrame(arch_rows)
