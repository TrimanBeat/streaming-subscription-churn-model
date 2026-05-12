from __future__ import annotations

from pathlib import Path
import subprocess

import dagster as dg
import pandas as pd

from churn_project.features import build_train_model_ready, align_prediction_ready_to_training_schema
from churn_project.rf_training_utils import export_rf_training_outputs, train_rf_final_model


RAW_TRAIN_PATH = Path("data/raw/train.csv")
RAW_INCOMING_PATH = Path("data/raw/incoming_customers.csv")

TRAIN_MODEL_READY_OUTPUT_PATH = Path("data/processed/train_model_ready.csv")
TRAIN_MODEL_READY_BASE_PATH = Path("data/processed/train_model_ready_base.csv")
RETRAINING_POOL_PATH = Path("data/processed/retraining_pool.csv")
INCOMING_PREDICTION_READY_PATH = Path("data/processed/incoming_prediction_ready.csv")

NEW_CUSTOMERS_FOR_TRAINING_PATH = Path("data/new_data/new_customers_for_training.csv")


@dg.asset
def raw_data() -> pd.DataFrame:
    """Carga el dataset raw principal de entrenamiento."""
    return pd.read_csv(RAW_TRAIN_PATH)


@dg.asset
def train_model_ready(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataframe procesado completo a partir del train raw."""
    return build_train_model_ready(raw_data)


@dg.asset
def train_model_ready_csv(train_model_ready: pd.DataFrame) -> str:
    """Guarda train_model_ready.csv en data/processed/."""
    TRAIN_MODEL_READY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_model_ready.to_csv(TRAIN_MODEL_READY_OUTPUT_PATH, index=False)
    return str(TRAIN_MODEL_READY_OUTPUT_PATH)


@dg.asset
def train_model_ready_base() -> pd.DataFrame:
    """Carga la base de entrenamiento reducida reservando un pool etiquetado aparte."""
    return pd.read_csv(TRAIN_MODEL_READY_BASE_PATH)


@dg.asset
def retraining_pool() -> pd.DataFrame:
    """Carga el pool etiquetado reservado para simular nuevos datos de entrenamiento."""
    return pd.read_csv(RETRAINING_POOL_PATH)


@dg.asset
def raw_incoming_data() -> pd.DataFrame:
    """Carga el pool de clientes entrantes raw."""
    return pd.read_csv(RAW_INCOMING_PATH)


@dg.asset
def incoming_prediction_ready(
    raw_incoming_data: pd.DataFrame,
    train_model_ready: pd.DataFrame,
) -> pd.DataFrame:
    """Pasa incoming_customers por el mismo pipeline de preparación y lo alinea al schema del train."""
    prepared_incoming = build_train_model_ready(raw_incoming_data)
    aligned_incoming = align_prediction_ready_to_training_schema(
        prepared_incoming,
        train_model_ready,
        target="churned",
    )
    return aligned_incoming


@dg.asset
def incoming_prediction_ready_csv(incoming_prediction_ready: pd.DataFrame) -> str:
    """Guarda incoming_prediction_ready.csv en data/processed/."""
    INCOMING_PREDICTION_READY_PATH.parent.mkdir(parents=True, exist_ok=True)
    incoming_prediction_ready.to_csv(INCOMING_PREDICTION_READY_PATH, index=False)
    return str(INCOMING_PREDICTION_READY_PATH)


@dg.asset(deps=[train_model_ready_csv])
def segment_summary_r_csv() -> str:
    """Genera un resumen por segmentos usando R a partir del dataset train preparado completo."""
    input_path = TRAIN_MODEL_READY_OUTPUT_PATH
    output_path = Path("data/exports/segment_summary_r.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "Rscript",
            "scripts/segment_summary.R",
            str(input_path),
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error al ejecutar el script de R:\n{result.stderr}")

    return str(output_path)


@dg.asset
def new_customers_for_training() -> pd.DataFrame:
    """Carga el lote seleccionado desde la app para incorporarlo al entrenamiento.
    Si no existe o está vacío, devuelve un DataFrame vacío.
    """
    if NEW_CUSTOMERS_FOR_TRAINING_PATH.exists():
        try:
            return pd.read_csv(NEW_CUSTOMERS_FOR_TRAINING_PATH)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


@dg.asset
def combined_training_data(
    train_model_ready_base: pd.DataFrame,
    new_customers_for_training: pd.DataFrame,
) -> pd.DataFrame:
    """Combina la base de entrenamiento reducida con el lote nuevo seleccionado para reentrenamiento."""
    base_df = train_model_ready_base.copy()

    if new_customers_for_training.empty:
        return base_df

    missing_cols = [col for col in base_df.columns if col not in new_customers_for_training.columns]
    if missing_cols:
        raise ValueError(
            "El archivo data/new_data/new_customers_for_training.csv no tiene todas las columnas necesarias. "
            f"Faltan: {missing_cols}"
        )

    ordered_new = new_customers_for_training[base_df.columns].copy()
    combined_df = pd.concat([base_df, ordered_new], ignore_index=True)
    return combined_df


@dg.asset(deps=[combined_training_data])
def rf_retraining_outputs(combined_training_data: pd.DataFrame) -> dict:
    """Reentrena el Random Forest final y actualiza los artefactos que consume la app."""
    bundle = train_rf_final_model(combined_training_data, target="churned")
    return export_rf_training_outputs(bundle)
