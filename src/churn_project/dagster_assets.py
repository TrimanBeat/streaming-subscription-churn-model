from __future__ import annotations

from pathlib import Path

import dagster as dg
import pandas as pd
import subprocess

from churn_project.features import build_train_model_ready


RAW_PATH = Path("data/raw/train.csv")
PROCESSED_OUTPUT_PATH = Path("data/processed/train_model_ready.csv")


@dg.asset
def raw_data() -> pd.DataFrame:
    """Carga el dataset raw principal."""
    return pd.read_csv(RAW_PATH)


@dg.asset
def train_model_ready(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataframe procesado oficial usado por la app."""
    return build_train_model_ready(raw_data)


@dg.asset
def train_model_ready_csv(train_model_ready: pd.DataFrame) -> str:
    """Guarda train_model_ready.csv en data/processed/."""
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_model_ready.to_csv(PROCESSED_OUTPUT_PATH, index=False)
    return str(PROCESSED_OUTPUT_PATH)

@dg.asset(deps=[train_model_ready_csv])
def segment_summary_r_csv() -> str:
    """Genera un resumen por segmentos usando R."""
    input_path = Path("data/processed/train_model_ready.csv")
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