from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import streamlit as st

from .features import add_features_dask


DEFAULT_RAW_PATH = Path("data/raw/train.csv")
DEFAULT_PROCESSED_PATH = Path("data/processed/train_model_ready.csv")


@st.cache_data
def load_data_with_dask(data_path: str | Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Carga el dataset principal con Dask, aplica transformaciones y devuelve pandas."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

    ddf = dd.read_csv(data_path)
    ddf = add_features_dask(ddf)
    return ddf.compute()


@st.cache_data
def load_processed_data(data_path: str | Path = DEFAULT_PROCESSED_PATH) -> pd.DataFrame:
    """Carga un CSV pequeño/mediano ya procesado con pandas."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")
    return pd.read_csv(data_path)


@st.cache_data
def load_export_csv(path: str | Path, index_col: int | str | None = None) -> pd.DataFrame:
    """Carga un CSV de exports con pandas."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_csv(path, index_col=index_col)
