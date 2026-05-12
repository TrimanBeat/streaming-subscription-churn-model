from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

INCOMING_CUSTOMERS_PATH = ROOT_DIR / "data" / "raw" / "incoming_customers.csv"
INCOMING_PREDICTION_READY_PATH = ROOT_DIR / "data" / "processed" / "incoming_prediction_ready.csv"
RETRAINING_POOL_PATH = ROOT_DIR / "data" / "processed" / "retraining_pool.csv"
NEW_CUSTOMERS_FOR_TRAINING_PATH = ROOT_DIR / "data" / "new_data" / "new_customers_for_training.csv"


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


def load_incoming_customers(path: Path | str = INCOMING_CUSTOMERS_PATH) -> pd.DataFrame:
    """Carga el pool raw de clientes entrantes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_csv(path)


def load_retraining_pool(path: Path | str = RETRAINING_POOL_PATH) -> pd.DataFrame:
    """Carga el pool etiquetado reservado para simular nuevos datos de entrenamiento."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_csv(path)


def sample_retraining_pool(
    n_customers: int,
    path: Path | str = RETRAINING_POOL_PATH,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Toma una muestra aleatoria del retraining_pool.csv."""
    df = load_retraining_pool(path)

    if len(df) == 0:
        return df.copy()

    n_customers = min(int(n_customers), len(df))
    if n_customers <= 0:
        return df.iloc[0:0].copy()

    return df.sample(n=n_customers, random_state=random_state).copy()


def save_training_batch(
    batch_df: pd.DataFrame,
    output_path: Path | str = NEW_CUSTOMERS_FOR_TRAINING_PATH,
) -> str:
    """Guarda el lote seleccionado para reentrenamiento."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_df.to_csv(output_path, index=False)
    return str(output_path)


def load_new_customers_for_training(
    path: Path | str = NEW_CUSTOMERS_FOR_TRAINING_PATH,
) -> pd.DataFrame:
    """Carga el lote actual pendiente de incorporar al pipeline."""
    path = Path(path)
    if path.exists():
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def clear_new_customers_for_training(
    path: Path | str = NEW_CUSTOMERS_FOR_TRAINING_PATH,
) -> str:
    """Vacía el archivo del lote actual para reentrenamiento."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(path, index=False)
    return str(path)


def generate_retraining_batch(
    n_customers: int,
    pool_path: Path | str = RETRAINING_POOL_PATH,
    output_path: Path | str = NEW_CUSTOMERS_FOR_TRAINING_PATH,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Genera un lote aleatorio desde retraining_pool.csv y lo guarda para Dagster."""
    batch_df = sample_retraining_pool(
        n_customers=n_customers,
        path=pool_path,
        random_state=random_state,
    )
    save_training_batch(batch_df, output_path=output_path)
    return batch_df
