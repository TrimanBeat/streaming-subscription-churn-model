from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "train_model_ready.csv"
BASE_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "train_model_ready_base.csv"
POOL_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "retraining_pool.csv"

POOL_SIZE = 1000
RANDOM_STATE = 42
TARGET = "churned"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"No se encontró la columna target '{TARGET}' en el dataset.")

    if len(df) <= POOL_SIZE:
        raise ValueError(
            f"El dataset tiene {len(df)} filas y no permite separar un pool de {POOL_SIZE}."
        )

    base_df, pool_df = train_test_split(
        df,
        test_size=POOL_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )

    base_df = base_df.reset_index(drop=True)
    pool_df = pool_df.reset_index(drop=True)

    BASE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(BASE_OUTPUT_PATH, index=False)
    pool_df.to_csv(POOL_OUTPUT_PATH, index=False)

    print("Archivos creados correctamente:")
    print(f"- Base: {BASE_OUTPUT_PATH} | filas: {len(base_df)}")
    print(f"- Pool: {POOL_OUTPUT_PATH} | filas: {len(pool_df)}")
    print("\\nDistribución de churn:")
    print("Base:")
    print(base_df[TARGET].value_counts(normalize=True, dropna=False))
    print("\\nPool:")
    print(pool_df[TARGET].value_counts(normalize=True, dropna=False))


if __name__ == "__main__":
    main()