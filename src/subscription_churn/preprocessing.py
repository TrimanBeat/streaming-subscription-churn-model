import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "signup_date" in df.columns:
        df["tenure_days"] = -df["signup_date"]

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[18, 25, 35, 50, 65, 80],
            labels=["18-24", "25-34", "35-49", "50-64", "65-79"],
            include_lowest=True
        )

    if "weekly_hours" in df.columns:
        df["weekly_hours_bin"] = pd.cut(
            df["weekly_hours"],
            bins=[0, 5, 10, 20, 30, 40, 50],
            labels=["0-5", "5-10", "10-20", "20-30", "30-40", "40-50"],
            include_lowest=True
        )

    if "song_skip_rate" in df.columns:
        df["skip_rate_bin"] = pd.cut(
            df["song_skip_rate"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
            include_lowest=True
        )

    return df