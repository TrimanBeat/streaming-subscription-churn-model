from __future__ import annotations

import numpy as np
import pandas as pd


STATE_NAME_TO_CODE = {
    "Alabama": "AL",
    "California": "CA",
    "Florida": "FL",
    "Georgia": "GA",
    "Idaho": "ID",
    "Maine": "ME",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nebrasksa": "NE",  # typo presente en algunos datos
    "New Jersey": "NJ",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "South Carolina": "SC",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
}

STATE_TO_REGION = {
    "AL": "South",
    "CA": "West",
    "FL": "South",
    "GA": "South",
    "ID": "West",
    "ME": "Northeast",
    "MT": "West",
    "NE": "Midwest",
    "NJ": "Northeast",
    "NY": "Northeast",
    "NC": "South",
    "ND": "Midwest",
    "SC": "South",
    "UT": "West",
    "VT": "Northeast",
    "VA": "South",
    "WA": "West",
    "WV": "South",
    "WI": "Midwest",
}


def clean_streaming_churn_df(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza y variables derivadas base del proyecto."""
    df = df.copy()

    if "signup_date" in df.columns:
        df["tenure_days"] = -df["signup_date"]
        df["tenure_months"] = df["tenure_days"] / 30

    if {"weekly_songs_played", "weekly_hours"}.issubset(df.columns):
        weekly_hours_safe = df["weekly_hours"].replace(0, np.nan)
        df["songs_per_hour"] = df["weekly_songs_played"] / weekly_hours_safe
        df["songs_per_hour"] = df["songs_per_hour"].replace([np.inf, -np.inf], np.nan)

    if "song_skip_rate" in df.columns:
        df["high_skip_user"] = (df["song_skip_rate"] >= 0.7).astype(int)

    return df


def add_model_ready_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade exactamente las variables derivadas que aparecen en train_model_ready.csv."""
    df = df.copy()

    # Corregir typo en location antes de mapear
    if "location" in df.columns:
        df["location"] = df["location"].replace({"Nebrasksa": "Nebraska"})

    # age_group
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[18, 25, 35, 50, 65, 80],
            labels=["18-24", "25-34", "35-49", "50-64", "65-79"],
            include_lowest=True,
        )

    # weekly_hours_bin
    if "weekly_hours" in df.columns:
        df["weekly_hours_bin"] = pd.cut(
            df["weekly_hours"],
            bins=[0, 5, 10, 20, 30, 40, 50],
            labels=["0-5", "5-10", "10-20", "20-30", "30-40", "40-50"],
            right=False,
            include_lowest=True,
        )

    # skip_rate_bin
    if "song_skip_rate" in df.columns:
        df["skip_rate_bin"] = pd.cut(
            df["song_skip_rate"],
            bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001],
            labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
            right=False,
            include_lowest=True,
        )

    # state_code / region
    if "location" in df.columns:
        location_clean = df["location"].astype(str).str.strip()
        df["state_code"] = location_clean.map(STATE_NAME_TO_CODE)
        df["region"] = df["state_code"].map(STATE_TO_REGION)

    return df


def build_train_model_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataframe oficial train_model_ready.csv usado por la app."""
    df = clean_streaming_churn_df(df)
    df = add_model_ready_features(df)
    return df
