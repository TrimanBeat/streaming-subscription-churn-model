from __future__ import annotations

import numpy as np
import dask.dataframe as dd
import pandas as pd


def add_features_dask(ddf: dd.DataFrame) -> dd.DataFrame:
    """Añade variables derivadas sencillas sobre un DataFrame de Dask."""
    cols = set(ddf.columns)

    if {"weekly_hours", "song_skip_rate"}.issubset(cols):
        ddf["engagement_index"] = ddf["weekly_hours"] * (1 - ddf["song_skip_rate"])

    if "subscription_type" in cols:
        ddf["is_free_plan"] = (ddf["subscription_type"] == "Free").astype("int64")

    if "customer_service_inquiries" in cols:
        ddf["is_high_inquiries"] = (ddf["customer_service_inquiries"] == "High").astype("int64")

    if {"subscription_type", "customer_service_inquiries"}.issubset(cols):
        ddf["risk_segment_rule"] = "Lower Risk Segment"
        ddf["risk_segment_rule"] = ddf["risk_segment_rule"].mask(
            ddf["customer_service_inquiries"] == "High",
            "Medium Risk Segment"
        )
        ddf["risk_segment_rule"] = ddf["risk_segment_rule"].mask(
            ddf["subscription_type"] == "Free",
            "High Risk Segment"
        )
        ddf["risk_segment_rule"] = ddf["risk_segment_rule"].mask(
            (ddf["subscription_type"] == "Free") & (ddf["customer_service_inquiries"] == "High"),
            "Very High Risk Segment"
        )

    return ddf


def add_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Versión pandas de las transformaciones, útil si ya tienes el DataFrame materializado."""
    df = df.copy()
    cols = set(df.columns)

    if {"weekly_hours", "song_skip_rate"}.issubset(cols):
        df["engagement_index"] = df["weekly_hours"] * (1 - df["song_skip_rate"])

    if "subscription_type" in cols:
        df["is_free_plan"] = (df["subscription_type"] == "Free").astype("int64")

    if "customer_service_inquiries" in cols:
        df["is_high_inquiries"] = (df["customer_service_inquiries"] == "High").astype("int64")

    if {"subscription_type", "customer_service_inquiries"}.issubset(cols):
        conditions = [
            (df["subscription_type"] == "Free") & (df["customer_service_inquiries"] == "High"),
            (df["subscription_type"] == "Free"),
            (df["customer_service_inquiries"] == "High"),
        ]
        choices = ["Very High Risk Segment", "High Risk Segment", "Medium Risk Segment"]
        df["risk_segment_rule"] = np.select(conditions, choices, default="Lower Risk Segment")

    return df
