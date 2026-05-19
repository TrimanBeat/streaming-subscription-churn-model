from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 2,
    "max_features": None,
    "class_weight": None,
    "random_state": 42,
    "n_jobs": -1,
}


def build_rf_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_candidates = [
        "location",
        "subscription_type",
        "payment_plan",
        "payment_method",
        "customer_service_inquiries",
        "age_group",
        "weekly_hours_bin",
        "skip_rate_bin",
        "state_code",
        "region",
    ]

    categorical_features = [col for col in categorical_candidates if col in X.columns]

    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

def train_rf_final_model(
    df: pd.DataFrame,
    target: str = "churned",
    test_size: float = 0.2,
    random_state: int = 42,
    rf_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    working = df.copy()

    drop_cols = [
        col
        for col in ["y_true", "y_pred", "p_churn", "customer_id"]
        if col in working.columns
    ]
    if drop_cols:
        working = working.drop(columns=drop_cols)

    X = working.drop(columns=[target])
    y = working[target]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_rf_preprocessor(X_train)
    params = DEFAULT_RF_PARAMS.copy()
    if rf_params:
        params.update(rf_params)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**params))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    metrics_df = pd.DataFrame([{
        "model": "RF Tuned",
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred, zero_division=0),
        "recall": recall_score(y_valid, y_pred, zero_division=0),
        "f1": f1_score(y_valid, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_valid, y_proba),
    }])

    cm = confusion_matrix(y_valid, y_pred, normalize="true") * 100
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"]
    )

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    feature_importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    feature_importance_df["feature_clean"] = (
        feature_importance_df["feature"]
        .astype(str)
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    valid_predictions_df = X_valid.copy()
    valid_predictions_df["y_true"] = y_valid.values
    valid_predictions_df["y_pred"] = y_pred
    valid_predictions_df["p_churn"] = y_proba

    return {
        "model": model,
        "metrics_df": metrics_df,
        "cm_df": cm_df,
        "feature_importance_df": feature_importance_df,
        "valid_predictions_df": valid_predictions_df,
    }


def export_rf_training_outputs(
    bundle: dict[str, Any],
    model_path: str | Path = "models/rf_tuned_model_compressed.joblib",
    metrics_path: str | Path = "data/exports/rf_tuned_metrics.csv",
    cm_path: str | Path = "data/exports/rf_tuned_confusion_matrix_percentage.csv",
    feature_importance_path: str | Path = "data/exports/rf_tuned_feature_importance.csv",
    valid_predictions_path: str | Path = "data/exports/rf_tuned_validation_predictions.csv",
) -> dict[str, str]:
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    cm_path = Path(cm_path)
    feature_importance_path = Path(feature_importance_path)
    valid_predictions_path = Path(valid_predictions_path)

    for path in [model_path, metrics_path, cm_path, feature_importance_path, valid_predictions_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle["model"], model_path, compress=3)
    bundle["metrics_df"].to_csv(metrics_path, index=False)
    bundle["cm_df"].to_csv(cm_path)
    bundle["feature_importance_df"].to_csv(feature_importance_path, index=False)
    bundle["valid_predictions_df"].to_csv(valid_predictions_path, index=False)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "cm_path": str(cm_path),
        "feature_importance_path": str(feature_importance_path),
        "valid_predictions_path": str(valid_predictions_path),
    }
