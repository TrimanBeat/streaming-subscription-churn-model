import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(model, X_valid: pd.DataFrame, y_valid: pd.Series, model_name: str = "model"):
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    metrics = pd.DataFrame({
        "model": [model_name],
        "accuracy": [accuracy_score(y_valid, y_pred)],
        "precision": [precision_score(y_valid, y_pred)],
        "recall": [recall_score(y_valid, y_pred)],
        "f1": [f1_score(y_valid, y_pred)],
        "roc_auc": [roc_auc_score(y_valid, y_proba)],
    })

    preds = X_valid.copy()
    preds["y_true"] = y_valid.values
    preds["y_pred"] = y_pred
    preds["p_churn"] = y_proba

    return metrics, preds