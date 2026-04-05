import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🤖 Modelos Predictivos")

@st.cache_data
def load_model_outputs():
    model_metrics = pd.read_csv("data/exports/model_metrics.csv")
    rf_feature_importance = pd.read_csv("data/exports/rf_feature_importance.csv")
    rf_preds = pd.read_csv("data/exports/rf_validation_predictions.csv")
    return model_metrics, rf_feature_importance, rf_preds

model_metrics, rf_feature_importance, rf_preds = load_model_outputs()

st.subheader("Comparación de modelos")
st.dataframe(model_metrics, use_container_width=True)

metrics_long = model_metrics.melt(
    id_vars="model",
    var_name="metric",
    value_name="score"
)

fig_metrics = px.bar(
    metrics_long,
    x="metric",
    y="score",
    color="model",
    barmode="group",
    title="Comparación de métricas por modelo"
)

st.plotly_chart(fig_metrics, use_container_width=True)

st.subheader("Importancia de variables - Random Forest")

top_features = rf_feature_importance.head(15).sort_values("importance", ascending=True)

feature_col = "feature_clean" if "feature_clean" in top_features.columns else "feature"

fig_importance = px.bar(
    top_features,
    x="importance",
    y=feature_col,
    orientation="h",
    title="Top variables más importantes"
)

st.plotly_chart(fig_importance, use_container_width=True)

st.subheader("Ejemplos de predicciones")
st.dataframe(
    rf_preds.sort_values("p_churn", ascending=False).head(20),
    use_container_width=True
)