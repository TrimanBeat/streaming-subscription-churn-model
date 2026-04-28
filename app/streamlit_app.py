import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dashboard de Predicción del Churn",
    layout="wide"
)

@st.cache_data
def load_main_data():
    model_metrics = pd.read_csv("data/exports/model_metrics.csv")
    train_model_ready = pd.read_csv("data/processed/train_model_ready.csv")
    return model_metrics, train_model_ready

model_metrics, train_model_ready = load_main_data()

n_customers = len(train_model_ready)
global_churn_rate = train_model_ready["churned"].mean() if "churned" in train_model_ready.columns else 0

best_model_name = "N/A"
best_model_auc = None

if "model" in model_metrics.columns and "roc_auc" in model_metrics.columns:
    best_row = model_metrics.sort_values("roc_auc", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model_auc = best_row["roc_auc"]

st.subheader("Dashboard de churn de subscripción a la plataforma")
st.markdown("Dashboard interactivo para explorar churn, comparar modelos y simular nuevos clientes.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Clientes", f"{n_customers:,}")

with col2:
    st.metric("Tasa global de churn", f"{global_churn_rate:.2%}")

with col3:
    st.metric("Mejor modelo", best_model_name)

with col4:
    st.metric("Mejor ROC AUC", f"{best_model_auc:.3f}" if best_model_auc is not None else "N/A")

st.markdown("---")
st.info("Usa el menú lateral para navegar entre EDA y Segmentos, Modelos y Simulación.")