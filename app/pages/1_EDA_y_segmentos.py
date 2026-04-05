import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 EDA y Segmentos")

@st.cache_data
def load_data():
    segment_summary = pd.read_csv("data/exports/segment_summary.csv")
    train_model_ready = pd.read_csv("data/processed/train_model_ready.csv")
    return segment_summary, train_model_ready

segment_summary, train_model_ready = load_data()

st.subheader("Tabla de segmentos")
st.dataframe(segment_summary, use_container_width=True)

st.subheader("Churn rate por segmento")

fig = px.bar(
    segment_summary.sort_values("churn_rate", ascending=False),
    x="subscription_type",
    y="churn_rate",
    color="customer_service_inquiries",
    barmode="group",
    title="Churn rate por tipo de suscripción e incidencias"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Vista previa del dataset preparado")
st.dataframe(train_model_ready.head(20), use_container_width=True)