import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.title("🧪 Simulación en vivo")

@st.cache_resource
def load_model():
    return joblib.load("models/rf_model.joblib")

@st.cache_data
def load_simulation_base():
    return pd.read_csv("data/exports/rf_simulation_base.csv")

rf_model = load_model()
simulation_base = load_simulation_base()

if "history" not in st.session_state:
    st.session_state["history"] = []

if "simulated_customers" not in st.session_state:
    st.session_state["simulated_customers"] = pd.DataFrame()

st.subheader("Añadir cliente desde base de validación")

if st.button("Simular nuevo cliente"):
    sampled = simulation_base.sample(1).copy()

    sampled["timestamp_entry"] = datetime.now()
    sampled["source"] = "validation_sample"

    if st.session_state["simulated_customers"].empty:
        st.session_state["simulated_customers"] = sampled
    else:
        st.session_state["simulated_customers"] = pd.concat(
            [st.session_state["simulated_customers"], sampled],
            ignore_index=True
        )

    n_active = len(st.session_state["simulated_customers"])

    st.session_state["history"].append({
        "timestamp": datetime.now(),
        "n_active": n_active
    })

st.subheader("KPIs")

n_active = len(st.session_state["simulated_customers"])

col1, col2 = st.columns(2)

with col1:
    st.metric("Clientes simulados activos", n_active)

with col2:
    if n_active > 0 and "p_churn" in st.session_state["simulated_customers"].columns:
        avg_risk = st.session_state["simulated_customers"]["p_churn"].mean()
        st.metric("Riesgo medio esperado", f"{avg_risk:.2%}")
    else:
        st.metric("Riesgo medio esperado", "N/A")

st.subheader("Historial de clientes simulados")
if not st.session_state["simulated_customers"].empty:
    st.dataframe(st.session_state["simulated_customers"], use_container_width=True)
else:
    st.write("Todavía no se han simulado clientes.")

st.subheader("Evolución de clientes activos")
if len(st.session_state["history"]) > 0:
    history_df = pd.DataFrame(st.session_state["history"])
    st.line_chart(history_df.set_index("timestamp")["n_active"])
else:
    st.write("Aún no hay historial de simulación.")