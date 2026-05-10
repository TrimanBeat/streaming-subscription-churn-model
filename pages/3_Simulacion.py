from __future__ import annotations

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from churn_project.data_loader import load_processed_data
from churn_project.simulation_utils import (
    add_simulation_record,
    build_manual_input,
    classify_risk,
    filter_by_risk_level,
    get_random_customer,
    predict_proba_single,
    risk_color,
)
from churn_project.summaries import metric_columns_for_display


st.title("Simulación de clientes")
st.caption("Simula nuevos clientes y estima su riesgo de churn con el modelo final del proyecto.")
st.markdown("---")


@st.cache_resource
def load_rf_model():
    return joblib.load("models/rf_tuned_model_compressed.joblib")


df = load_processed_data("data/processed/train_model_ready.csv")
rf_model = load_rf_model()

TARGET = "churned"

drop_cols_if_present = [col for col in ["y_true", "y_pred", "p_churn"] if col in df.columns]
available_df = df.drop(columns=drop_cols_if_present, errors="ignore").copy()

if TARGET in available_df.columns:
    feature_df = available_df.drop(columns=[TARGET]).copy()
else:
    feature_df = available_df.copy()

numeric_cols = feature_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

preferred_form_cols = [
    "age",
    "subscription_type",
    "weekly_hours",
    "song_skip_rate",
    "num_subscription_pauses",
    "customer_service_inquiries",
    "average_session_length",
    "weekly_unique_songs",
    "weekly_songs_played",
    "notifications_clicked",
]

form_cols = [c for c in preferred_form_cols if c in feature_df.columns]

if "simulated_customers" not in st.session_state:
    st.session_state["simulated_customers"] = pd.DataFrame()


st.subheader("Nuevo cliente")

tab_manual, tab_random = st.tabs(["Entrada manual", "Cliente aleatorio"])

latest_prediction = None
latest_input_df = None

with tab_manual:
    with st.form("manual_simulation_form"):
        form_values = {}

        col1, col2, col3 = st.columns(3)

        for idx, col in enumerate(form_cols):
            target_col = [col1, col2, col3][idx % 3]

            with target_col:
                if col in numeric_cols:
                    min_val = float(feature_df[col].min()) if feature_df[col].notna().any() else 0.0
                    max_val = float(feature_df[col].max()) if feature_df[col].notna().any() else 100.0
                    median_val = float(feature_df[col].median()) if feature_df[col].notna().any() else 0.0

                    form_values[col] = st.number_input(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val
                    )
                else:
                    options = feature_df[col].dropna().astype(str).unique().tolist()
                    options = sorted(options)

                    form_values[col] = st.selectbox(
                        label=col,
                        options=options if options else ["Unknown"],
                        index=0
                    )

        manual_submit = st.form_submit_button("Simular cliente manual")

    if manual_submit:
        try:
            latest_input_df = build_manual_input(feature_df, form_values, numeric_cols)
            latest_prediction = predict_proba_single(rf_model, latest_input_df)
            st.session_state["simulated_customers"] = add_simulation_record(
                st.session_state["simulated_customers"],
                "manual",
                "Random Forest Tuned",
                latest_input_df,
                latest_prediction
            )
        except Exception as e:
            st.error(f"No se pudo generar la predicción: {e}")

with tab_random:
    st.markdown("Pulsa el botón para simular un cliente aleatorio desde el dataset procesado.")
    random_submit = st.button("Simular cliente aleatorio")

    if random_submit:
        try:
            latest_input_df = get_random_customer(feature_df)
            latest_prediction = predict_proba_single(rf_model, latest_input_df)
            st.session_state["simulated_customers"] = add_simulation_record(
                st.session_state["simulated_customers"],
                "pool",
                "Random Forest Tuned",
                latest_input_df,
                latest_prediction
            )
        except Exception as e:
            st.error(f"No se pudo generar la predicción: {e}")

if latest_prediction is not None:
    st.subheader("Resultado de la simulación")

    risk_label = classify_risk(latest_prediction)
    risk_hex = risk_color(risk_label)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Probabilidad de churn", f"{latest_prediction:.2%}")
    with r2:
        st.markdown(
            f"""
            <div style="
                padding: 0.6rem 0.8rem;
                border-radius: 0.6rem;
                background-color: {risk_hex};
                color: black;
                font-weight: 700;
                text-align: center;
                margin-top: 1.9rem;
            ">
                Riesgo {risk_label}
            </div>
            """,
            unsafe_allow_html=True
        )
    with r3:
        st.metric("Modelo usado", "Random Forest Tuned")

    st.markdown("#### Cliente simulado")
    st.dataframe(latest_input_df, use_container_width=True, hide_index=True)

sim_df = st.session_state["simulated_customers"].copy()

if len(sim_df) == 0:
    st.info("Todavía no hay simulaciones. Crea un cliente manual o usa el botón de cliente aleatorio.")
    st.stop()

st.subheader("Resumen acumulado")

n_sim = len(sim_df)
avg_risk = sim_df["p_churn"].mean()
high_risk_pct = (sim_df["risk_level"] == "Alto").mean()

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Clientes simulados", f"{n_sim:,}")
with k2:
    st.metric("Riesgo medio", f"{avg_risk:.2%}")
with k3:
    st.metric("% alto riesgo", f"{high_risk_pct:.2%}")

st.subheader("Distribución por nivel de riesgo")

risk_counts = sim_df["risk_level"].value_counts().reset_index()
risk_counts.columns = ["risk_level", "count"]

risk_order = ["Bajo", "Medio", "Alto"]
risk_counts["risk_level"] = pd.Categorical(
    risk_counts["risk_level"],
    categories=risk_order,
    ordered=True
)
risk_counts = risk_counts.sort_values("risk_level")

fig_risk = px.pie(
    risk_counts,
    names="risk_level",
    values="count",
    hole=0.5,
    color="risk_level",
    color_discrete_map={
        "Bajo": "#2E8B57",
        "Medio": "#E6C200",
        "Alto": "#C0392B"
    }
)

fig_risk.update_traces(
    textinfo="percent+label",
    pull=[0.02, 0.02, 0.04]
)

fig_risk.update_layout(
    title_text="",
    legend_title="Nivel de riesgo"
)

st.plotly_chart(fig_risk, use_container_width=True)

st.subheader("Historial de simulaciones")

col_filter1, col_filter2 = st.columns(2)

with col_filter1:
    source_filter = st.selectbox(
        "Filtrar por origen",
        ["Todos", "Solo manual", "Solo pool"]
    )

with col_filter2:
    risk_filter = st.selectbox(
        "Filtrar por nivel de riesgo",
        ["Todos", "Bajo", "Medio", "Alto"]
    )

hist_df = sim_df.copy()

if source_filter == "Solo manual":
    hist_df = hist_df[hist_df["source"] == "manual"]
elif source_filter == "Solo pool":
    hist_df = hist_df[hist_df["source"] == "pool"]

hist_df = filter_by_risk_level(hist_df, risk_filter)
hist_df = hist_df.sort_values("p_churn", ascending=False)

preferred_hist_cols = [
    "source",
    "subscription_type",
    "customer_service_inquiries",
    "weekly_hours",
    "song_skip_rate",
    "num_subscription_pauses",
    "p_churn",
    "risk_level",
]

hist_cols = metric_columns_for_display(hist_df, preferred_hist_cols)
st.dataframe(hist_df[hist_cols], use_container_width=True, hide_index=True)
