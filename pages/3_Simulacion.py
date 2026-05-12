from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from churn_project.data_loader import load_export_csv, load_processed_data
from churn_project.simulation_utils import (
    add_simulation_record,
    build_manual_input,
    classify_risk,
    clear_new_customers_for_training,
    filter_by_risk_level,
    generate_retraining_batch,
    get_random_customer,
    load_incoming_customers,
    load_new_customers_for_training,
    load_retraining_pool,
    predict_proba_single,
    risk_color,
)
from churn_project.summaries import metric_columns_for_display


st.title("Simulación de clientes")
st.caption("Simula nuevos clientes, consulta el modelo final y prepara lotes para reentrenamiento.")
st.markdown("---")


@st.cache_resource
def load_rf_model():
    return joblib.load("models/rf_tuned_model_compressed.joblib")


@st.cache_data
def load_rf_outputs():
    rf_tuned_metrics = load_export_csv("data/exports/rf_tuned_metrics.csv")
    rf_tuned_cm_pct = load_export_csv(
        "data/exports/rf_tuned_confusion_matrix_percentage.csv",
        index_col=0
    )
    rf_tuned_feature_importance = load_export_csv("data/exports/rf_tuned_feature_importance.csv")
    return rf_tuned_metrics, rf_tuned_cm_pct, rf_tuned_feature_importance


incoming_df = load_processed_data("data/processed/incoming_prediction_ready.csv")
rf_model = load_rf_model()
rf_tuned_metrics, rf_tuned_cm_pct, rf_tuned_feature_importance = load_rf_outputs()

TARGET = "churned"

drop_cols_if_present = [col for col in ["y_true", "y_pred", "p_churn"] if col in incoming_df.columns]
available_df = incoming_df.drop(columns=drop_cols_if_present, errors="ignore").copy()

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

tab_random, tab_manual, tab_model, tab_retraining = st.tabs(
    ["Cliente aleatorio", "Entrada manual", "Modelo final RF", "Lote para reentrenamiento"]
)

latest_prediction = None
latest_input_df = None

with tab_random:
    st.markdown("Pulsa el botón para simular un cliente aleatorio desde incoming_prediction_ready.csv.")
    random_submit = st.button("Simular cliente aleatorio")

    if random_submit:
        try:
            latest_input_df = get_random_customer(feature_df)
            latest_prediction = predict_proba_single(rf_model, latest_input_df)
            st.session_state["simulated_customers"] = add_simulation_record(
                st.session_state["simulated_customers"],
                "incoming_pool",
                "Random Forest Tuned",
                latest_input_df,
                latest_prediction
            )
        except Exception as e:
            st.error(f"No se pudo generar la predicción: {e}")

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

with tab_model:
    st.markdown("#### Rendimiento del Random Forest final")

    model_metrics = rf_tuned_metrics.iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
    with m2:
        st.metric("Precision", f"{model_metrics['precision']:.3f}")
    with m3:
        st.metric("Recall", f"{model_metrics['recall']:.3f}")
    with m4:
        st.metric("ROC AUC", f"{model_metrics['roc_auc']:.3f}")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Matriz de confusión (%)")
        fig_cm = px.imshow(
            rf_tuned_cm_pct,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Reds"
        )
        fig_cm.update_traces(texttemplate="%{z:.2f}%")
        fig_cm.update_layout(
            title_text="",
            xaxis_title="Clase predicha",
            yaxis_title="Clase real"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        st.markdown("#### Variables más importantes")

        top_features = rf_tuned_feature_importance.head(12).copy()
        feature_col = "feature_clean" if "feature_clean" in top_features.columns else "feature"

        if feature_col == "feature":
            top_features["feature_clean"] = (
                top_features["feature"]
                .astype(str)
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
                .str.replace("_", " ", regex=False)
            )
            feature_col = "feature_clean"

        top_features = top_features.sort_values("importance", ascending=True)

        fig_imp = px.bar(
            top_features,
            x="importance",
            y=feature_col,
            orientation="h",
            text="importance",
            color_discrete_sequence=["#B42318"]
        )
        fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_imp.update_layout(
            title_text="",
            xaxis_title="Importancia",
            yaxis_title="Variable"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "Este es el modelo final que usa la simulación. "
        "La idea es priorizar acciones sobre perfiles de riesgo medio y alto según el contexto de negocio."
    )

with tab_retraining:
    retraining_pool_df = load_retraining_pool()

    st.markdown("#### Generar lote desde retraining_pool")
    st.caption(
        "Esta pestaña toma una muestra aleatoria del pool etiquetado reservado para simular nuevos datos de entrenamiento "
        "y la guarda en data/new_data/new_customers_for_training.csv para que Dagster la incorpore al reentrenamiento."
    )

    total_pool = len(retraining_pool_df)
    st.metric("Clientes disponibles en retraining_pool", f"{total_pool:,}")

    batch_size = st.number_input(
        "Número de clientes para el lote de reentrenamiento",
        min_value=1,
        max_value=max(1, total_pool),
        value=min(10, max(1, total_pool)),
        step=1,
    )

    batch_seed = st.number_input(
        "Semilla aleatoria (opcional)",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

    btn_col1, btn_col2 = st.columns([1, 1])

    with btn_col1:
        if st.button("Generar lote para reentrenamiento", key="generate_retraining_batch_btn"):
            try:
                batch_df = generate_retraining_batch(
                    n_customers=int(batch_size),
                    random_state=int(batch_seed),
                )
                st.success("Lote generado en data/new_data/new_customers_for_training.csv")
                st.dataframe(batch_df.head(20), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"No se pudo generar el lote: {e}")

    with btn_col2:
        if st.button("Vaciar lote actual", key="clear_retraining_batch_btn"):
            try:
                clear_new_customers_for_training()
                st.warning("Se ha vaciado el archivo data/new_data/new_customers_for_training.csv")
            except Exception as e:
                st.error(f"No se pudo vaciar el lote: {e}")

    current_batch_df = load_new_customers_for_training()
    if len(current_batch_df) > 0:
        st.markdown("#### Lote actual pendiente para Dagster")
        preview_cols = metric_columns_for_display(
            current_batch_df,
            [
                "subscription_type",
                "customer_service_inquiries",
                "weekly_hours",
                "song_skip_rate",
                "num_subscription_pauses",
                "churned",
            ],
        )
        st.dataframe(current_batch_df[preview_cols].head(20), use_container_width=True, hide_index=True)

    st.markdown("#### Pool etiquetado reservado (referencia)")
    preview_cols = metric_columns_for_display(
        retraining_pool_df,
        [
            "subscription_type",
            "customer_service_inquiries",
            "weekly_hours",
            "song_skip_rate",
            "num_subscription_pauses",
            "churned",
        ],
    )
    st.dataframe(retraining_pool_df[preview_cols].head(20), use_container_width=True, hide_index=True)

    st.markdown("#### Pool raw de incoming (referencia)")
    try:
        incoming_raw_df = load_incoming_customers()
        incoming_cols = metric_columns_for_display(
            incoming_raw_df,
            [
                "subscription_type",
                "customer_service_inquiries",
                "weekly_hours",
                "song_skip_rate",
                "num_subscription_pauses",
                "churned",
            ],
        )
        st.dataframe(incoming_raw_df[incoming_cols].head(20), use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"No se pudo cargar incoming_customers.csv directamente: {e}")

if latest_prediction is not None:
    st.subheader("Resultado de la simulación")

    risk_label = classify_risk(latest_prediction)
    risk_hex = risk_color(risk_label)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Probabilidad de churn", f"{latest_prediction:.2%}")
    with r2:
        st.markdown(
            f'''
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
            ''',
            unsafe_allow_html=True
        )
    with r3:
        st.metric("Modelo usado", "Random Forest Tuned")

    st.markdown("#### Cliente simulado")
    st.dataframe(latest_input_df, use_container_width=True, hide_index=True)

sim_df = st.session_state["simulated_customers"].copy()

if len(sim_df) == 0:
    st.info("Todavía no hay simulaciones. Usa la pestaña de cliente aleatorio o la de entrada manual.")
    st.stop()

st.subheader("Resumen acumulado")

n_sim = len(sim_df)
avg_risk = sim_df["p_churn"].mean()
high_risk_pct = (sim_df["risk_level"] == "Alto").mean()
medium_risk_pct = (sim_df["risk_level"] == "Medio").mean()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Clientes simulados", f"{n_sim:,}")
with k2:
    st.metric("Riesgo medio", f"{avg_risk:.2%}")
with k3:
    st.metric("% riesgo medio", f"{medium_risk_pct:.2%}")
with k4:
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
        ["Todos", "Solo manual", "Solo incoming_pool"]
    )

with col_filter2:
    risk_filter = st.selectbox(
        "Filtrar por nivel de riesgo",
        ["Todos", "Bajo", "Medio", "Alto"]
    )

hist_df = sim_df.copy()

if source_filter == "Solo manual":
    hist_df = hist_df[hist_df["source"] == "manual"]
elif source_filter == "Solo incoming_pool":
    hist_df = hist_df[hist_df["source"] == "incoming_pool"]

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
