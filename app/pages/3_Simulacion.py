import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# Configuración
# =========================================================
st.title("Simulación de clientes")
st.markdown(
    """
Simula nuevos clientes, estima su riesgo de churn y analiza el conjunto
de simulaciones recientes desde una perspectiva más operativa.
"""
)

# =========================================================
# Carga de datos y modelo
# =========================================================
@st.cache_data
def load_simulation_data():
    df = pd.read_csv("data/processed/train_model_ready.csv").copy()
    return df


@st.cache_resource
def load_rf_model():
    return joblib.load("models/rf_tuned_model_compressed.joblib")


df = load_simulation_data()
rf_model = load_rf_model()

# =========================================================
# Utilidades
# =========================================================
TARGET = "churned"

drop_cols_if_present = [col for col in ["y_true", "y_pred", "p_churn"] if col in df.columns]
available_df = df.drop(columns=drop_cols_if_present, errors="ignore").copy()

if TARGET in available_df.columns:
    feature_df = available_df.drop(columns=[TARGET]).copy()
else:
    feature_df = available_df.copy()

categorical_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
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

def classify_risk(p):
    if p < 0.33:
        return "Bajo"
    elif p < 0.66:
        return "Medio"
    return "Alto"

def predict_with_model(model_name: str, input_df: pd.DataFrame):
    if model_name == "Random Forest Tuned":
        p_churn = rf_model.predict_proba(input_df)[:, 1][0]
        return p_churn

    tree_model = st.session_state.get("tree_model_in_situ", None)
    if tree_model is None:
        raise ValueError("El árbol in situ no está entrenado todavía.")
    p_churn = tree_model.predict_proba(input_df)[:, 1][0]
    return p_churn

def get_random_customer():
    sample_row = feature_df.sample(1, random_state=np.random.randint(0, 10_000)).copy()
    return sample_row

def build_manual_input(form_values: dict):
    row = {}
    for col in feature_df.columns:
        if col in form_values:
            row[col] = form_values[col]
        else:
            if col in numeric_cols:
                row[col] = float(feature_df[col].median()) if feature_df[col].notna().any() else 0
            else:
                mode_vals = feature_df[col].mode(dropna=True)
                row[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"

    return pd.DataFrame([row])

def add_simulation_record(source_name: str, model_name: str, input_df: pd.DataFrame, p_churn: float):
    record = input_df.copy()
    record["source"] = source_name
    record["model_used"] = model_name
    record["p_churn"] = p_churn
    record["risk_level"] = classify_risk(p_churn)

    current = st.session_state["simulated_customers"]
    st.session_state["simulated_customers"] = pd.concat([current, record], ignore_index=True)

# =========================================================
# Selector de modelo
# =========================================================
st.subheader("Configuración de simulación")

model_options = ["Random Forest Tuned", "Árbol in situ"]
selected_model = st.selectbox("Modelo para la simulación", model_options)

if selected_model == "Árbol in situ" and st.session_state.get("tree_model_in_situ", None) is None:
    st.warning("Primero entrena el árbol de decisión en la página de Modelos para poder usarlo aquí.")

# =========================================================
# Formulario manual + cliente aleatorio
# =========================================================
st.subheader("Nuevo cliente")

tab_manual, tab_random = st.tabs(["Entrada manual", "Cliente aleatorio"])

latest_prediction = None
latest_input_df = None
latest_source = None

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
                    default_value = options[0] if options else "Unknown"

                    form_values[col] = st.selectbox(
                        label=col,
                        options=options if options else ["Unknown"],
                        index=0
                    )

        manual_submit = st.form_submit_button("Simular cliente manual")

    if manual_submit:
        try:
            latest_input_df = build_manual_input(form_values)
            latest_prediction = predict_with_model(selected_model, latest_input_df)
            latest_source = "manual"
            add_simulation_record(latest_source, selected_model, latest_input_df, latest_prediction)
        except Exception as e:
            st.error(f"No se pudo generar la predicción: {e}")

with tab_random:
    st.markdown("Pulsa el botón para cargar un cliente aleatorio desde el dataset procesado.")
    random_submit = st.button("Simular cliente aleatorio")

    if random_submit:
        try:
            latest_input_df = get_random_customer()
            latest_prediction = predict_with_model(selected_model, latest_input_df)
            latest_source = "pool"
            add_simulation_record(latest_source, selected_model, latest_input_df, latest_prediction)
        except Exception as e:
            st.error(f"No se pudo generar la predicción: {e}")

# =========================================================
# Resultado inmediato
# =========================================================
if latest_prediction is not None:
    st.subheader("Resultado de la simulación")

    risk_label = classify_risk(latest_prediction)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Probabilidad de churn", f"{latest_prediction:.2%}")
    with r2:
        st.metric("Nivel de riesgo", risk_label)
    with r3:
        st.metric("Modelo usado", selected_model)

    st.markdown("#### Riesgo estimado")
    st.progress(int(latest_prediction * 100))

    with st.expander("Ver datos del cliente simulado"):
        st.dataframe(latest_input_df, use_container_width=True, hide_index=True)

# =========================================================
# Dataset acumulado de simulaciones
# =========================================================
sim_df = st.session_state["simulated_customers"].copy()

if len(sim_df) == 0:
    st.info("Todavía no hay simulaciones. Crea un cliente manual o usa el botón de cliente aleatorio.")
    st.stop()

# =========================================================
# KPIs acumulados
# =========================================================
st.subheader("Resumen acumulado de simulaciones")

n_sim = len(sim_df)
avg_risk = sim_df["p_churn"].mean()
high_risk_pct = (sim_df["risk_level"] == "Alto").mean()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Clientes simulados", f"{n_sim:,}")
with k2:
    st.metric("Riesgo medio", f"{avg_risk:.2%}")
with k3:
    st.metric("% alto riesgo", f"{high_risk_pct:.2%}")
with k4:
    st.metric("Modelo activo", selected_model)

# =========================================================
# Visualizaciones principales
# =========================================================
st.subheader("Análisis del conjunto simulado")

v1, v2 = st.columns(2)

with v1:
    fig_hist = px.histogram(
        sim_df,
        x="p_churn",
        nbins=20,
        title="Distribución de probabilidad de churn"
    )
    fig_hist.update_layout(
        title_x=0.5,
        xaxis_title="Probabilidad estimada de churn",
        yaxis_title="Número de clientes"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with v2:
    risk_counts = sim_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]

    fig_risk = px.pie(
        risk_counts,
        names="risk_level",
        values="count",
        hole=0.45,
        title="Distribución por nivel de riesgo"
    )
    fig_risk.update_layout(title_x=0.5)
    st.plotly_chart(fig_risk, use_container_width=True)

# =========================================================
# Scatter interactivo
# =========================================================
numeric_plot_candidates = [c for c in numeric_cols if c in sim_df.columns]

if len(numeric_plot_candidates) >= 2:
    st.subheader("Exploración interactiva de clientes simulados")

    s1, s2 = st.columns(2)
    with s1:
        x_axis = st.selectbox(
            "Eje X",
            numeric_plot_candidates,
            index=0
        )
    with s2:
        default_y_idx = 1 if len(numeric_plot_candidates) > 1 else 0
        y_axis = st.selectbox(
            "Eje Y",
            numeric_plot_candidates,
            index=default_y_idx
        )

    hover_cols = [c for c in ["subscription_type", "customer_service_inquiries", "risk_level", "p_churn"] if c in sim_df.columns]

    fig_scatter = px.scatter(
        sim_df,
        x=x_axis,
        y=y_axis,
        color="risk_level",
        size="p_churn",
        hover_data=hover_cols,
        title="Perfiles simulados y riesgo estimado"
    )
    fig_scatter.update_layout(title_x=0.5)
    st.plotly_chart(fig_scatter, use_container_width=True)

# =========================================================
# Historial y top riesgo
# =========================================================
st.subheader("Historial de simulaciones")

filter_option = st.selectbox(
    "Filtrar historial",
    ["Todos", "Solo alto riesgo", "Solo manual", "Solo pool"]
)

hist_df = sim_df.copy()

if filter_option == "Solo alto riesgo":
    hist_df = hist_df[hist_df["risk_level"] == "Alto"]
elif filter_option == "Solo manual":
    hist_df = hist_df[hist_df["source"] == "manual"]
elif filter_option == "Solo pool":
    hist_df = hist_df[hist_df["source"] == "pool"]

hist_df = hist_df.sort_values("p_churn", ascending=False)

preferred_hist_cols = [
    "source",
    "model_used",
    "subscription_type",
    "customer_service_inquiries",
    "weekly_hours",
    "song_skip_rate",
    "num_subscription_pauses",
    "p_churn",
    "risk_level",
]

hist_cols = [c for c in preferred_hist_cols if c in hist_df.columns]
st.dataframe(hist_df[hist_cols], use_container_width=True, hide_index=True)

st.subheader("Clientes simulados con mayor riesgo")

top_risk_df = sim_df.sort_values("p_churn", ascending=False).head(10)
top_cols = [c for c in preferred_hist_cols if c in top_risk_df.columns]
st.dataframe(top_risk_df[top_cols], use_container_width=True, hide_index=True)