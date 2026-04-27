import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.title("Simulación en vivo")
st.markdown("""
Esta sección permite simular nuevos clientes y estimar su riesgo de churn
utilizando el modelo de Logistic Regression desplegado en la aplicación.
""")

# =========================
# Cargar modelo y base
# =========================
@st.cache_resource
def load_model():
    return joblib.load("models/logreg_model.joblib")

@st.cache_data
def load_simulation_base():
    return pd.read_csv("data/exports/rf_simulation_base.csv")

logreg_model = load_model()
simulation_base = load_simulation_base()

# =========================
# Session state
# =========================
if "history" not in st.session_state:
    st.session_state["history"] = []

if "simulated_customers" not in st.session_state:
    st.session_state["simulated_customers"] = pd.DataFrame()

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

# =========================
# Funciones auxiliares
# =========================
def classify_risk(p_churn: float) -> str:
    if p_churn >= 0.70:
        return "High Risk"
    elif p_churn >= 0.40:
        return "Medium Risk"
    return "Low Risk"

def risk_color(risk_label: str) -> str:
    if risk_label == "High Risk":
        return "red"
    elif risk_label == "Medium Risk":
        return "orange"
    return "green"

def update_history():
    n_active = len(st.session_state["simulated_customers"])
    st.session_state["history"].append({
        "timestamp": datetime.now(),
        "n_active": n_active
    })

def score_customer(model, customer_df: pd.DataFrame):
    """
    Devuelve probabilidad de churn y predicción binaria.
    """
    p_churn = model.predict_proba(customer_df)[0, 1]
    y_pred = model.predict(customer_df)[0]
    return p_churn, y_pred

# =========================
# KPIs
# =========================
st.subheader("Resumen de simulación")

n_active = len(st.session_state["simulated_customers"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Clientes simulados", n_active)

with col2:
    if n_active > 0 and "p_churn" in st.session_state["simulated_customers"].columns:
        avg_risk = st.session_state["simulated_customers"]["p_churn"].mean()
        st.metric("Riesgo medio", f"{avg_risk:.2%}")
    else:
        st.metric("Riesgo medio", "N/A")

with col3:
    if n_active > 0 and "risk_label" in st.session_state["simulated_customers"].columns:
        high_risk_n = (st.session_state["simulated_customers"]["risk_label"] == "High Risk").sum()
        st.metric("Clientes High Risk", int(high_risk_n))
    else:
        st.metric("Clientes High Risk", 0)

with col4:
    if st.session_state["history"]:
        st.metric("Eventos simulados", len(st.session_state["history"]))
    else:
        st.metric("Eventos simulados", 0)

# =========================
# Bloques principales
# =========================
tab1, tab2 = st.tabs(["Añadir desde base", "Entrada manual"])

# ---------------------------------
# TAB 1: añadir desde base
# ---------------------------------
with tab1:
    st.subheader("Añadir cliente desde la base de validación")

    st.write("""
    Este botón toma un cliente de la base de validación exportada,
    calcula su riesgo de churn y lo añade al historial de simulación.
    """)

    if st.button("Simular nuevo cliente desde base"):
        sampled = simulation_base.sample(1).copy()

        # Dejamos solo columnas útiles para scoring:
        # quitamos columnas que no son features del modelo si existen
        cols_to_drop = [col for col in ["y_true", "y_pred", "p_churn"] if col in sampled.columns]
        customer_features = sampled.drop(columns=cols_to_drop).copy()

        p_churn, y_pred = score_customer(logreg_model, customer_features)

        sampled["timestamp_entry"] = datetime.now()
        sampled["source"] = "validation_sample"
        sampled["p_churn"] = p_churn
        sampled["predicted_churn"] = y_pred
        sampled["risk_label"] = classify_risk(p_churn)

        if st.session_state["simulated_customers"].empty:
            st.session_state["simulated_customers"] = sampled
        else:
            st.session_state["simulated_customers"] = pd.concat(
                [st.session_state["simulated_customers"], sampled],
                ignore_index=True
            )

        st.session_state["last_prediction"] = {
            "p_churn": p_churn,
            "risk_label": classify_risk(p_churn),
            "source": "validation_sample"
        }

        update_history()

        st.success("Cliente simulado añadido correctamente.")

# ---------------------------------
# TAB 2: entrada manual
# ---------------------------------
with tab2:
    st.subheader("Crear cliente manualmente")

    st.write("""
    Introduce algunas características del cliente para estimar su riesgo de churn.
    """)

    # OJO:
    # Este formulario usa algunas variables base del dataset.
    # Para que funcione, deben existir como columnas en simulation_base.
    # Si alguna no existe en tu CSV exportado, la quitamos luego.

    with st.form("manual_customer_form"):
        age = st.slider("Age", min_value=18, max_value=79, value=30)
        weekly_hours = st.slider("Weekly hours", min_value=0, max_value=50, value=10)
        song_skip_rate = st.slider("Song skip rate", min_value=0.0, max_value=1.0, value=0.5)
        num_subscription_pauses = st.slider("Subscription pauses", min_value=0, max_value=5, value=0)

        subscription_type = st.selectbox(
            "Subscription type",
            sorted(simulation_base["subscription_type"].dropna().unique().tolist())
            if "subscription_type" in simulation_base.columns else ["Free", "Premium", "Student", "Family"]
        )

        customer_service_inquiries = st.selectbox(
            "Customer service inquiries",
            sorted(simulation_base["customer_service_inquiries"].dropna().unique().tolist())
            if "customer_service_inquiries" in simulation_base.columns else ["Low", "Medium", "High"]
        )

        submit_manual = st.form_submit_button("Calcular riesgo")

    if submit_manual:
        # Construimos un registro base usando medianas/modas del dataset
        base_row = simulation_base.drop(columns=[c for c in ["y_true", "y_pred", "p_churn"] if c in simulation_base.columns]).head(1).copy()

        # Rellenamos numéricas con medianas
        for col in base_row.columns:
            if pd.api.types.is_numeric_dtype(base_row[col]):
                base_row[col] = simulation_base[col].median()

        # Rellenamos categóricas con moda
        for col in base_row.columns:
            if base_row[col].dtype == "object":
                mode_vals = simulation_base[col].mode()
                if len(mode_vals) > 0:
                    base_row[col] = mode_vals.iloc[0]

        # Sobrescribimos con lo introducido por el usuario
        if "age" in base_row.columns:
            base_row["age"] = age
        if "weekly_hours" in base_row.columns:
            base_row["weekly_hours"] = weekly_hours
        if "song_skip_rate" in base_row.columns:
            base_row["song_skip_rate"] = song_skip_rate
        if "num_subscription_pauses" in base_row.columns:
            base_row["num_subscription_pauses"] = num_subscription_pauses
        if "subscription_type" in base_row.columns:
            base_row["subscription_type"] = subscription_type
        if "customer_service_inquiries" in base_row.columns:
            base_row["customer_service_inquiries"] = customer_service_inquiries

        p_churn, y_pred = score_customer(logreg_model, base_row)

        result_row = base_row.copy()
        result_row["timestamp_entry"] = datetime.now()
        result_row["source"] = "manual"
        result_row["p_churn"] = p_churn
        result_row["predicted_churn"] = y_pred
        result_row["risk_label"] = classify_risk(p_churn)

        if st.session_state["simulated_customers"].empty:
            st.session_state["simulated_customers"] = result_row
        else:
            st.session_state["simulated_customers"] = pd.concat(
                [st.session_state["simulated_customers"], result_row],
                ignore_index=True
            )

        st.session_state["last_prediction"] = {
            "p_churn": p_churn,
            "risk_label": classify_risk(p_churn),
            "source": "manual"
        }

        update_history()

        st.success("Cliente manual añadido correctamente.")

# =========================
# Última predicción
# =========================
st.subheader("Última predicción")

if st.session_state["last_prediction"] is not None:
    last_pred = st.session_state["last_prediction"]
    risk_label = last_pred["risk_label"]
    p_churn = last_pred["p_churn"]
    color = risk_color(risk_label)

    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 0.75rem;
            border: 2px solid {color};
            background-color: rgba(255,255,255,0.02);
        ">
            <h4 style="margin-bottom: 0.5rem;">Resultado del scoring</h4>
            <p><strong>Origen:</strong> {last_pred['source']}</p>
            <p><strong>Probabilidad estimada de churn:</strong> {p_churn:.2%}</p>
            <p><strong>Nivel de riesgo:</strong> <span style="color:{color};"><strong>{risk_label}</strong></span></p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Todavía no se ha generado ninguna predicción.")

# =========================
# Evolución de clientes activos
# =========================
st.subheader("Evolución de clientes activos")

if len(st.session_state["history"]) > 0:
    history_df = pd.DataFrame(st.session_state["history"])
    st.line_chart(history_df.set_index("timestamp")["n_active"])
else:
    st.write("Aún no hay historial de simulación.")

# =========================
# Historial de clientes simulados
# =========================
st.subheader("Historial de clientes simulados")

if not st.session_state["simulated_customers"].empty:
    display_cols = [col for col in [
        "source",
        "timestamp_entry",
        "subscription_type",
        "customer_service_inquiries",
        "weekly_hours",
        "song_skip_rate",
        "num_subscription_pauses",
        "age",
        "p_churn",
        "risk_label",
        "predicted_churn"
    ] if col in st.session_state["simulated_customers"].columns]

    history_display = st.session_state["simulated_customers"][display_cols].sort_values(
        "timestamp_entry", ascending=False
    )

    st.dataframe(history_display, use_container_width=True)
else:
    st.write("Todavía no se han simulado clientes.")