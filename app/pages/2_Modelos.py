import os
import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# Configuración de página
# =========================
st.title("🤖 Modelos Predictivos")
st.markdown("""
En esta sección se comparan los modelos entrenados y se pueden explorar
sus métricas, matriz de confusión y ejemplos de predicción.
""")

# =========================
# Cargar datos
# =========================
@st.cache_data
def load_model_outputs():
    model_metrics = pd.read_csv("data/exports/model_metrics.csv")
    logreg_preds = pd.read_csv("data/exports/logreg_validation_predictions.csv")
    rf_preds = pd.read_csv("data/exports/rf_validation_predictions.csv")
    logreg_cm_pct = pd.read_csv("data/exports/logreg_confusion_matrix_percentage.csv", index_col=0)
    rf_cm_pct = pd.read_csv("data/exports/rf_confusion_matrix_percentage.csv", index_col=0)
    rf_feature_importance = pd.read_csv("data/exports/rf_feature_importance.csv")
    return (
        model_metrics,
        logreg_preds,
        rf_preds,
        logreg_cm_pct,
        rf_cm_pct,
        rf_feature_importance,
    )

(
    model_metrics,
    logreg_preds,
    rf_preds,
    logreg_cm_pct,
    rf_cm_pct,
    rf_feature_importance,
) = load_model_outputs()

# =========================
# Selector de modelo
# =========================
st.subheader("Selección de modelo")

selected_model = st.selectbox(
    "Selecciona un modelo para explorar",
    ["Logistic Regression", "Random Forest"]
)

# =========================
# Asignar datos según modelo seleccionado
# =========================
if selected_model == "Logistic Regression":
    current_metrics = model_metrics[
        model_metrics["model"].str.contains("Logistic", case=False, na=False)
    ].iloc[0]
    current_preds = logreg_preds
    current_cm = logreg_cm_pct
    show_feature_importance = False
else:
    current_metrics = model_metrics[
        model_metrics["model"].str.contains("Random", case=False, na=False)
    ].iloc[0]
    current_preds = rf_preds
    current_cm = rf_cm_pct
    show_feature_importance = True

# =========================
# KPIs del modelo seleccionado
# =========================
st.subheader("Resumen del modelo seleccionado")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{current_metrics['accuracy']:.3f}")

with col2:
    st.metric("Precision", f"{current_metrics['precision']:.3f}")

with col3:
    st.metric("Recall", f"{current_metrics['recall']:.3f}")

with col4:
    st.metric("ROC AUC", f"{current_metrics['roc_auc']:.3f}")

# =========================
# Comparación global de modelos
# =========================
st.subheader("Comparación global de modelos")

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
    title="Comparación de métricas por modelo",
    text="score"
)

fig_metrics.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_metrics.update_layout(
    title_x=0.5,
    xaxis_title="Métrica",
    yaxis_title="Score"
)

st.plotly_chart(fig_metrics, use_container_width=True)

# =========================
# Matriz de confusión + importance
# =========================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"Matriz de confusión (%) - {selected_model}")

    fig_cm = px.imshow(
        current_cm,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Reds",
        title=f"Matriz de confusión normalizada - {selected_model}"
    )

    fig_cm.update_traces(texttemplate="%{z:.2f}%")
    fig_cm.update_layout(
        title_x=0.5,
        xaxis_title="Clase predicha",
        yaxis_title="Clase real"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

with col_right:
    st.subheader("Interpretación de variables")

    if show_feature_importance:
        top_features = rf_feature_importance.head(15).copy()
        feature_col = "feature_clean" if "feature_clean" in top_features.columns else "feature"
        top_features = top_features.sort_values("importance", ascending=True)

        fig_importance = px.bar(
            top_features,
            x="importance",
            y=feature_col,
            orientation="h",
            title="Top 15 variables más importantes",
            text="importance"
        )

        fig_importance.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_importance.update_layout(
            title_x=0.5,
            xaxis_title="Importancia",
            yaxis_title="Variable"
        )

        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info(
            "Para Logistic Regression no se está mostrando aún una visualización específica "
            "de importancia de variables en esta versión de la app."
        )

# =========================
# Predicciones de mayor riesgo
# =========================
st.subheader(f"Clientes con mayor riesgo estimado - {selected_model}")

show_cols = [col for col in [
    "subscription_type",
    "customer_service_inquiries",
    "weekly_hours",
    "song_skip_rate",
    "num_subscription_pauses",
    "age",
    "p_churn",
    "y_true",
    "y_pred"
] if col in current_preds.columns]

top_risk = current_preds.sort_values("p_churn", ascending=False)[show_cols].head(20)

st.dataframe(top_risk, use_container_width=True)

# =========================
# Gemini: imports opcionales
# =========================
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# =========================
# Funciones para Gemini
# =========================
def get_gemini_api_key():
    # 1. Intentar leer desde Streamlit secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # 2. Si no existe secrets.toml o no está la clave, probar variable de entorno
    return os.getenv("GEMINI_API_KEY")

def build_model_summary_prompt(
    selected_model: str,
    current_metrics: pd.Series,
    rf_feature_importance: pd.DataFrame | None = None,
    top_n: int = 12
) -> str:
    metrics_text = f"""
Modelo seleccionado: {selected_model}

Rendimiento:
- Accuracy: {current_metrics['accuracy']:.3f}
- Precision: {current_metrics['precision']:.3f}
- Recall: {current_metrics['recall']:.3f}
- F1-score: {current_metrics['f1']:.3f}
- ROC AUC: {current_metrics['roc_auc']:.3f}
"""

    if selected_model == "Random Forest" and rf_feature_importance is not None:
        feature_col = "feature_clean" if "feature_clean" in rf_feature_importance.columns else "feature"
        top_df = rf_feature_importance[[feature_col, "importance"]].head(top_n).copy()

        lines = []
        for _, row in top_df.iterrows():
            lines.append(f"- {row[feature_col]}: {row['importance']:.4f}")

        features_text = "\n".join(lines)

        prompt = f"""
Eres un asistente que ayuda a explicar un dashboard de predicción de churn a una audiencia universitaria.

{metrics_text}

Top variables más importantes:
{features_text}

Escribe la respuesta en español y con este formato exacto en Markdown:

### Resumen general
2 o 3 frases breves sobre el rendimiento del modelo.

### Variables clave
- 3 o 4 viñetas explicando los factores más importantes
- agrupa las ideas en bloques si es posible, como engagement, fricción y tipo de suscripción

### Nota de interpretación
1 viñeta breve aclarando que la importancia de variables es predictiva y no causal.

Instrucciones:
- No escribas la respuesta en un solo párrafo
- Usa títulos y viñetas
- Sé claro, breve y presentable
- No inventes relaciones no presentes en la tabla
"""
    else:
        prompt = f"""
Eres un asistente que ayuda a explicar un dashboard de predicción de churn a una audiencia universitaria.

{metrics_text}

Escribe la respuesta en español y con este formato exacto en Markdown:

### Resumen general
2 o 3 frases breves sobre el rendimiento del modelo.

### Lectura de métricas
- 2 o 3 viñetas comentando si el modelo parece equilibrado entre precision y recall
- 1 viñeta breve sobre el ROC AUC

### Nota final
1 frase breve sobre si el modelo parece adecuado para una demo o despliegue ligero.

Instrucciones:
- No escribas la respuesta en un solo párrafo
- Usa títulos y viñetas
- Sé claro, breve y presentable
"""
    return prompt

def generate_gemini_summary(prompt: str) -> str:
    api_key = get_gemini_api_key()

    if not api_key:
        return (
            "No se encontró ninguna API key de Gemini. "
            "En local usa una variable de entorno GEMINI_API_KEY o crea un secrets.toml. "
            "En Streamlit Cloud añádela en la sección Secrets."
        )

    if not GEMINI_AVAILABLE:
        return "El paquete google-genai no está instalado en este entorno."

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error al generar el resumen con Gemini: {e}"

# =========================
# Bloque de interpretación asistida
# =========================
st.subheader("Interpretación asistida")

st.caption(
    "Este bloque genera un resumen automático del modelo seleccionado."
)

summary_key = f"llm_summary_{selected_model}"

if summary_key not in st.session_state:
    st.session_state[summary_key] = ""

col_a, col_b = st.columns([1, 3])

with col_a:
    generate_clicked = st.button(f"Generar resumen - {selected_model}")

with col_b:
    st.write("Usa Gemini para generar una explicación breve del modelo seleccionado.")

if generate_clicked:
    prompt = build_model_summary_prompt(
        selected_model=selected_model,
        current_metrics=current_metrics,
        rf_feature_importance=rf_feature_importance if selected_model == "Random Forest" else None,
        top_n=12
    )
    with st.spinner("Generando resumen..."):
        st.session_state[summary_key] = generate_gemini_summary(prompt)

if st.session_state[summary_key]:
    st.success("Resumen generado")
    with st.container(border=True):
        st.markdown(st.session_state[summary_key])
else:
    st.info("Pulsa el botón para generar una interpretación automática.")