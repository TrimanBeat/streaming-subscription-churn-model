import streamlit as st
import pandas as pd
import plotly.express as px
import os

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.title("🤖 Modelos Predictivos")
st.markdown("""
En esta sección se comparan los modelos entrenados y se muestran
las variables más importantes del Random Forest.
""")

@st.cache_data
def load_model_outputs():
    model_metrics = pd.read_csv("data/exports/model_metrics.csv")
    rf_feature_importance = pd.read_csv("data/exports/rf_feature_importance.csv")
    rf_preds = pd.read_csv("data/exports/rf_validation_predictions.csv")
    rf_cm_pct = pd.read_csv("data/exports/rf_confusion_matrix_percentage.csv", index_col=0)
    return model_metrics, rf_feature_importance, rf_preds, rf_cm_pct

model_metrics, rf_feature_importance, rf_preds, rf_cm_pct = load_model_outputs()

# =========================
# KPIs
# =========================
st.subheader("Resumen del rendimiento")

rf_row = model_metrics[model_metrics["model"].str.contains("Random", case=False, na=False)].iloc[0]
logreg_row = model_metrics[model_metrics["model"].str.contains("Logistic", case=False, na=False)].iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy RF", f"{rf_row['accuracy']:.3f}")

with col2:
    st.metric("F1 RF", f"{rf_row['f1']:.3f}")

with col3:
    st.metric("ROC AUC RF", f"{rf_row['roc_auc']:.3f}")

with col4:
    delta_auc = rf_row["roc_auc"] - logreg_row["roc_auc"]
    st.metric("Mejora RF vs LogReg", f"{delta_auc:.3f}")

# =========================
# Comparación de modelos
# =========================
st.subheader("Comparación de modelos")

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
# Matriz de confusión + importancia
# =========================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Matriz de confusión (%) - Random Forest")

    fig_cm = px.imshow(
        rf_cm_pct,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Reds",
        title="Matriz de confusión normalizada"
    )

    fig_cm.update_traces(texttemplate="%{z:.2f}%")
    fig_cm.update_layout(
        title_x=0.5,
        xaxis_title="Clase predicha",
        yaxis_title="Clase real"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

with col_right:
    st.subheader("Top variables más importantes")

    top_features = rf_feature_importance.head(15).copy()
    feature_col = "feature_clean" if "feature_clean" in top_features.columns else "feature"
    top_features = top_features.sort_values("importance", ascending=True)

    fig_importance = px.bar(
        top_features,
        x="importance",
        y=feature_col,
        orientation="h",
        title="Top 15 variables - Random Forest",
        text="importance"
    )

    fig_importance.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_importance.update_layout(
        title_x=0.5,
        xaxis_title="Importancia",
        yaxis_title="Variable"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

# =========================
# Predicciones de riesgo alto
# =========================
st.subheader("Clientes con mayor riesgo estimado")

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
] if col in rf_preds.columns]

top_risk = rf_preds.sort_values("p_churn", ascending=False)[show_cols].head(20)

st.dataframe(top_risk, use_container_width=True)

# =========================
# Hueco para interpretación LLM
# =========================

def build_feature_importance_prompt(rf_feature_importance: pd.DataFrame, top_n: int = 12) -> str:
    feature_col = "feature_clean" if "feature_clean" in rf_feature_importance.columns else "feature"
    top_df = rf_feature_importance[[feature_col, "importance"]].head(top_n).copy()

    lines = []
    for _, row in top_df.iterrows():
        lines.append(f"- {row[feature_col]}: {row['importance']:.4f}")

    features_text = "\n".join(lines)

    prompt = f"""
You are helping explain a churn prediction dashboard to a university audience.

Below is the top feature importance output from a Random Forest model:

{features_text}

Write a short interpretation in Spanish:
- 5 to 7 lines maximum
- Explain the main patterns in business language
- Group the findings into themes if possible, such as engagement, friction, and subscription type
- Mention that feature importance is predictive, not causal
- Do not invent variables or relationships that are not present
- Keep the tone clear, concise, and presentation-ready
"""
    return prompt


def generate_gemini_summary(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "No se encontró la variable de entorno GEMINI_API_KEY."

    if not GEMINI_AVAILABLE:
        return "El paquete google-genai no está instalado en este entorno."

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error al generar el resumen con Gemini: {e}"

st.subheader("Interpretación asistida")

st.caption("Este bloque genera un resumen automático de la importancia de variables del Random Forest.")

if "llm_summary" not in st.session_state:
    st.session_state["llm_summary"] = ""

col_a, col_b = st.columns([1, 3])

with col_a:
    generate_clicked = st.button("Generar resumen con Gemini")

with col_b:
    st.write("Usa el modelo para resumir los principales drivers del churn en lenguaje natural.")

if generate_clicked:
    prompt = build_feature_importance_prompt(rf_feature_importance, top_n=12)
    with st.spinner("Generando resumen..."):
        st.session_state["llm_summary"] = generate_gemini_summary(prompt)

if st.session_state["llm_summary"]:
    st.success("Resumen generado")
    st.write(st.session_state["llm_summary"])
else:
    st.info("Pulsa el botón para generar una interpretación automática.")