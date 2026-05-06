import os
import time

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import ast

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================================================
# Gemini import opcional
# =========================================================
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# =========================================================
# Título
# =========================================================
st.subheader("Modelos Predictivos")
st.markdown("""
En esta sección se comparan los modelos ajustados del proyecto y se exploran
sus métricas, errores y principales señales de riesgo.
""")

# =========================================================
# Carga de outputs
# =========================================================
@st.cache_data
@st.cache_data
def load_model_outputs():
    tuned_model_comparison = pd.read_csv("data/exports/tuned_model_comparison.csv")
    tuned_model_comparison_long = pd.read_csv("data/exports/tuned_model_comparison_long.csv")
    tuned_model_best_params = pd.read_csv("data/exports/tuned_model_best_params.csv")

    logreg_tuned_metrics = pd.read_csv("data/exports/logreg_tuned_metrics.csv")
    logreg_tuned_preds = pd.read_csv("data/exports/logreg_tuned_validation_predictions.csv")
    logreg_tuned_cm_pct = pd.read_csv(
        "data/exports/logreg_tuned_confusion_matrix_percentage.csv",
        index_col=0
    )
    logreg_tuned_coefficients = pd.read_csv("data/exports/logreg_tuned_coefficients.csv")

    rf_tuned_metrics = pd.read_csv("data/exports/rf_tuned_metrics.csv")
    rf_tuned_preds = pd.read_csv("data/exports/rf_tuned_validation_predictions.csv")
    rf_tuned_cm_pct = pd.read_csv(
        "data/exports/rf_tuned_confusion_matrix_percentage.csv",
        index_col=0
    )
    rf_tuned_feature_importance = pd.read_csv("data/exports/rf_tuned_feature_importance.csv")

    dnn_tuned_metrics = pd.read_csv("data/exports/dnn_tuned_metrics.csv")
    dnn_tuned_preds = pd.read_csv("data/exports/dnn_tuned_validation_predictions.csv")
    dnn_tuned_cm_pct = pd.read_csv(
        "data/exports/dnn_tuned_confusion_matrix_percentage.csv",
        index_col=0
    )

    train_model_ready = pd.read_csv("data/processed/train_model_ready.csv")

    return (
        tuned_model_comparison,
        tuned_model_comparison_long,
        tuned_model_best_params,
        logreg_tuned_metrics,
        logreg_tuned_preds,
        logreg_tuned_cm_pct,
        logreg_tuned_coefficients,
        rf_tuned_metrics,
        rf_tuned_preds,
        rf_tuned_cm_pct,
        rf_tuned_feature_importance,
        dnn_tuned_metrics,
        dnn_tuned_preds,
        dnn_tuned_cm_pct,
        train_model_ready,
    )


(
    tuned_model_comparison,
    tuned_model_comparison_long,
    tuned_model_best_params,
    logreg_tuned_metrics,
    logreg_tuned_preds,
    logreg_tuned_cm_pct,
    logreg_tuned_coefficients,
    rf_tuned_metrics,
    rf_tuned_preds,
    rf_tuned_cm_pct,
    rf_tuned_feature_importance,
    dnn_tuned_metrics,
    dnn_tuned_preds,
    dnn_tuned_cm_pct,
    train_model_ready,
) = load_model_outputs()

# =========================================================
# Funciones auxiliares - Gemini
# =========================================================
def get_gemini_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


def build_model_summary_prompt(
    selected_model: str,
    current_metrics: pd.Series,
    rf_feature_importance: pd.DataFrame | None = None,
    logreg_coefficients: pd.DataFrame | None = None,
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
- 1 viñeta breve aclarando que la importancia de variables es predictiva y no causal

Instrucciones:
- No escribas la respuesta en un solo párrafo
- Usa títulos y viñetas
- Sé claro, breve y presentable
- No inventes relaciones no presentes en la tabla
"""
        return prompt

    if selected_model == "Logistic Regression" and logreg_coefficients is not None:
        coef_col = "feature_clean" if "feature_clean" in logreg_coefficients.columns else "feature"

        pos_df = logreg_coefficients.sort_values("coefficient", ascending=False).head(6)
        neg_df = logreg_coefficients.sort_values("coefficient", ascending=True).head(6)

        pos_lines = [f"- {row[coef_col]}: {row['coefficient']:.4f}" for _, row in pos_df.iterrows()]
        neg_lines = [f"- {row[coef_col]}: {row['coefficient']:.4f}" for _, row in neg_df.iterrows()]

        prompt = f"""
Eres un asistente que ayuda a explicar un dashboard de predicción de churn a una audiencia universitaria.

{metrics_text}

Coeficientes positivos más relevantes:
{chr(10).join(pos_lines)}

Coeficientes negativos más relevantes:
{chr(10).join(neg_lines)}

Escribe la respuesta en español y con este formato exacto en Markdown:

### Resumen general
2 o 3 frases breves sobre el rendimiento del modelo.

### Lectura de coeficientes
- 2 o 3 viñetas explicando qué variables parecen empujar hacia churn
- 2 o 3 viñetas explicando qué variables parecen asociarse con menor riesgo
- aclara que los coeficientes dependen del resto de variables del modelo

### Nota final
- 1 viñeta breve sobre el carácter interpretable de la regresión logística

Instrucciones:
- No escribas la respuesta en un solo párrafo
- Usa títulos y viñetas
- Sé claro, breve y presentable
- No inventes relaciones no presentes en la tabla
"""
        return prompt

    if selected_model == "Deep Neural Network":
        prompt = f"""
Eres un asistente que ayuda a explicar un dashboard de predicción de churn a una audiencia universitaria.

Modelo seleccionado: {selected_model}

Rendimiento:
- Accuracy: {current_metrics['accuracy']:.3f}
- Precision: {current_metrics['precision']:.3f}
- Recall: {current_metrics['recall']:.3f}
- F1-score: {current_metrics['f1']:.3f}
- ROC AUC: {current_metrics['roc_auc']:.3f}

Escribe la respuesta en español y con este formato exacto en Markdown:

### Resumen general
2 o 3 frases sobre el rendimiento de la red neuronal.

### Lectura del modelo
- 2 o 3 viñetas explicando que se trata de una red densa para datos tabulares
- 1 viñeta explicando que su interpretación interna es menos directa que la de LogReg o RF
- 1 viñeta explicando por qué sigue siendo útil compararla

### Nota final
- 1 frase breve indicando si el modelo mejora o no a los modelos clásicos del proyecto

Instrucciones:
- No escribas la respuesta en un solo párrafo
- Usa títulos y viñetas
- Sé claro, breve y presentable
"""
        return prompt

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
- 1 frase breve sobre si el modelo parece adecuado para una demo o despliegue ligero

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
    
    
    
# =========================================================
# Funciones auxiliares - Gráfico DNN
# =========================================================    
    
def get_dnn_config(best_params_df: pd.DataFrame):
    dnn_row = best_params_df[best_params_df["model"] == "DNN Tuned"].iloc[0]

    hidden_units_raw = dnn_row["classifier__model__hidden_units"]
    if isinstance(hidden_units_raw, str):
        hidden_units = ast.literal_eval(hidden_units_raw)
    else:
        hidden_units = hidden_units_raw

    config = {
        "hidden_units": hidden_units,
        "dropout_rate": float(dnn_row["classifier__model__dropout_rate"]),
        "learning_rate": float(dnn_row["classifier__model__learning_rate"]),
        "batch_size": int(dnn_row["classifier__batch_size"]),
        "epochs": int(dnn_row["classifier__epochs"]),
    }
    return config

# =========================================================
# Funciones auxiliares - Árbol in situ
# =========================================================
def build_tree_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


def train_decision_tree_in_situ(
    df: pd.DataFrame,
    target: str = "churned",
    max_depth: int = 4,
    min_samples_split: int = 20,
    test_size: float = 0.2,
    random_state: int = 42
):
    df = df.copy()

    drop_cols = [col for col in ["y_true", "y_pred", "p_churn"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target])
    y = df[target]

    preprocessor = build_tree_preprocessor(X)

    tree_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        ))
    ])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    tree_model.fit(X_train, y_train)

    y_pred = tree_model.predict(X_valid)
    y_proba = tree_model.predict_proba(X_valid)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred, zero_division=0),
        "recall": recall_score(y_valid, y_pred, zero_division=0),
        "f1": f1_score(y_valid, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_valid, y_proba),
    }

    feature_names = tree_model.named_steps["preprocessor"].get_feature_names_out()
    fitted_tree = tree_model.named_steps["classifier"]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": fitted_tree.feature_importances_
    }).sort_values("importance", ascending=False)

    return tree_model, metrics, importance_df, feature_names


def build_tree_figure(tree_model, feature_names, max_depth_display=3):
    clf = tree_model.named_steps["classifier"]

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["No Churn", "Churn"],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth_display,
        ax=ax
    )
    plt.tight_layout()
    return fig

# =========================================================
# Comparación global
# =========================================================
st.subheader("Comparación global de modelos")

comparison_long = tuned_model_comparison_long.copy()

metric_order = ["accuracy", "precision", "recall", "f1", "roc_auc"]
comparison_long["metric"] = pd.Categorical(
    comparison_long["metric"],
    categories=metric_order,
    ordered=True
)

model_order = ["LogReg Tuned", "RF Tuned", "DNN Tuned"]
comparison_long["model"] = pd.Categorical(
    comparison_long["model"],
    categories=model_order,
    ordered=True
)

comparison_long = comparison_long.sort_values(["metric", "model"])

fig_metrics = px.bar(
    comparison_long,
    x="metric",
    y="score",
    color="model",
    barmode="group",
    title="Comparación de modelos tuneados",
    text="score"
)

fig_metrics.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_metrics.update_layout(
    title_x=0.5,
    xaxis_title="Métrica",
    yaxis_title="Score"
)

st.plotly_chart(fig_metrics, use_container_width=True)

# =========================================================
# Selector de modelo
# =========================================================
st.subheader("Selección de modelo")

selected_model = st.selectbox(
    "Selecciona un modelo para explorar",
    ["Logistic Regression", "Random Forest", "Deep Neural Network"]
)

if selected_model == "Logistic Regression":
    current_metrics = logreg_tuned_metrics.iloc[0]
    current_preds = logreg_tuned_preds
    current_cm = logreg_tuned_cm_pct
    show_feature_visual = "coefficients"

elif selected_model == "Random Forest":
    current_metrics = rf_tuned_metrics.iloc[0]
    current_preds = rf_tuned_preds
    current_cm = rf_tuned_cm_pct
    show_feature_visual = "importance"

else:
    current_metrics = dnn_tuned_metrics.iloc[0]
    current_preds = dnn_tuned_preds
    current_cm = dnn_tuned_cm_pct
    show_feature_visual = "dnn"

# =========================================================
# Resumen del modelo seleccionado
# =========================================================
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

# =========================================================
# Matriz + visualización del modelo
# =========================================================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"Matriz de confusión (%) - {selected_model}")

    fig_cm = px.imshow(
        current_cm,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Reds",
        title=" "
    )

    fig_cm.update_traces(texttemplate="%{z:.2f}%")
    fig_cm.update_layout(
        title_x=0.5,
        xaxis_title="Clase predicha",
        yaxis_title="Clase real"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

with col_right:
    st.subheader("Interpretación del modelo")

    if show_feature_visual == "importance":
        top_features = rf_tuned_feature_importance.head(15).copy()
        feature_col = "feature_clean" if "feature_clean" in top_features.columns else "feature"
        top_features = top_features.sort_values("importance", ascending=True)

        fig_importance = px.bar(
            top_features,
            x="importance",
            y=feature_col,
            orientation="h",
            title=" ",
            text="importance"
        )

        fig_importance.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_importance.update_layout(
            title_x=0.5,
            xaxis_title="Importancia",
            yaxis_title="Variable"
        )

        st.plotly_chart(fig_importance, use_container_width=True)

    elif show_feature_visual == "coefficients":
        coef_df = logreg_tuned_coefficients.copy()
        feature_col = "feature_clean" if "feature_clean" in coef_df.columns else "feature"

        top_pos = coef_df.sort_values("coefficient", ascending=False).head(8)
        top_neg = coef_df.sort_values("coefficient", ascending=True).head(8)
        coef_plot_df = pd.concat([top_neg, top_pos], axis=0)

        fig_coef = px.bar(
            coef_plot_df,
            x="coefficient",
            y=feature_col,
            orientation="h",
            title=" ",
            text="coefficient"
        )

        fig_coef.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_coef.update_layout(
            title_x=0.5,
            xaxis_title="Coeficiente",
            yaxis_title="Variable"
        )

        st.plotly_chart(fig_coef, use_container_width=True)

    else:
        dnn_config = get_dnn_config(tuned_model_best_params)

        st.markdown("#### Arquitectura de la red")

        arch_rows = [{"Capa": "Entrada", "Configuración": "Features preprocesadas"}]

        for i, units in enumerate(dnn_config["hidden_units"], start=1):
            arch_rows.append({
                "Capa": f"Dense {i}",
                "Configuración": f"{units} neuronas + ReLU"
            })
            arch_rows.append({
                "Capa": f"Dropout {i}",
                "Configuración": f"rate = {dnn_config['dropout_rate']}"
            })

        arch_rows.append({
            "Capa": "Salida",
            "Configuración": "1 neurona + Sigmoid"
        })

        arch_df = pd.DataFrame(arch_rows)
        st.dataframe(arch_df, use_container_width=True, hide_index=True)

        st.markdown("#### Mejores hiperparámetros")
        dnn_config_df = pd.DataFrame({
            "Parámetro": ["Hidden units", "Dropout rate", "Learning rate", "Batch size", "Epochs"],
            "Valor": [
                str(dnn_config["hidden_units"]),
                dnn_config["dropout_rate"],
                dnn_config["learning_rate"],
                dnn_config["batch_size"],
                dnn_config["epochs"],
            ]
        })
        st.dataframe(dnn_config_df, use_container_width=True, hide_index=True)

# =========================================================
# Clientes activos con mayor riesgo
# =========================================================
st.subheader(f"Clientes activos con mayor riesgo estimado - {selected_model}")

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

current_preds_filtered = current_preds[current_preds["y_true"] == 0].copy()
top_risk = current_preds_filtered.sort_values("p_churn", ascending=False)[show_cols].head(20)

st.dataframe(top_risk, use_container_width=True)

if selected_model == "Deep Neural Network":
    st.subheader("Distribución de probabilidades predichas - DNN")

    fig_dnn_probs = px.histogram(
        current_preds,
        x="p_churn",
        nbins=30,
        title=" "
    )

    fig_dnn_probs.update_layout(
        title_x=0.5,
        xaxis_title="Probabilidad estimada de churn",
        yaxis_title="Número de clientes"
    )

    st.plotly_chart(fig_dnn_probs, use_container_width=True)

# =========================================================
# Gemini
# =========================================================
st.subheader("Interpretación asistida")

st.caption("Este bloque genera un resumen automático del modelo seleccionado.")

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
        rf_feature_importance=rf_tuned_feature_importance if selected_model == "Random Forest" else None,
        logreg_coefficients=logreg_tuned_coefficients if selected_model == "Logistic Regression" else None,
        top_n=12
    )
    with st.spinner("Generando resumen..."):
        st.session_state[summary_key] = generate_gemini_summary(prompt)

if st.session_state[summary_key]:
    st.success("Resumen generado")
    st.markdown(st.session_state[summary_key])
else:
    st.info("Pulsa el botón para generar una interpretación automática.")

# =========================================================
# Árbol in situ
# =========================================================
st.markdown("---")
st.subheader("Entrenamiento in situ: Árbol de decisión")
st.caption(
    "Esta sección entrena un modelo sencillo directamente desde la app para demostrar "
    "el flujo completo de entrenamiento, evaluación e interpretación."
)

train_col1, train_col2, train_col3 = st.columns(3)

with train_col1:
    max_depth_in_situ = st.slider(
        "max_depth",
        min_value=2,
        max_value=8,
        value=4,
        step=1
    )

with train_col2:
    min_samples_split_in_situ = st.slider(
        "min_samples_split",
        min_value=2,
        max_value=50,
        value=20,
        step=2
    )

with train_col3:
    train_clicked = st.button("Entrenar árbol in situ")

if "tree_model_in_situ" not in st.session_state:
    st.session_state["tree_model_in_situ"] = None
    st.session_state["tree_metrics_in_situ"] = None
    st.session_state["tree_importance_in_situ"] = None
    st.session_state["tree_feature_names_in_situ"] = None

if train_clicked:
    progress_text = st.empty()
    progress_bar = st.progress(0)

    progress_text.write("Preparando datos...")
    progress_bar.progress(20)
    time.sleep(0.2)

    progress_text.write("Entrenando árbol de decisión...")
    progress_bar.progress(50)

    tree_model_in_situ, tree_metrics_in_situ, tree_importance_in_situ, tree_feature_names_in_situ = train_decision_tree_in_situ(
        train_model_ready,
        target="churned",
        max_depth=max_depth_in_situ,
        min_samples_split=min_samples_split_in_situ
    )

    progress_text.write("Calculando métricas e importancias...")
    progress_bar.progress(80)
    time.sleep(0.2)

    st.session_state["tree_model_in_situ"] = tree_model_in_situ
    st.session_state["tree_metrics_in_situ"] = tree_metrics_in_situ
    st.session_state["tree_importance_in_situ"] = tree_importance_in_situ
    st.session_state["tree_feature_names_in_situ"] = tree_feature_names_in_situ

    progress_text.write("Entrenamiento completado")
    progress_bar.progress(100)
    time.sleep(0.3)

    progress_bar.empty()
    progress_text.empty()

if st.session_state["tree_model_in_situ"] is not None:
    st.success("Modelo entrenado correctamente")

    metrics = st.session_state["tree_metrics_in_situ"]

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with m2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with m3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with m4:
        st.metric("F1", f"{metrics['f1']:.3f}")
    with m5:
        st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

    viz_col1, viz_col2 = st.columns([1.4, 1])

    with viz_col1:
        st.markdown("#### Árbol de decisión")
        tree_fig = build_tree_figure(
            st.session_state["tree_model_in_situ"],
            st.session_state["tree_feature_names_in_situ"],
            max_depth_display=3
        )
        st.pyplot(tree_fig)

    with viz_col2:
        st.markdown("#### Importancia de variables")
        tree_importance_df = st.session_state["tree_importance_in_situ"].head(10).copy()
        tree_importance_df["feature_clean"] = (
            tree_importance_df["feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
            .str.replace("_", " ", regex=False)
        )

        fig_tree_imp = px.bar(
            tree_importance_df.sort_values("importance", ascending=True),
            x="importance",
            y="feature_clean",
            orientation="h",
            title=" ",
            text="importance"
        )

        fig_tree_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_tree_imp.update_layout(title_x=0.5)
        st.plotly_chart(fig_tree_imp, use_container_width=True)

else:
    st.info("Pulsa el botón para entrenar un árbol de decisión dentro de la app.")
