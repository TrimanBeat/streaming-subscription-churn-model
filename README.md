# Streaming Subscription Churn Prediction

Proyecto de análisis, modelización y despliegue sobre abandono de clientes en una plataforma de streaming musical.

El objetivo es entender qué factores están más relacionados con el churn y construir una solución reproducible que combine:

- preparación de datos
- análisis exploratorio
- comparación de modelos
- despliegue en Streamlit
- orquestación con Dagster
- uso complementario de Dask y R

## Objetivo del proyecto

Este proyecto busca estimar el riesgo de abandono (`churn`) de clientes de una plataforma de streaming a partir de variables de uso, comportamiento, tipo de suscripción y contexto geográfico.

A nivel práctico, el trabajo cubre el ciclo completo de un proyecto de datos:

- importación y preparación del dataset
- análisis exploratorio interactivo
- creación de variables derivadas
- comparación de modelos de machine learning
- selección de un modelo final
- simulación de nuevos clientes
- generación de artefactos reproducibles para la app
- orquestación del flujo con Dagster

## Dataset

El dataset contiene información de clientes de una plataforma de streaming, con variables como:

- tipo de suscripción
- horas semanales de uso
- canciones reproducidas
- skip rate
- tiempo medio de sesión
- pausas de suscripción
- consultas a atención al cliente
- edad y localización

La variable objetivo es:

- `churned`: indica si el cliente abandonó la plataforma o no

## Estructura del proyecto

```text
streaming-subscription-churn-model/
├── .streamlit/
│   └── config.toml
├── assets/
│   └── logo.png
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── train_model_ready.csv
│   └── exports/
│       ├── tuned_model_comparison.csv
│       ├── tuned_model_comparison_long.csv
│       ├── tuned_model_best_params.csv
│       ├── logreg_tuned_metrics.csv
│       ├── logreg_tuned_validation_predictions.csv
│       ├── logreg_tuned_confusion_matrix_percentage.csv
│       ├── logreg_tuned_coefficients.csv
│       ├── rf_tuned_metrics.csv
│       ├── rf_tuned_validation_predictions.csv
│       ├── rf_tuned_confusion_matrix_percentage.csv
│       ├── rf_tuned_feature_importance.csv
│       ├── dnn_tuned_metrics.csv
│       ├── dnn_tuned_validation_predictions.csv
│       ├── dnn_tuned_confusion_matrix_percentage.csv
│       └── segment_summary_r.csv
├── models/
│   └── rf_tuned_model_compressed.joblib
├── notebooks/
│   ├── 01_model_comparison_tuned.ipynb
│   ├── 02_train_final_model_rf.ipynb
│   └── EDAinteractivo_final.ipynb
├── pages/
│   ├── 1_EDA_y_segmentos.py
│   ├── 2_Modelos.py
│   └── 3_Simulacion.py
├── scripts/
│   └── segment_summary.R
├── src/
│   └── churn_project/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── eda_utils.py
│       ├── features.py
│       ├── summaries.py
│       ├── simulation_utils.py
│       ├── modeling_utils.py
│       ├── rf_training_utils.py
│       ├── dagster_assets.py
│       └── definitions.py
├── app/
│   └── styles.css
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Preparación de datos y feature engineering

El dataset procesado oficial del proyecto es `data/processed/train_model_ready.csv`.

Durante la preparación se generaron, entre otras, las siguientes variables derivadas:

- `tenure_days`
- `tenure_months`
- `songs_per_hour`
- `high_skip_user`
- `age_group`
- `weekly_hours_bin`
- `skip_rate_bin`
- `state_code`
- `region`

La lógica de preparación quedó centralizada en funciones reutilizables dentro de `src/churn_project/features.py`, lo que permite usar la misma lógica en la app y en Dagster.

## Análisis exploratorio

La página de EDA permite explorar el dataset de forma interactiva mediante:

- resumen general
- análisis por segmentos
- análisis geográfico
- distribuciones de variables numéricas
- tablas de apoyo
- carga de resúmenes analíticos generados en R

El análisis confirma patrones coherentes con el churn, por ejemplo:

- mayor churn en determinados tipos de suscripción
- relación entre abandono y menor uso / mayor skip rate
- influencia de pausas de suscripción y consultas a atención al cliente
- diferencias geográficas por estado o región

## Modelos comparados

Se compararon tres modelos principales:

- Logistic Regression
- Random Forest
- Deep Neural Network para datos tabulares

La comparación se hizo con validación cruzada estratificada y ajuste de hiperparámetros con `RandomizedSearchCV`.

La métrica principal de selección fue `ROC AUC`, complementada por:

- accuracy
- precision
- recall
- f1-score

## Modelo final

El mejor modelo del proyecto fue **Random Forest Tuned**.

Además de su uso en la página de Modelos, también es el modelo que alimenta la simulación de clientes en la app.

Los artefactos principales del Random Forest final son:

- `models/rf_tuned_model_compressed.joblib`
- `data/exports/rf_tuned_metrics.csv`
- `data/exports/rf_tuned_confusion_matrix_percentage.csv`
- `data/exports/rf_tuned_feature_importance.csv`
- `data/exports/rf_tuned_validation_predictions.csv`

## Aplicación en Streamlit

La aplicación está organizada en tres páginas:

### 1. EDA y Segmentos
Incluye:
- resumen general del dataset
- análisis por grupos de clientes
- análisis geográfico
- distribuciones numéricas
- tabla de resumen por segmentos generada en R

### 2. Modelos
Incluye:
- comparación global de modelos
- métricas del modelo seleccionado
- matriz de confusión
- importancia de variables o coeficientes
- arquitectura de la DNN
- clientes activos con mayor riesgo
- resumen asistido
- entrenamiento in situ de un árbol de decisión sencillo

### 3. Simulación
Incluye:
- simulación de cliente aleatorio
- entrada manual de cliente
- pestaña específica del Random Forest final
- métricas del modelo final
- matriz de confusión
- importancias del RF
- historial de simulaciones
- filtro por nivel de riesgo

## Dask, R y Dagster

### Dask
Se integró Dask para la carga del dataset principal y para estructurar mejor la preparación inicial de datos en la app y en funciones reutilizables.

### R
Se utilizó R mediante un script en `scripts/segment_summary.R` para generar un resumen por segmentos a partir de `train_model_ready.csv`. El resultado se exporta en:

- `data/exports/segment_summary_r.csv`

### Dagster
Se incorporó Dagster para orquestar parte del pipeline del proyecto. Actualmente puede:

- generar `train_model_ready.csv`
- ejecutar el resumen por segmentos en R
- reentrenar el Random Forest final
- regenerar artefactos consumidos por la app

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- Dask
- scikit-learn
- TensorFlow / Keras
- SciKeras
- Plotly
- Streamlit
- Dagster
- R
- joblib

## Cómo ejecutar la app en local

### 1. Crear y activar un entorno virtual

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

### 3. Lanzar la app

```bash
streamlit run main.py
```

## Cómo ejecutar Dagster en local

### Opción recomendada (comando único)

Desde la raíz del proyecto:

```bash
./run_dagster_clean.sh
```

Este script:

- limpia estado local de Dagster (`.dagster/`)
- limpia `__pycache__`
- limpia caché de Streamlit (si está disponible)
- configura `PYTHONPATH` y `DAGSTER_HOME`
- lanza `dagster dev -m churn_project.definitions`

### Opción manual (alternativa)

Si prefieres lanzarlo sin script:

```bash
export PYTHONPATH=$(pwd)/src
export DAGSTER_HOME=$(pwd)/.dagster
mkdir -p .dagster
dagster dev -m churn_project.definitions
```

Después abre Dagster en `http://localhost:3000`.

## Notebooks principales

- `01_model_comparison_tuned.ipynb`: comparación de modelos, validación cruzada y tuning
- `02_train_final_model_rf.ipynb`: entrenamiento final del Random Forest
- `EDAinteractivo_final.ipynb`: trabajo exploratorio y preparación del dataset procesado

## Resultados generales

A nivel global, el proyecto muestra que:

- el churn puede modelarse con bastante precisión usando variables de uso, suscripción y comportamiento
- Random Forest fue el modelo con mejor rendimiento final
- los modelos clásicos siguen siendo muy competitivos en datos tabulares
- una app interactiva mejora mucho la comunicación de resultados
- la modularización del proyecto facilita reutilización, mantenimiento y orquestación

## Posibles mejoras futuras

- ampliar el pipeline de Dagster a más artefactos
- incluir reentrenamiento programado
- incorporar más control de versiones sobre datasets y modelos
- añadir monitorización del rendimiento del modelo
- mejorar la capa de explicación para usuarios de negocio

## Autor

Proyecto realizado por Manuel Triviño como trabajo final académico sobre análisis y predicción de churn en una plataforma de streaming.
