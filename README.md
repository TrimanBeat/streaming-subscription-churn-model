# Streaming Subscription Churn Prediction

Este proyecto analiza el abandono de clientes en una plataforma de streaming y desarrolla un modelo de machine learning para predecir el riesgo de churn a partir de variables de comportamiento, tipo de suscripción y uso del servicio.

Además del análisis y modelado, el proyecto incluye una aplicación en Streamlit para visualizar resultados, comparar modelos y simular predicciones de nuevos clientes.

## Objetivo del proyecto

El objetivo principal es entender qué factores están más relacionados con el churn y construir un modelo capaz de estimar qué clientes tienen mayor riesgo de abandonar la plataforma.

A nivel práctico, el proyecto busca cubrir todo el flujo de trabajo de un proyecto de datos:

- carga y preparación de datos
- análisis exploratorio
- feature engineering
- comparación de modelos
- ajuste de hiperparámetros
- evaluación final
- despliegue de resultados en una aplicación web

## Dataset

El dataset contiene información de usuarios de una plataforma de streaming, con variables relacionadas con:

- tipo de suscripción
- uso semanal de la plataforma
- canciones reproducidas
- skip rate
- tiempo medio de sesión
- pausas de suscripción
- incidencias con atención al cliente
- variables demográficas y de actividad

La variable objetivo del proyecto es:

- `churned`: indica si el cliente abandonó la plataforma o no

## Estructura del proyecto

```text
streaming-subscription-churn-model/
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_EDA_y_segmentos.py
│       ├── 2_Modelos.py
│       └── 3_Simulacion.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── exports/
├── models/
├── notebooks/
├── scripts/
├── src/
│   └── churn_project/
├── requirements.txt
├── README.md
└── .gitignore
```

## Análisis exploratorio

En la fase de EDA se analizaron patrones generales de churn y diferencias entre perfiles de usuarios. Algunos de los puntos más relevantes fueron:

- los usuarios con suscripción `Free` presentan mayor riesgo de churn
- ciertas señales de comportamiento, como menos horas de uso o mayor skip rate, aparecen asociadas a mayor riesgo
- las incidencias con atención al cliente y las pausas de suscripción también tienen relación con el abandono
- se construyeron visualizaciones interactivas para explorar churn por segmento, variables numéricas y localización geográfica

La aplicación de Streamlit incluye una página específica de EDA con filtros interactivos y visualizaciones segmentadas.

## Feature engineering

Durante la preparación de datos se generaron variables derivadas para enriquecer el análisis y mejorar el modelado. Entre ellas:

- agrupaciones de edad
- variables binned de uso semanal
- reglas de segmentación de riesgo
- variables geográficas como `state_code` y `region`
- columnas derivadas para facilitar visualización y simulación

Estas transformaciones se aplicaron de forma estructurada para mantener un flujo reproducible y coherente con el modelado posterior.

## Modelos comparados

Se compararon tres modelos principales mediante validación cruzada estratificada y ajuste de hiperparámetros con `RandomizedSearchCV`:

- Logistic Regression
- Random Forest
- Deep Neural Network para datos tabulares

La comparación se hizo usando como métrica principal `ROC AUC`, ya que permite evaluar la capacidad del modelo para discriminar entre churn y no churn de forma global, sin depender de un único umbral de clasificación.

También se analizaron otras métricas complementarias como:

- accuracy
- precision
- recall
- f1-score

## Modelo final

Tras la comparación de modelos, el mejor rendimiento se obtuvo con **Random Forest Tuned**, por lo que se seleccionó como modelo final del proyecto.

La regresión logística se mantuvo como baseline interpretable y como opción ligera para ciertas partes de la app. La red neuronal se incluyó como parte del bloque de deep learning, aunque en este caso no superó a los modelos clásicos de datos tabulares.

## Aplicación en Streamlit

El proyecto incluye una aplicación desarrollada en Streamlit con varias páginas:

### 1. EDA y Segmentos
Permite explorar el dataset con filtros interactivos, mapas, boxplots y análisis por grupos de clientes.

### 2. Modelos
Incluye:
- comparación global de modelos
- visualización de métricas
- matriz de confusión
- importancia de variables
- clientes activos con mayor riesgo
- un bloque de entrenamiento in situ con un árbol de decisión sencillo

### 3. Simulación
Permite simular nuevos clientes y estimar su riesgo de churn desde la propia aplicación.

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- scikit-learn
- TensorFlow / Keras
- SciKeras
- Plotly
- Streamlit
- joblib

## Cómo ejecutar el proyecto en local

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
python -m streamlit run app/streamlit_app.py
```

## Notebooks principales

Los notebooks más importantes del proyecto son:

- `01_model_comparison_tuned.ipynb`: comparación de modelos con validación cruzada y tuning
- `02_train_final_model_rf.ipynb`: entrenamiento final del Random Forest seleccionado como modelo definitivo

## Resultados del proyecto

A nivel general, el proyecto muestra que:

- el churn puede modelarse con bastante precisión usando variables de comportamiento y tipo de suscripción
- los modelos clásicos para datos tabulares siguen funcionando muy bien en este problema
- Random Forest ofrece el mejor equilibrio entre rendimiento y capacidad de interpretación
- una app interactiva permite comunicar mucho mejor los resultados que un notebook aislado

## Posibles mejoras futuras

Algunas mejoras que se podrían añadir más adelante son:

- integrar Dagster para automatizar el pipeline
- incorporar interoperabilidad con R dentro del flujo del proyecto
- mejorar la simulación de entrada de nuevos clientes
- desplegar una versión final optimizada del modelo ganador en la app
- añadir monitorización o reentrenamiento periódico

## Autor

Proyecto realizado por Manuel Triviño como trabajo de modelización y análisis de churn dentro de un proyecto académico de ciencia de datos.
