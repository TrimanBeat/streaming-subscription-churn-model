# 🎧 Streaming Subscription Churn Prediction

## 🇪🇸 Español

### Descripción general

Este es un proyecto end-to-end de **data science** centrado en la **predicción de churn** en una plataforma de streaming musical y en la traducción de los resultados a un **dashboard interactivo** construido con **Streamlit**.

El churn es uno de los problemas de negocio más relevantes en servicios basados en suscripción. En este proyecto analicé un dataset de usuarios de una plataforma de streaming, identifiqué los principales patrones asociados al abandono, entrené varios modelos predictivos y construí una aplicación interactiva para comunicar los resultados de forma clara y visual.

---

### Objetivos del proyecto

Los objetivos principales fueron:

- entender qué comportamientos y segmentos de usuarios están más asociados al churn
- construir un modelo capaz de estimar la probabilidad de abandono
- comparar un baseline interpretable con un modelo ensemble más potente
- presentar los resultados mediante una app interactiva con formato de dashboard
- simular nuevos clientes y estimar su riesgo de churn en tiempo real

---

### Dataset

El dataset representa usuarios de una plataforma de streaming por suscripción e incluye variables relacionadas con:

- tipo de suscripción
- engagement del usuario
- comportamiento de escucha
- consultas o incidencias con atención al cliente
- pausas en la suscripción
- localización geográfica
- variable objetivo de churn

La variable objetivo es:

- **`churned`** → indica si el usuario abandonó o no la plataforma

---

### Análisis exploratorio (EDA)

El análisis exploratorio mostró que el churn está impulsado principalmente por tres bloques:

#### 1. Tipo de suscripción
Los usuarios del plan **Free** presentan tasas de churn mucho más altas que los usuarios con planes de pago como **Premium** o **Family**.

#### 2. Engagement
Variables como:

- `weekly_hours`
- `song_skip_rate`
- `weekly_unique_songs`
- `average_session_length`

mostraron relaciones importantes con el churn. Menor uso y mayor tasa de canciones saltadas se asociaron con mayor riesgo de abandono.

#### 3. Fricción / inestabilidad
Variables como:

- `customer_service_inquiries`
- `num_subscription_pauses`

aparecieron como señales claras de posible churn.

El EDA también incluyó una visualización geográfica mediante un **choropleth map** de EE. UU. construido con Plotly.

---

### Creación de variables

Para mejorar la interpretabilidad y el rendimiento del modelo, se crearon nuevas variables como:

- `tenure_days`
- `age_group`
- `weekly_hours_bin`
- `skip_rate_bin`
- `risk_segment_rule`
- `state_code` para visualización geográfica

Estas variables ayudaron a estructurar mejor el análisis y a conectar los resultados del modelo con una lectura más orientada a negocio.

---

### Modelos entrenados

Se entrenaron y compararon dos modelos principales:

#### 1. Logistic Regression
Se utilizó como **baseline** por ser ligera, interpretable y fácil de desplegar.

#### 2. Random Forest
Se utilizó como modelo más potente para capturar relaciones no lineales e interacciones entre variables.

#### Decisión final
- **Random Forest** obtuvo el mejor rendimiento predictivo
- **Logistic Regression** se eligió como modelo de despliegue para la simulación dentro de la app, por ser más ligera y fácil de subir a producción

---

### Evaluación de modelos

Los modelos se evaluaron con:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

El proyecto incluye:

- comparación visual entre modelos
- matrices de confusión normalizadas
- predicciones de validación con probabilidad de churn
- análisis de importancia de variables para Random Forest

---

### Interpretación del modelo

El análisis de importancia de variables en Random Forest mostró que los factores más relevantes están relacionados con:

- engagement
- fricción / inestabilidad
- tipo de suscripción

Esto confirmó bastante bien los hallazgos del análisis exploratorio.

Además, el dashboard incluye una **capa opcional de interpretación asistida mediante LLM** usando **Google Gemini**, que genera un pequeño resumen en lenguaje natural a partir del modelo seleccionado.

> Nota: la importancia de variables se utilizó como herramienta de interpretación predictiva, no como prueba de causalidad.

---

### Dashboard en Streamlit

El proyecto incluye una aplicación multipágina en Streamlit diseñada con mentalidad de dashboard, en una línea parecida a una herramienta tipo BI.

#### Secciones principales

##### 1. EDA y Segmentos
- tabla de segmentos
- churn por tipo de suscripción e incidencias
- mapa coroplético de EE. UU.
- vista previa del dataset procesado

##### 2. Modelos
- selector de modelo
- tarjetas KPI
- comparación entre modelos
- matriz de confusión
- importancia de variables
- tabla de clientes con mayor riesgo
- resumen opcional generado con Gemini

##### 3. Simulación en vivo
- scoring de clientes a partir de una base
- creación manual de perfiles
- estimación del riesgo de churn en tiempo real
- semáforo de riesgo
- historial de simulación y KPIs

---

### Stack tecnológico

- **Python**
- **pandas**
- **NumPy**
- **scikit-learn**
- **Plotly**
- **Streamlit**
- **joblib**
- **Google Gemini API** *(opcional para interpretación automática)*

---

### Estructura del proyecto

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
├── src/
│   └── churn_project/
├── README.md
├── requirements.txt
└── .gitignore
```

---

### Cómo ejecutar el proyecto en local

#### 1. Crear y activar un entorno virtual

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### 2. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

#### 3. Lanzar la app de Streamlit

```bash
python -m streamlit run app/streamlit_app.py
```

---

### Configuración de Gemini (opcional)

Para activar los resúmenes automáticos con LLM:

#### En local

```bash
export GEMINI_API_KEY="tu_api_key"
```

#### En Streamlit Community Cloud

Añadir la clave en **App Settings → Secrets**:

```toml
GEMINI_API_KEY = "tu_api_key"
```

---

### Aprendizajes principales

Este proyecto me ayudó a practicar y reforzar:

- estructuración de proyectos end-to-end
- análisis exploratorio orientado a negocio
- feature engineering en datos tabulares
- comparación entre modelos interpretables y modelos ensemble
- despliegue de resultados mediante dashboards interactivos
- traducción de resultados técnicos a explicaciones en lenguaje natural

---

### Mejoras futuras

Algunas posibles extensiones serían:

- desplegar una versión más ligera de Random Forest en la app
- añadir explicabilidad más avanzada con SHAP
- ampliar la simulación de clientes y retención
- orquestar el pipeline con Dagster
- añadir filtros y drilldowns más avanzados en el dashboard

---

### Autor

**Manuel Triviño**  
Estudiante de Estadística con interés en **Data Analytics**, **Data Science** y en construir proyectos que conecten modelado técnico con comunicación clara de negocio.

---

## 🇬🇧 English

### Overview

This is an end-to-end **data science** project focused on **predicting churn** in a music streaming platform and translating the results into an **interactive dashboard** built with **Streamlit**.

Customer churn is one of the most important business problems in subscription-based services. In this project, I explored a streaming subscription dataset, identified the main behavioral and business patterns behind churn, trained predictive models, and built an interactive app to communicate the results clearly and visually.

---

### Project goals

The main goals were:

- understand which user behaviors and segments are most associated with churn
- build a model capable of estimating churn probability
- compare an interpretable baseline against a stronger ensemble model
- present the results through an interactive dashboard-style app
- simulate new customers and estimate their churn risk in real time

---

### Dataset

The dataset represents users of a subscription-based streaming service and includes variables related to:

- subscription type
- user engagement
- listening behavior
- customer support activity
- subscription pauses
- geographic location
- churn target variable

The target variable is:

- **`churned`** → indicates whether the user churned or not

---

### Exploratory Data Analysis (EDA)

The exploratory analysis showed that churn is mainly driven by three broad dimensions:

#### 1. Subscription type
Users in the **Free** plan showed much higher churn rates than users in paid plans such as **Premium** or **Family**.

#### 2. Engagement
Variables such as:

- `weekly_hours`
- `song_skip_rate`
- `weekly_unique_songs`
- `average_session_length`

showed strong relationships with churn. Lower usage and higher skip behavior were associated with higher churn risk.

#### 3. Friction / instability
Variables such as:

- `customer_service_inquiries`
- `num_subscription_pauses`

appeared as clear warning signals of possible churn.

The EDA also included a geographic visualization using a U.S. **choropleth map** built with Plotly.

---

### Feature engineering

To improve interpretability and model performance, I created additional variables such as:

- `tenure_days`
- `age_group`
- `weekly_hours_bin`
- `skip_rate_bin`
- `risk_segment_rule`
- `state_code` for geographic visualization

These features helped structure the analysis and connect model outputs with more business-oriented explanations.

---

### Models trained

Two main models were trained and compared:

#### 1. Logistic Regression
Used as the **baseline model** because it is lightweight, interpretable, and easy to deploy.

#### 2. Random Forest
Used as a stronger model to capture non-linear patterns and interactions.

#### Final decision
- **Random Forest** achieved the best predictive performance
- **Logistic Regression** was selected as the deployment model for the app simulation because it is lighter and easier to push to production

---

### Model evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

The project includes:

- visual model comparison
- normalized confusion matrices
- validation predictions with churn probabilities
- feature importance analysis for Random Forest

---

### Model interpretation

Random Forest feature importance analysis showed that the most relevant drivers are related to:

- engagement
- friction / instability
- subscription type

This confirmed the main findings from the exploratory analysis.

The dashboard also includes an **optional LLM-assisted interpretation layer** using **Google Gemini**, which generates a short natural-language summary based on the selected model.

> Note: feature importance was used as a predictive interpretation tool, not as proof of causality.

---

### Streamlit dashboard

The project includes a multi-page Streamlit app designed with a dashboard mindset, similar to a BI reporting tool.

#### Main sections

##### 1. EDA & Segments
- segmentation table
- churn by subscription type and customer inquiries
- U.S. choropleth map
- preview of the processed dataset

##### 2. Models
- model selector
- KPI cards
- model comparison
- confusion matrix
- feature importance
- high-risk customer table
- optional Gemini-generated summary

##### 3. Live Simulation
- customer scoring from a base dataset
- manual profile creation
- real-time churn risk estimation
- traffic-light style risk indicator
- simulation history and KPIs

---

### Tech stack

- **Python**
- **pandas**
- **NumPy**
- **scikit-learn**
- **Plotly**
- **Streamlit**
- **joblib**
- **Google Gemini API** *(optional automatic interpretation layer)*

---

### Project structure

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
├── src/
│   └── churn_project/
├── README.md
├── requirements.txt
└── .gitignore
```

---

### How to run the project locally

#### 1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

#### 3. Run the Streamlit app

```bash
python -m streamlit run app/streamlit_app.py
```

---

### Gemini setup (optional)

To enable LLM-generated summaries:

#### Local

```bash
export GEMINI_API_KEY="your_api_key"
```

#### Streamlit Community Cloud

Add the key under **App Settings → Secrets**:

```toml
GEMINI_API_KEY = "your_api_key"
```

---

### Key learnings

This project helped me strengthen skills in:

- end-to-end project structuring
- business-oriented exploratory analysis
- feature engineering for tabular data
- comparing interpretable and ensemble models
- deploying results through interactive dashboards
- translating technical outputs into natural-language explanations

---

### Future improvements

Possible future extensions include:

- deploying a lighter Random Forest version in the app
- adding SHAP-based explainability
- expanding customer simulation and retention logic
- orchestrating the pipeline with Dagster
- adding more advanced filters and drilldowns to the dashboard

---

### Author

**Manuel Triviño**  
Statistics student interested in **Data Analytics**, **Data Science**, and building projects that connect technical modeling with clear business communication.
