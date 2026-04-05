import streamlit as st

st.set_page_config(
    page_title="Churn Prediction Project",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Streaming Subscription Churn Project")

st.markdown("""
## Objetivo del proyecto
Este proyecto analiza el churn de usuarios en una plataforma de streaming.

La aplicación incluye:
- análisis exploratorio de datos
- segmentación de usuarios
- comparación de modelos predictivos
- simulación en vivo de nuevos clientes
""")

st.subheader("Resumen")
st.write("""
El objetivo es identificar patrones asociados al abandono de usuarios y construir un modelo
capaz de predecir el riesgo de churn.
""")

st.info("Usa el menú lateral para navegar entre las distintas secciones de la app.")