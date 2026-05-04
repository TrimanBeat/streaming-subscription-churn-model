from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="../assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_path = Path("app/styles.css")
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.sidebar.image("../assets/logo.png", use_container_width=True)
st.sidebar.markdown("## Churn Analytics Dashboard")
st.sidebar.caption("Predicción y análisis de abandono de clientes")

st.title("Churn Analytics Dashboard")
st.markdown(
    "Aplicación interactiva para analizar patrones de abandono, comparar modelos predictivos y simular riesgo de churn en nuevos clientes."
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### Problema")
    st.write(
        "Identificar clientes con mayor probabilidad de abandono y entender qué variables "
        "están más relacionadas con ese comportamiento."
    )

with c2:
    st.markdown("### Enfoque")
    st.write(
        "El proyecto combina análisis exploratorio, feature engineering, comparación de modelos "
        "y despliegue en una app interactiva."
    )

with c3:
    st.markdown("### Qué incluye")
    st.write(
        "EDA interactivo, comparación de modelos, simulación de clientes y visualización de resultados "
        "para apoyar decisiones de negocio."
    )

st.markdown("---")
st.markdown("### Navegación")
st.write(
    "Usa el menú lateral para explorar las secciones de EDA, modelos y simulación."
)