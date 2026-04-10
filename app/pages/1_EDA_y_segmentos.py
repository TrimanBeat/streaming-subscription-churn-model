import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 EDA y Segmentos")
st.markdown("""
Explora los principales patrones del churn mediante filtros interactivos,
segmentación visual y una vista geográfica de EE. UU.
""")

# =========================
# Carga de datos
# =========================
@st.cache_data
def load_data():
    segment_summary = pd.read_csv("data/exports/segment_summary.csv")
    train_model_ready = pd.read_csv("data/processed/train_model_ready.csv")
    return segment_summary, train_model_ready

segment_summary, train_model_ready = load_data()
df = train_model_ready.copy()

# =========================
# Preparación
# =========================
if "churned" in df.columns:
    df["churn_label"] = df["churned"].map({0: "No Churn", 1: "Churn"})

numeric_candidates = [
    col for col in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    if col != "churned"
]

# =========================
# Sidebar
# =========================
st.sidebar.header("Filtros")

view_mode = st.sidebar.selectbox(
    "Vista de clientes",
    ["Ambos", "Solo Churn", "Solo No Churn"]
)

subscription_filter = st.sidebar.multiselect(
    "Tipo de suscripción",
    sorted(df["subscription_type"].dropna().unique().tolist()) if "subscription_type" in df.columns else [],
    default=sorted(df["subscription_type"].dropna().unique().tolist()) if "subscription_type" in df.columns else []
)

inquiries_filter = st.sidebar.multiselect(
    "Customer service inquiries",
    sorted(df["customer_service_inquiries"].dropna().unique().tolist()) if "customer_service_inquiries" in df.columns else [],
    default=sorted(df["customer_service_inquiries"].dropna().unique().tolist()) if "customer_service_inquiries" in df.columns else []
)

numeric_var = st.sidebar.selectbox(
    "Variable numérica para boxplot",
    numeric_candidates if numeric_candidates else ["age"]
)

map_mode = st.sidebar.selectbox(
    "Mapa geográfico",
    ["Distribución de clientes", "Tasa de churn por estado"]
)

# =========================
# Filtros
# =========================
filtered_df = df.copy()

if view_mode == "Solo Churn":
    filtered_df = filtered_df[filtered_df["churned"] == 1]
elif view_mode == "Solo No Churn":
    filtered_df = filtered_df[filtered_df["churned"] == 0]

if "subscription_type" in filtered_df.columns and subscription_filter:
    filtered_df = filtered_df[filtered_df["subscription_type"].isin(subscription_filter)]

if "customer_service_inquiries" in filtered_df.columns and inquiries_filter:
    filtered_df = filtered_df[filtered_df["customer_service_inquiries"].isin(inquiries_filter)]

# =========================
# KPIs
# =========================
st.subheader("Resumen")

n_customers = len(filtered_df)
global_churn_rate = filtered_df["churned"].mean() if len(filtered_df) > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Clientes", f"{n_customers:,}")

with col2:
    st.metric("Tasa de churn", f"{global_churn_rate:.2%}")

with col3:
    if "weekly_hours" in filtered_df.columns and len(filtered_df) > 0:
        st.metric("Weekly hours medias", f"{filtered_df['weekly_hours'].mean():.1f}")
    else:
        st.metric("Weekly hours medias", "N/A")

with col4:
    if "song_skip_rate" in filtered_df.columns and len(filtered_df) > 0:
        st.metric("Skip rate medio", f"{filtered_df['song_skip_rate'].mean():.2f}")
    else:
        st.metric("Skip rate medio", "N/A")

if len(filtered_df) == 0:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Segmentos", "Geografía", "Distribuciones"])

# ---------------------------------
# TAB 1: Segmentos
# ---------------------------------
with tab1:
    st.subheader("Segmentación de churn")

    segment_filtered = (
        filtered_df
        .groupby(["subscription_type", "customer_service_inquiries"], as_index=False)
        .agg(
            churn_rate=("churned", "mean"),
            avg_weekly_hours=("weekly_hours", "mean"),
            avg_skip_rate=("song_skip_rate", "mean"),
            avg_pauses=("num_subscription_pauses", "mean"),
            n_customers=("churned", "size")
        )
        .sort_values("churn_rate", ascending=False)
    )

    seg_col1, seg_col2 = st.columns([1.2, 1])

    with seg_col1:
        fig_segments = px.bar(
            segment_filtered,
            x="subscription_type",
            y="churn_rate",
            color="customer_service_inquiries",
            barmode="group",
            title="Churn rate por tipo de suscripción e incidencias",
            labels={
                "subscription_type": "Tipo de suscripción",
                "churn_rate": "Tasa de churn",
                "customer_service_inquiries": "Incidencias"
            }
        )
        fig_segments.update_layout(title_x=0.5)
        st.plotly_chart(fig_segments, use_container_width=True)

    with seg_col2:
        heatmap_df = (
            filtered_df
            .groupby(["subscription_type", "customer_service_inquiries"])["churned"]
            .mean()
            .reset_index()
            .pivot(index="subscription_type", columns="customer_service_inquiries", values="churned")
        )

        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Reds",
            title="Heatmap de churn por segmento"
        )
        fig_heatmap.update_traces(texttemplate="%{z:.2%}")
        fig_heatmap.update_layout(title_x=0.5)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("### Tabla de segmentos")
    nice_segments = segment_filtered.copy()
    nice_segments["churn_rate"] = nice_segments["churn_rate"].map(lambda x: f"{x:.2%}")
    nice_segments["avg_weekly_hours"] = nice_segments["avg_weekly_hours"].map(lambda x: f"{x:.1f}")
    nice_segments["avg_skip_rate"] = nice_segments["avg_skip_rate"].map(lambda x: f"{x:.2f}")
    nice_segments["avg_pauses"] = nice_segments["avg_pauses"].map(lambda x: f"{x:.2f}")

    with st.expander("Ver tabla de segmentos"):
        st.dataframe(nice_segments, use_container_width=True, hide_index=True)

# ---------------------------------
# TAB 2: Geografía
# ---------------------------------
with tab2:
    st.subheader("Mapa geográfico de EE. UU.")

    if "state_code" not in filtered_df.columns or "location" not in filtered_df.columns:
        st.warning("No se encontraron las columnas `location` y `state_code`.")
    else:
        if map_mode == "Distribución de clientes":
            map_df = (
                filtered_df
                .groupby(["location", "state_code"])
                .size()
                .reset_index(name="n_customers")
            )

            fig_map = px.choropleth(
                map_df,
                locations="state_code",
                locationmode="USA-states",
                color="n_customers",
                scope="usa",
                color_continuous_scale="Burg",
                hover_name="location",
                hover_data={"state_code": False, "n_customers": True},
                title="Distribución de clientes por estado"
            )

            fig_map.update_layout(
                title_x=0.5,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=60, b=20),
                coloraxis_colorbar_title="Clientes"
            )
        else:
            map_df = (
                filtered_df
                .groupby(["location", "state_code"])["churned"]
                .mean()
                .reset_index(name="churn_rate")
            )

            fig_map = px.choropleth(
                map_df,
                locations="state_code",
                locationmode="USA-states",
                color="churn_rate",
                scope="usa",
                color_continuous_scale="Reds",
                hover_name="location",
                hover_data={"state_code": False, "churn_rate": ":.2%"},
                title="Tasa de churn por estado"
            )

            fig_map.update_layout(
                title_x=0.5,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=60, b=20),
                coloraxis_colorbar_title="Churn rate"
            )

        fig_map.update_geos(
            bgcolor="rgba(0,0,0,0)",
            showland=True,
            landcolor="rgb(245,245,245)"
        )

        st.plotly_chart(fig_map, use_container_width=True)

        with st.expander("Ver tabla geográfica"):
            st.dataframe(map_df.sort_values(map_df.columns[-1], ascending=False), use_container_width=True, hide_index=True)

# ---------------------------------
# TAB 3: Distribuciones
# ---------------------------------
with tab3:
    st.subheader("Distribución de variables numéricas")

    compare_df = df.copy()

    if "subscription_type" in compare_df.columns and subscription_filter:
        compare_df = compare_df[compare_df["subscription_type"].isin(subscription_filter)]

    if "customer_service_inquiries" in compare_df.columns and inquiries_filter:
        compare_df = compare_df[compare_df["customer_service_inquiries"].isin(inquiries_filter)]

    dist_col1, dist_col2 = st.columns([1.1, 0.9])

    with dist_col1:
        fig_box = px.box(
            compare_df,
            x="churn_label",
            y=numeric_var,
            color="churn_label",
            title=f"{numeric_var} según churn vs no churn",
            labels={"churn_label": "Grupo", numeric_var: numeric_var}
        )
        fig_box.update_layout(title_x=0.5, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with dist_col2:
        hist_df = filtered_df.copy()
        fig_hist = px.histogram(
            hist_df,
            x=numeric_var,
            color="churn_label" if view_mode == "Ambos" and "churn_label" in hist_df.columns else None,
            barmode="overlay",
            opacity=0.65,
            nbins=30,
            title=f"Distribución de {numeric_var}"
        )
        fig_hist.update_layout(title_x=0.5)
        st.plotly_chart(fig_hist, use_container_width=True)

    stats_df = (
        compare_df.groupby("churn_label")[numeric_var]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    st.markdown("### Estadísticos rápidos")
    with st.expander("Ver tabla de estadísticos"):
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

# =========================
# Vista previa del dataset
# =========================
st.subheader("Vista previa del dataset filtrado")
with st.expander("Ver muestra del dataset"):
    st.dataframe(filtered_df.head(30), use_container_width=True, hide_index=True)