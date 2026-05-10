
import streamlit as st
import pandas as pd
import plotly.express as px

st.subheader("EDA y Segmentos")
st.markdown("""
Explora los principales patrones del churn mediante filtros interactivos,
segmentación visual y una vista geográfica de EE. UU.
""")

# =========================================================
# Paleta y helpers visuales
# =========================================================
NAVY = "#0F172A"
RED = "#B42318"
SOFT_RED = "#F04438"
BLUE = "#1D4ED8"
SOFT_BLUE = "#93C5FD"
GRAY = "#475467"
LIGHT_BG = "#FFFFFF"
GRID = "#D0D5DD"

CHURN_COLOR_MAP = {
    "Churn": RED,
    "No Churn": BLUE
}

INQUIRIES_COLOR_MAP = {
    "High": RED,
    "Medium": "#F79009",
    "Low": BLUE
}

def apply_plot_style(fig):
    fig.update_layout(
        title_text="",
        paper_bgcolor=LIGHT_BG,
        plot_bgcolor=LIGHT_BG,
        font=dict(color=NAVY),
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title_font=dict(color=NAVY),
            tickfont=dict(color=NAVY)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            title_font=dict(color=NAVY),
            tickfont=dict(color=NAVY)
        ),
        legend=dict(title_font=dict(color=NAVY), font=dict(color=NAVY)),
        coloraxis_colorbar=dict(title_font=dict(color=NAVY), tickfont=dict(color=NAVY))
    )
    return fig

# =========================================================
# Diccionarios geográficos
# =========================================================
STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"
}

REGION_MAP = {
    "Connecticut": "Northeast", "Maine": "Northeast", "Massachusetts": "Northeast",
    "New Hampshire": "Northeast", "Rhode Island": "Northeast", "Vermont": "Northeast",
    "New Jersey": "Northeast", "New York": "Northeast", "Pennsylvania": "Northeast",

    "Illinois": "Midwest", "Indiana": "Midwest", "Michigan": "Midwest",
    "Ohio": "Midwest", "Wisconsin": "Midwest", "Iowa": "Midwest",
    "Kansas": "Midwest", "Minnesota": "Midwest", "Missouri": "Midwest",
    "Nebraska": "Midwest", "North Dakota": "Midwest", "South Dakota": "Midwest",

    "Delaware": "South", "Florida": "South", "Georgia": "South",
    "Maryland": "South", "North Carolina": "South", "South Carolina": "South",
    "Virginia": "South", "West Virginia": "South", "Alabama": "South",
    "Kentucky": "South", "Mississippi": "South", "Tennessee": "South",
    "Arkansas": "South", "Louisiana": "South", "Oklahoma": "South", "Texas": "South",

    "Arizona": "West", "Colorado": "West", "Idaho": "West", "Montana": "West",
    "Nevada": "West", "New Mexico": "West", "Utah": "West", "Wyoming": "West",
    "Alaska": "West", "California": "West", "Hawaii": "West", "Oregon": "West",
    "Washington": "West"
}

REGION_COORDS = {
    "West": {"lat": 39, "lon": -118},
    "Midwest": {"lat": 41, "lon": -93},
    "South": {"lat": 33, "lon": -86},
    "Northeast": {"lat": 42, "lon": -74}
}

# =========================================================
# Carga de datos
# =========================================================
@st.cache_data
def load_data():
    segment_summary = pd.read_csv("data/exports/segment_summary.csv")
    train_model_ready = pd.read_csv("data/processed/train_model_ready.csv")
    return segment_summary, train_model_ready

segment_summary, train_model_ready = load_data()
df = train_model_ready.copy()

# =========================================================
# Preparación robusta para geografía y labels
# =========================================================
if "location" in df.columns:
    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"Nebrasksa": "Nebraska"})
    )

if "state_code" not in df.columns and "location" in df.columns:
    df["state_code"] = df["location"].map(STATE_ABBREV)

if "region" not in df.columns and "location" in df.columns:
    df["region"] = df["location"].map(REGION_MAP)

if "churned" in df.columns:
    df["churn_label"] = df["churned"].map({0: "No Churn", 1: "Churn"})

# =========================================================
# Variables numéricas disponibles
# =========================================================
numeric_candidates = [
    col for col in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    if col != "churned"
]

preferred_order = [
    "weekly_hours",
    "song_skip_rate",
    "age",
    "num_subscription_pauses",
    "average_session_length",
    "weekly_unique_songs",
    "weekly_songs_played",
    "notifications_clicked",
]
numeric_candidates = [c for c in preferred_order if c in numeric_candidates] + [
    c for c in numeric_candidates if c not in preferred_order
]

# =========================================================
# Sidebar filtros
# =========================================================
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

geo_mode = st.sidebar.selectbox(
    "Visualización geográfica",
    [
        "Clientes por estado",
        "Churn por estado",
        "Clientes por región",
        "Churn por región"
    ]
)

# =========================================================
# Aplicar filtros
# =========================================================
filtered_df = df.copy()

if view_mode == "Solo Churn":
    filtered_df = filtered_df[filtered_df["churned"] == 1]
elif view_mode == "Solo No Churn":
    filtered_df = filtered_df[filtered_df["churned"] == 0]

if "subscription_type" in filtered_df.columns and subscription_filter:
    filtered_df = filtered_df[filtered_df["subscription_type"].isin(subscription_filter)]

if "customer_service_inquiries" in filtered_df.columns and inquiries_filter:
    filtered_df = filtered_df[filtered_df["customer_service_inquiries"].isin(inquiries_filter)]

if len(filtered_df) == 0:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# =========================================================
# Funciones de gráficos
# =========================================================
def plot_us_state_choropleth(data: pd.DataFrame, mode: str):
    if "state_code" not in data.columns or "location" not in data.columns:
        return None

    state_df = data.dropna(subset=["state_code"]).copy()
    if len(state_df) == 0:
        return None

    if mode == "Clientes por estado":
        plot_df = (
            state_df.groupby(["location", "state_code"])
            .size()
            .reset_index(name="value")
        )
        color_scale = "Blues"
        hover_fmt = {"state_code": False, "value": True}
        colorbar_title = "Clientes"

    else:
        plot_df = (
            state_df.groupby(["location", "state_code"])["churned"]
            .mean()
            .reset_index(name="value")
        )
        color_scale = "Reds"
        hover_fmt = {"state_code": False, "value": ":.2%"}
        colorbar_title = "Churn rate"

    fig = px.choropleth(
        plot_df,
        locations="state_code",
        locationmode="USA-states",
        color="value",
        scope="usa",
        color_continuous_scale=color_scale,
        hover_name="location",
        hover_data=hover_fmt
    )

    fig.update_layout(
        title_text="",
        paper_bgcolor=LIGHT_BG,
        plot_bgcolor=LIGHT_BG,
        margin=dict(l=20, r=20, t=10, b=20),
        coloraxis_colorbar_title=colorbar_title
    )

    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        showland=True,
        landcolor="#F2F4F7"
    )

    return fig, plot_df


def plot_region_bubble_map(data: pd.DataFrame, mode: str):
    if "region" not in data.columns:
        return None

    region_df = data.dropna(subset=["region"]).copy()
    if len(region_df) == 0:
        return None

    if mode == "Clientes por región":
        plot_df = (
            region_df.groupby("region")
            .size()
            .reset_index(name="value")
        )
        color_scale = "Blues"
    else:
        plot_df = (
            region_df.groupby("region")["churned"]
            .mean()
            .reset_index(name="value")
        )
        color_scale = "Reds"

    plot_df["lat"] = plot_df["region"].map(lambda x: REGION_COORDS.get(x, {}).get("lat"))
    plot_df["lon"] = plot_df["region"].map(lambda x: REGION_COORDS.get(x, {}).get("lon"))

    fig = px.scatter_geo(
        plot_df,
        lat="lat",
        lon="lon",
        size="value",
        color="value",
        hover_name="region",
        scope="usa",
        color_continuous_scale=color_scale,
    )

    fig.update_layout(
        title_text="",
        paper_bgcolor=LIGHT_BG,
        plot_bgcolor=LIGHT_BG,
        margin=dict(l=20, r=20, t=10, b=20)
    )

    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        showland=True,
        landcolor="#F2F4F7"
    )

    return fig, plot_df

# =========================================================
# KPIs
# =========================================================
st.subheader("Resumen")

n_customers = len(filtered_df)
global_churn_rate = filtered_df["churned"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Clientes", f"{n_customers:,}")

with col2:
    st.metric("Tasa de churn", f"{global_churn_rate:.2%}")

with col3:
    if "weekly_hours" in filtered_df.columns:
        st.metric("Weekly hours medias", f"{filtered_df['weekly_hours'].mean():.1f}")
    else:
        st.metric("Weekly hours medias", "N/A")

with col4:
    if "song_skip_rate" in filtered_df.columns:
        st.metric("Skip rate medio", f"{filtered_df['song_skip_rate'].mean():.2f}")
    else:
        st.metric("Skip rate medio", "N/A")

# =========================================================
# Tabs principales
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Segmentos", "Geografía", "Distribuciones"])

# ---------------------------------------------------------
# TAB 1: Resumen
# ---------------------------------------------------------
with tab1:
    top_left, top_right = st.columns([1.1, 1])

    with top_left:
        if "subscription_type" in filtered_df.columns:
            churn_by_plan = (
                filtered_df.groupby("subscription_type")["churned"]
                .mean()
                .reset_index(name="churn_rate")
                .sort_values("churn_rate", ascending=False)
            )

            st.markdown("#### Tasa de churn por tipo de suscripción")
            fig_plan = px.bar(
                churn_by_plan,
                x="subscription_type",
                y="churn_rate",
                text="churn_rate",
                color="churn_rate",
                color_continuous_scale=[[0, SOFT_BLUE], [1, RED]]
            )
            fig_plan.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            apply_plot_style(fig_plan)
            fig_plan.update_layout(xaxis_title="Tipo de suscripción", yaxis_title="Tasa de churn")
            st.plotly_chart(fig_plan, use_container_width=True)

    with top_right:
        if "customer_service_inquiries" in filtered_df.columns:
            churn_by_inquiries = (
                filtered_df.groupby("customer_service_inquiries")["churned"]
                .mean()
                .reset_index(name="churn_rate")
                .sort_values("churn_rate", ascending=False)
            )

            st.markdown("#### Tasa de churn por incidencias")
            fig_inquiries = px.bar(
                churn_by_inquiries,
                x="customer_service_inquiries",
                y="churn_rate",
                text="churn_rate",
                color="customer_service_inquiries",
                color_discrete_map=INQUIRIES_COLOR_MAP
            )
            fig_inquiries.update_traces(texttemplate="%{y:.2%}", textposition="outside")
            apply_plot_style(fig_inquiries)
            fig_inquiries.update_layout(xaxis_title="Incidencias", yaxis_title="Tasa de churn")
            st.plotly_chart(fig_inquiries, use_container_width=True)

    bottom_left, bottom_right = st.columns([1.1, 1])

    with bottom_left:
        if "weekly_hours" in df.columns:
            compare_df = df.copy()
            if subscription_filter:
                compare_df = compare_df[compare_df["subscription_type"].isin(subscription_filter)]
            if inquiries_filter:
                compare_df = compare_df[compare_df["customer_service_inquiries"].isin(inquiries_filter)]

            st.markdown("#### Weekly hours según churn vs no churn")
            fig_weekly = px.box(
                compare_df,
                x="churn_label",
                y="weekly_hours",
                color="churn_label",
                color_discrete_map=CHURN_COLOR_MAP
            )
            apply_plot_style(fig_weekly)
            fig_weekly.update_layout(showlegend=False, xaxis_title="Grupo", yaxis_title="Weekly hours")
            st.plotly_chart(fig_weekly, use_container_width=True)

    with bottom_right:
        if "song_skip_rate" in df.columns:
            compare_df = df.copy()
            if subscription_filter:
                compare_df = compare_df[compare_df["subscription_type"].isin(subscription_filter)]
            if inquiries_filter:
                compare_df = compare_df[compare_df["customer_service_inquiries"].isin(inquiries_filter)]

            st.markdown("#### Song skip rate según churn vs no churn")
            fig_skip = px.box(
                compare_df,
                x="churn_label",
                y="song_skip_rate",
                color="churn_label",
                color_discrete_map=CHURN_COLOR_MAP
            )
            apply_plot_style(fig_skip)
            fig_skip.update_layout(showlegend=False, xaxis_title="Grupo", yaxis_title="Song skip rate")
            st.plotly_chart(fig_skip, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Segmentos
# ---------------------------------------------------------
with tab2:
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
        st.markdown("#### Churn rate por tipo de suscripción e incidencias")
        fig_segments = px.bar(
            segment_filtered,
            x="subscription_type",
            y="churn_rate",
            color="customer_service_inquiries",
            barmode="group",
            labels={
                "subscription_type": "Tipo de suscripción",
                "churn_rate": "Tasa de churn",
                "customer_service_inquiries": "Incidencias"
            },
            color_discrete_map=INQUIRIES_COLOR_MAP
        )
        apply_plot_style(fig_segments)
        st.plotly_chart(fig_segments, use_container_width=True)

    with seg_col2:
        heatmap_df = (
            filtered_df
            .groupby(["subscription_type", "customer_service_inquiries"])["churned"]
            .mean()
            .reset_index()
            .pivot(index="subscription_type", columns="customer_service_inquiries", values="churned")
        )

        st.markdown("#### Heatmap de churn por segmento")
        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=[[0, SOFT_BLUE], [1, RED]]
        )
        fig_heatmap.update_traces(texttemplate="%{z:.2%}")
        fig_heatmap.update_layout(
            title_text="",
            paper_bgcolor=LIGHT_BG,
            plot_bgcolor=LIGHT_BG,
            font=dict(color=NAVY),
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("### Tabla de segmentos")
    nice_segments = segment_filtered.copy()
    nice_segments["churn_rate"] = nice_segments["churn_rate"].map(lambda x: f"{x:.2%}")
    nice_segments["avg_weekly_hours"] = nice_segments["avg_weekly_hours"].map(lambda x: f"{x:.1f}")
    nice_segments["avg_skip_rate"] = nice_segments["avg_skip_rate"].map(lambda x: f"{x:.2f}")
    nice_segments["avg_pauses"] = nice_segments["avg_pauses"].map(lambda x: f"{x:.2f}")

    with st.expander("Ver tabla de segmentos"):
        st.dataframe(nice_segments, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 3: Geografía
# ---------------------------------------------------------
with tab3:
    st.subheader("Vista geográfica")

    if geo_mode in ["Clientes por estado", "Churn por estado"]:
        result = plot_us_state_choropleth(filtered_df, geo_mode)
    else:
        result = plot_region_bubble_map(filtered_df, geo_mode)

    if result is None:
        st.warning("No se pudo generar la visualización geográfica con los datos disponibles.")
    else:
        fig_geo, geo_df = result
        st.markdown(f"#### {geo_mode}")
        st.plotly_chart(fig_geo, use_container_width=True)

        with st.expander("Ver tabla geográfica"):
            st.dataframe(geo_df.sort_values("value", ascending=False), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 4: Distribuciones
# ---------------------------------------------------------
with tab4:
    compare_df = df.copy()

    if subscription_filter:
        compare_df = compare_df[compare_df["subscription_type"].isin(subscription_filter)]
    if inquiries_filter:
        compare_df = compare_df[compare_df["customer_service_inquiries"].isin(inquiries_filter)]

    dist_col1, dist_col2 = st.columns([1.1, 0.9])

    with dist_col1:
        st.markdown(f"#### {numeric_var} según churn vs no churn")
        fig_box = px.box(
            compare_df,
            x="churn_label",
            y=numeric_var,
            color="churn_label",
            labels={"churn_label": "Grupo", numeric_var: numeric_var},
            color_discrete_map=CHURN_COLOR_MAP
        )
        apply_plot_style(fig_box)
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with dist_col2:
        hist_df = filtered_df.copy()
        st.markdown(f"#### Distribución de {numeric_var}")
        fig_hist = px.histogram(
            hist_df,
            x=numeric_var,
            color="churn_label" if view_mode == "Ambos" and "churn_label" in hist_df.columns else None,
            barmode="overlay",
            opacity=0.65,
            nbins=30,
            color_discrete_map=CHURN_COLOR_MAP
        )
        apply_plot_style(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    if "age_group" in filtered_df.columns:
        age_df = (
            filtered_df.groupby("age_group", observed=False)["churned"]
            .mean()
            .reset_index(name="churn_rate")
        )

        st.markdown("#### Tasa de churn por grupo de edad")
        fig_age = px.bar(
            age_df,
            x="age_group",
            y="churn_rate",
            text="churn_rate",
            color="churn_rate",
            color_continuous_scale=[[0, SOFT_BLUE], [1, RED]]
        )
        fig_age.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        apply_plot_style(fig_age)
        fig_age.update_layout(xaxis_title="Grupo de edad", yaxis_title="Tasa de churn")
        st.plotly_chart(fig_age, use_container_width=True)

    stats_df = (
        compare_df.groupby("churn_label")[numeric_var]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    with st.expander("Ver tabla de estadísticos"):
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

# =========================================================
# Vista previa del dataset
# =========================================================
st.subheader("Vista previa del dataset filtrado")
with st.expander("Ver muestra del dataset"):
    st.dataframe(filtered_df.head(30), use_container_width=True, hide_index=True)
