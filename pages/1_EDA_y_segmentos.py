from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from churn_project.data_loader import load_data_with_dask, load_export_csv
from churn_project.display_labels import prettify_columns, pretty_label
from churn_project.eda_utils import get_numeric_plot_candidates

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
# Datos base
# =========================================================
df = load_data_with_dask().copy()
segment_summary_r = load_export_csv("data/exports/segment_summary_r.csv").copy()

if "location" in df.columns:
    df["state_code"] = df["location"].map(STATE_ABBREV)
    df["region"] = df["location"].map(REGION_MAP)

if "churned" in df.columns:
    df["churn_label"] = df["churned"].map({1: "Churn", 0: "No Churn"})

numeric_candidates = get_numeric_plot_candidates(df)
if not numeric_candidates:
    numeric_candidates = [c for c in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist() if c != "churned"]

# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("## Filtros")

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
    pretty_label("customer_service_inquiries"),
    sorted(df["customer_service_inquiries"].dropna().unique().tolist()) if "customer_service_inquiries" in df.columns else [],
    default=sorted(df["customer_service_inquiries"].dropna().unique().tolist()) if "customer_service_inquiries" in df.columns else []
)

numeric_var = st.sidebar.selectbox(
    "Variable numérica para boxplot",
    numeric_candidates if numeric_candidates else ["age"],
    format_func=pretty_label
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
    else:
        plot_df = (
            state_df.groupby(["location", "state_code"])["churned"]
            .mean()
            .reset_index(name="value")
        )
        color_scale = [[0, SOFT_BLUE], [1, RED]]

    fig = px.choropleth(
        plot_df,
        locations="state_code",
        locationmode="USA-states",
        color="value",
        hover_name="location",
        scope="usa",
        color_continuous_scale=color_scale,
        labels={"value": "Valor", "state_code": pretty_label("state_code"), "location": pretty_label("location")}
    )
    fig.update_layout(
        title_text="",
        paper_bgcolor=LIGHT_BG,
        plot_bgcolor=LIGHT_BG,
        geo=dict(bgcolor=LIGHT_BG)
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
        color_scale = [[0, SOFT_BLUE], [1, RED]]

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
        labels={"region": pretty_label("region"), "value": "Valor"},
    )
    fig.update_layout(
        title_text="",
        paper_bgcolor=LIGHT_BG,
        plot_bgcolor=LIGHT_BG,
        geo=dict(bgcolor=LIGHT_BG)
    )
    return fig, plot_df

# =========================================================
# Tabs principales
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Segmentos", "Geografía", "Distribuciones"])

# TAB 1: Resumen
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
                color_continuous_scale=[[0, SOFT_BLUE], [1, RED]],
                labels={
                    "subscription_type": pretty_label("subscription_type"),
                    "churn_rate": "Tasa de churn",
                },
            )
            fig_plan.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            apply_plot_style(fig_plan)
            fig_plan.update_layout(
                xaxis_title=pretty_label("subscription_type"),
                yaxis_title="Tasa de churn",
                coloraxis_showscale=False
            )
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
                color_discrete_map=INQUIRIES_COLOR_MAP,
                labels={
                    "customer_service_inquiries": pretty_label("customer_service_inquiries"),
                    "churn_rate": "Tasa de churn",
                },
            )
            fig_inquiries.update_traces(texttemplate="%{y:.2%}", textposition="outside")
            apply_plot_style(fig_inquiries)
            fig_inquiries.update_layout(
                xaxis_title=pretty_label("customer_service_inquiries"),
                yaxis_title="Tasa de churn"
            )
            st.plotly_chart(fig_inquiries, use_container_width=True)

    bottom_left, bottom_right = st.columns([1.1, 1])

    with bottom_left:
        if "weekly_hours" in df.columns:
            compare_df = df.copy()
            if subscription_filter:
                compare_df = compare_df[compare_df["subscription_type"].isin(subscription_filter)]
            if inquiries_filter:
                compare_df = compare_df[compare_df["customer_service_inquiries"].isin(inquiries_filter)]

            st.markdown(f"#### {pretty_label('weekly_hours')} según churn vs no churn")
            fig_weekly = px.box(
                compare_df,
                x="churn_label",
                y="weekly_hours",
                color="churn_label",
                color_discrete_map=CHURN_COLOR_MAP,
                labels={"churn_label": "Grupo", "weekly_hours": pretty_label("weekly_hours")},
            )
            apply_plot_style(fig_weekly)
            fig_weekly.update_layout(showlegend=False, xaxis_title="Grupo", yaxis_title=pretty_label("weekly_hours"))
            st.plotly_chart(fig_weekly, use_container_width=True)

    with bottom_right:
        if "song_skip_rate" in df.columns:
            compare_df = df.copy()
            if subscription_filter:
                compare_df = compare_df[compare_df["subscription_type"].isin(subscription_filter)]
            if inquiries_filter:
                compare_df = compare_df[compare_df["customer_service_inquiries"].isin(inquiries_filter)]

            st.markdown(f"#### {pretty_label('song_skip_rate')} según churn vs no churn")
            fig_skip = px.box(
                compare_df,
                x="churn_label",
                y="song_skip_rate",
                color="churn_label",
                color_discrete_map=CHURN_COLOR_MAP,
                labels={"churn_label": "Grupo", "song_skip_rate": pretty_label("song_skip_rate")},
            )
            apply_plot_style(fig_skip)
            fig_skip.update_layout(showlegend=False, xaxis_title="Grupo", yaxis_title=pretty_label("song_skip_rate"))
            st.plotly_chart(fig_skip, use_container_width=True)

# TAB 2: Segmentos
with tab2:
    if {"subscription_type", "customer_service_inquiries", "weekly_hours", "song_skip_rate", "num_subscription_pauses", "churned"}.issubset(filtered_df.columns):
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
            st.markdown("#### Tasa de churn por tipo de suscripción e incidencias")
            fig_segments = px.bar(
                segment_filtered,
                x="subscription_type",
                y="churn_rate",
                color="customer_service_inquiries",
                barmode="group",
                labels={
                    "subscription_type": pretty_label("subscription_type"),
                    "churn_rate": "Tasa de churn",
                    "customer_service_inquiries": pretty_label("customer_service_inquiries")
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

            heatmap_df.index.name = pretty_label("subscription_type")
            heatmap_df.columns.name = pretty_label("customer_service_inquiries")

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
        for col in ["churn_rate"]:
            nice_segments[col] = nice_segments[col].map(lambda x: f"{x:.2%}")
        for col in ["avg_weekly_hours", "avg_skip_rate", "avg_pauses"]:
            nice_segments[col] = nice_segments[col].map(lambda x: f"{x:.2f}")

        with st.expander("Ver tabla de segmentos"):
            st.dataframe(prettify_columns(nice_segments), use_container_width=True, hide_index=True)
    else:
        st.info("No están disponibles todas las columnas necesarias para la vista de segmentos.")

st.markdown("#### Resumen por segmentos generado con R")
st.caption(
    "Esta tabla se genera a partir de train_model_ready.csv mediante un asset de Dagster "
    "que ejecuta un script en R."
)

segment_filter = st.selectbox(
    "Filtrar resumen R por tipo de suscripción",
    ["Todos"] + sorted(segment_summary_r["subscription_type"].dropna().unique().tolist())
)

segment_summary_display = segment_summary_r.copy()

if segment_filter != "Todos":
    segment_summary_display = segment_summary_display[
        segment_summary_display["subscription_type"] == segment_filter
    ]

st.dataframe(prettify_columns(segment_summary_display), use_container_width=True, hide_index=True)

# TAB 3: Geografía
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
            geo_display = prettify_columns(geo_df.sort_values("value", ascending=False))
            st.dataframe(geo_display, use_container_width=True, hide_index=True)

# TAB 4: Distribuciones
with tab4:
    compare_df = df.copy()

    if subscription_filter:
        compare_df = compare_df[compare_df["subscription_type"].isin(subscription_filter)]
    if inquiries_filter:
        compare_df = compare_df[compare_df["customer_service_inquiries"].isin(inquiries_filter)]

    dist_col1, dist_col2 = st.columns([1.1, 0.9])

    with dist_col1:
        st.markdown(f"#### {pretty_label(numeric_var)} según churn vs no churn")
        fig_box = px.box(
            compare_df,
            x="churn_label",
            y=numeric_var,
            color="churn_label",
            labels={"churn_label": "Grupo", numeric_var: pretty_label(numeric_var)},
            color_discrete_map=CHURN_COLOR_MAP
        )
        apply_plot_style(fig_box)
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with dist_col2:
        hist_df = filtered_df.copy()
        st.markdown(f"#### Distribución de {pretty_label(numeric_var)}")
        fig_hist = px.histogram(
            hist_df,
            x=numeric_var,
            color="churn_label" if view_mode == "Ambos" and "churn_label" in hist_df.columns else None,
            barmode="overlay",
            opacity=0.65,
            nbins=30,
            color_discrete_map=CHURN_COLOR_MAP,
            labels={numeric_var: pretty_label(numeric_var), "churn_label": "Grupo"}
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
            color_continuous_scale=[[0, SOFT_BLUE], [1, RED]],
            labels={"age_group": pretty_label("age_group"), "churn_rate": "Tasa de churn"},
        )
        fig_age.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        apply_plot_style(fig_age)
        fig_age.update_layout(xaxis_title=pretty_label("age_group"), yaxis_title="Tasa de churn", coloraxis_showscale=False)
        st.plotly_chart(fig_age, use_container_width=True)

    if numeric_var in compare_df.columns:
        stats_df = (
            compare_df.groupby("churn_label")[numeric_var]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )
        for col in ["mean", "median", "std", "min", "max"]:
            if col in stats_df.columns:
                stats_df[col] = stats_df[col].round(3)

        stats_df = stats_df.rename(columns={"churn_label": "Grupo"})
        with st.expander("Ver estadísticos descriptivos"):
            st.dataframe(prettify_columns(stats_df), use_container_width=True, hide_index=True)
