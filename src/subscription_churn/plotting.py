import pandas as pd
import plotly.express as px


def plot_churn_by_category(df: pd.DataFrame, category_col: str, target: str = "churned"):
    plot_df = (
        df.groupby(category_col)[target]
        .mean()
        .reset_index()
        .assign(churn_pct=lambda x: x[target] * 100)
        [[category_col, "churn_pct"]]
        .sort_values("churn_pct", ascending=False)
    )

    fig = px.bar(
        plot_df,
        x=category_col,
        y="churn_pct",
        text="churn_pct",
        title=f"Tasa de churn por {category_col}"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis_title="Churn (%)", xaxis_title=category_col)
    return fig