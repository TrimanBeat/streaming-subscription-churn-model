from dagster import Definitions

from churn_project.dagster_assets import (
    raw_data,
    train_model_ready,
    train_model_ready_csv,
    segment_summary_r_csv,
)
defs = Definitions(
    assets=[raw_data, train_model_ready, train_model_ready_csv, segment_summary_r_csv]
)
