from dagster import Definitions

from churn_project.dagster_assets import (
    raw_data,
    train_model_ready,
    train_model_ready_csv,
    raw_incoming_data,
    incoming_prediction_ready,
    incoming_prediction_ready_csv,
    segment_summary_r_csv,
    new_customers_for_training,
    combined_training_data,
    rf_retraining_outputs,
)

defs = Definitions(
    assets=[
        raw_data,
        train_model_ready,
        train_model_ready_csv,
        raw_incoming_data,
        incoming_prediction_ready,
        incoming_prediction_ready_csv,
        segment_summary_r_csv,
        new_customers_for_training,
        combined_training_data,
        rf_retraining_outputs,
    ]
)
