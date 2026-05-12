from dagster import Definitions, define_asset_job, AssetSelection

from churn_project.dagster_assets import (
    raw_data,
    train_model_ready,
    train_model_ready_csv,
    train_model_ready_base,
    retraining_pool,
    raw_incoming_data,
    incoming_prediction_ready,
    incoming_prediction_ready_csv,
    segment_summary_r_csv,
    new_customers_for_training,
    combined_training_data,
    rf_retraining_outputs,
)
from churn_project.dagster_sensors import new_training_batch_sensor


rf_retraining_job = define_asset_job(
    name="rf_retraining_job",
    selection=AssetSelection.assets("combined_training_data", "rf_retraining_outputs"),
)

defs = Definitions(
    assets=[
        raw_data,
        train_model_ready,
        train_model_ready_csv,
        train_model_ready_base,
        retraining_pool,
        raw_incoming_data,
        incoming_prediction_ready,
        incoming_prediction_ready_csv,
        segment_summary_r_csv,
        new_customers_for_training,
        combined_training_data,
        rf_retraining_outputs,
    ],
    jobs=[rf_retraining_job],
    sensors=[new_training_batch_sensor],
)
