from pathlib import Path

import dagster as dg
import pandas as pd


NEW_CUSTOMERS_PATH = Path("data/new_data/new_customers_for_training.csv")


@dg.sensor(job_name="rf_retraining_job")
def new_training_batch_sensor(context: dg.SensorEvaluationContext):
    if not NEW_CUSTOMERS_PATH.exists():
        return dg.SkipReason("No existe data/new_data/new_customers_for_training.csv")

    try:
        stat = NEW_CUSTOMERS_PATH.stat()
        current_cursor = str(stat.st_mtime)
    except FileNotFoundError:
        return dg.SkipReason("El archivo no está disponible")

    if context.cursor == current_cursor:
        return dg.SkipReason("No hay cambios en new_customers_for_training.csv")

    try:
        df = pd.read_csv(NEW_CUSTOMERS_PATH)
    except pd.errors.EmptyDataError:
        context.update_cursor(current_cursor)
        return dg.SkipReason("El archivo existe pero está vacío")
    except Exception as e:
        return dg.SkipReason(f"No se pudo leer el archivo: {e}")

    if df.empty:
        context.update_cursor(current_cursor)
        return dg.SkipReason("El lote existe pero no contiene filas")

    context.update_cursor(current_cursor)

    return dg.RunRequest(
        run_key=f"rf-retraining-{current_cursor}",
        run_config={},
        tags={"trigger": "new_training_batch_sensor"},
    )
