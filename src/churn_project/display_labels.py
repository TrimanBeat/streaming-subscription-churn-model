from __future__ import annotations

DISPLAY_LABELS = {
    "age": "Edad",
    "subscription_type": "Tipo de suscripción",
    "weekly_hours": "Horas semanales",
    "song_skip_rate": "Tasa de salto de canciones",
    "num_subscription_pauses": "N.º de pausas de suscripción",
    "customer_service_inquiries": "Consultas a atención al cliente",
    "average_session_length": "Duración media de sesión",
    "weekly_unique_songs": "Canciones únicas por semana",
    "weekly_songs_played": "Canciones reproducidas por semana",
    "notifications_clicked": "Notificaciones pulsadas",
    "location": "Ubicación",
    "state_code": "Estado",
    "region": "Región",
    "tenure_days": "Antigüedad (días)",
    "tenure_months": "Antigüedad (meses)",
    "songs_per_hour": "Canciones por hora",
    "high_skip_user": "Usuario con alto skip",
    "age_group": "Grupo de edad",
    "weekly_hours_bin": "Tramo de horas semanales",
    "skip_rate_bin": "Tramo de tasa de salto",
    "engagement_index": "Índice de engagement",
    "is_free_plan": "Plan gratuito",
    "is_high_inquiries": "Muchas consultas a soporte",
    "churned": "Churn observado",
    "p_churn": "Probabilidad de churn",
    "risk_level": "Nivel de riesgo",
    "source": "Origen",
    "model_used": "Modelo usado",
}


def pretty_label(column_name: str) -> str:
    return DISPLAY_LABELS.get(column_name, column_name.replace("_", " ").capitalize())


def prettify_columns(df):
    return df.rename(columns=DISPLAY_LABELS)