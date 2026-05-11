args <- commandArgs(trailingOnly = TRUE)

input_path <- args[1]
output_path <- args[2]

suppressPackageStartupMessages(library(dplyr))

df <- read.csv(input_path)

summary_df <- df %>%
  group_by(subscription_type, customer_service_inquiries) %>%
  summarise(
    churn_rate = mean(churned, na.rm = TRUE),
    avg_weekly_hours = mean(weekly_hours, na.rm = TRUE),
    avg_skip_rate = mean(song_skip_rate, na.rm = TRUE),
    avg_pauses = mean(num_subscription_pauses, na.rm = TRUE),
    n_customers = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(churn_rate))

write.csv(summary_df, output_path, row.names = FALSE)