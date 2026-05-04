## 02_eda.R
## Quick exploratory plots of the FluView weekly ILI data.
##
## Outputs (figures/):
##   eda_timeseries_by_state.png   — %ILI over time, all states overlaid
##   eda_seasonal_facet.png        — %ILI per season, faceted by region
##   eda_peak_summary.png          — peak %ILI per state-season

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(scales)
})

theme_set(theme_minimal(base_size = 11) +
            theme(panel.grid.minor = element_blank()))

dir.create("figures", showWarnings = FALSE, recursive = TRUE)

d <- read_csv("data/processed/fluview_weekly.csv", show_col_types = FALSE) |>
  mutate(week_end = as.Date(week_end))

states <- d |> filter(region != "NAT")

## --- 1) Time series, all states overlaid ---------------------------------
p1 <- ggplot(states, aes(week_end, ili, color = region)) +
  geom_line(linewidth = 0.4, alpha = 0.85) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(
    title = "Weekly Influenza-Like Illness (%ILI), US States",
    subtitle = "CDC FluView, 2017-2018 through 2022-2023 seasons",
    x = NULL, y = "%ILI", color = "State"
  )

ggsave("figures/eda_timeseries_by_state.png", p1,
       width = 9, height = 4.5, dpi = 150)

## --- 2) Faceted by region, season aligned --------------------------------
season_aligned <- states |>
  group_by(region, season) |>
  mutate(week_in_season = row_number()) |>
  ungroup()

p2 <- ggplot(season_aligned, aes(week_in_season, ili, color = season)) +
  geom_line(linewidth = 0.4) +
  facet_wrap(~ region, ncol = 4, scales = "free_y") +
  labs(
    title = "Season-aligned %ILI trajectories by state",
    subtitle = "Week 1 = first reporting week of each season (typically MMWR week 40)",
    x = "Week of season", y = "%ILI", color = "Season"
  )

ggsave("figures/eda_seasonal_facet.png", p2,
       width = 11, height = 6, dpi = 150)

## --- 3) Peak %ILI per state-season ---------------------------------------
peaks <- states |>
  group_by(region, season) |>
  summarise(peak_ili = max(ili, na.rm = TRUE), .groups = "drop")

p3 <- ggplot(peaks, aes(season, peak_ili, fill = region)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  labs(
    title = "Peak %ILI per state and season",
    x = NULL, y = "Peak %ILI", fill = "State"
  ) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave("figures/eda_peak_summary.png", p3,
       width = 9, height = 4.5, dpi = 150)

cat("Figures written to figures/\n")
