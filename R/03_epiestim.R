## 03_epiestim.R
## Method 1: Estimate Rt with EpiEstim (Cori et al. 2013).
##
## EpiEstim implements the renewal-equation Bayesian estimator
##   I_t | I_{0:t-1}, R_t  ~  Poisson( R_t * sum_s I_{t-s} * w_s )
## where w is the discretized serial interval (SI) distribution.
##
## We use weekly sliding windows on weekly %ILI as a proxy for incidence,
## and run a sensitivity analysis over plausible flu serial intervals.
##
## Outputs (figures/):
##   m1_rt_by_state.png        — Rt trajectory per state for one season
##   m1_rt_by_season.png       — Rt trajectory per season for one state
##   m1_si_sensitivity.png     — Rt under three serial-interval assumptions

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(EpiEstim)
  library(patchwork)
})

theme_set(theme_minimal(base_size = 11) +
            theme(panel.grid.minor = element_blank()))

dir.create("figures", showWarnings = FALSE, recursive = TRUE)
dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)

d <- read_csv("data/processed/fluview_weekly.csv", show_col_types = FALSE) |>
  mutate(week_end = as.Date(week_end))

## ---------------------------------------------------------------------------
## Use weekly incidence directly (time step = 1 week)
## ---------------------------------------------------------------------------
inc_data <- d |>
  filter(region != "NAT", !is.na(num_ili)) |>
  arrange(region, season, week_end) |>
  transmute(region, season, date = week_end, I = num_ili)

## ---------------------------------------------------------------------------
## Serial interval distributions in WEEKS (non-parametric form).
## We discretize a gamma SI on the daily grid, then fold to weekly bins.
## Flu daily SI ~ gamma(mean, sd) for three plausible settings:
##   short:  mean 2.5 d, sd 1.0 d
##   medium: mean 3.6 d, sd 1.6 d   [main]
##   long:   mean 4.8 d, sd 2.0 d
## ---------------------------------------------------------------------------
weekly_si <- function(mean_d, sd_d, n_days = 28) {
  shape <- (mean_d / sd_d)^2
  rate  <- mean_d / sd_d^2
  pmf_d <- diff(pgamma(0:n_days, shape = shape, rate = rate))
  pmf_d <- pmf_d / sum(pmf_d)
  ## Aggregate days into weeks; index 1 = same week (lag 0w), 2 = next week ...
  n_weeks <- ceiling(n_days / 7)
  pmf_w <- numeric(n_weeks + 1)
  for (i in seq_along(pmf_d)) {
    w_lag <- (i - 1) %/% 7 + 1   # 1..n_weeks
    pmf_w[w_lag + 1] <- pmf_w[w_lag + 1] + pmf_d[i]
  }
  ## EpiEstim convention: si_distr[1] = P(SI = 0 time units) = 0
  pmf_w[1] <- 0
  pmf_w / sum(pmf_w)
}

si_settings <- list(
  short  = list(label = "short",  mean_d = 2.5, sd_d = 1.0),
  medium = list(label = "medium", mean_d = 3.6, sd_d = 1.6),
  long   = list(label = "long",   mean_d = 4.8, sd_d = 2.0)
)

## ---------------------------------------------------------------------------
## Helper: run EpiEstim on one (region, season) with a given non-parametric SI
## window = 2 weeks (rolling window for Rt smoothing)
## ---------------------------------------------------------------------------
run_epiestim <- function(df_one, si_distr, window = 2) {
  if (nrow(df_one) < window + 2 || sum(df_one$I, na.rm = TRUE) < 10) return(NULL)

  res <- tryCatch(
    EpiEstim::estimate_R(
      incid = df_one$I,
      method = "non_parametric_si",
      config = EpiEstim::make_config(list(
        si_distr = si_distr,
        t_start = seq(2, nrow(df_one) - window),
        t_end   = seq(2 + window, nrow(df_one))
      ))
    ),
    error = function(e) {
      message("  skip: ", conditionMessage(e)); NULL
    }
  )
  if (is.null(res)) return(NULL)

  tibble(
    t_end   = res$R$t_end,
    date    = df_one$date[res$R$t_end],
    Rt_mean = res$R$`Mean(R)`,
    Rt_lo   = res$R$`Quantile.0.025(R)`,
    Rt_hi   = res$R$`Quantile.0.975(R)`
  )
}

## ---------------------------------------------------------------------------
## Run main analysis: medium SI, all (region, season) combos
## ---------------------------------------------------------------------------
message("Estimating Rt for all (region, season) with medium SI ...")
si_main <- weekly_si(si_settings$medium$mean_d, si_settings$medium$sd_d)
main_results <- inc_data |>
  group_by(region, season) |>
  group_modify(~ {
    out <- run_epiestim(.x, si_distr = si_main)
    if (is.null(out)) return(tibble())
    out
  }) |>
  ungroup()

write_csv(main_results, "data/processed/rt_epiestim.csv")
message(sprintf("  -> %d Rt estimates across %d state-seasons",
                nrow(main_results),
                main_results |> distinct(region, season) |> nrow()))

## ---------------------------------------------------------------------------
## Plot 1: Rt by state, fixed season (2017-2018, the bad one)
## ---------------------------------------------------------------------------
focus_season <- "2019-2020"  # ends abruptly with COVID lockdowns -> Rt drops below 1
p1 <- main_results |>
  filter(season == focus_season) |>
  ggplot(aes(date, Rt_mean, color = region, fill = region)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  geom_ribbon(aes(ymin = Rt_lo, ymax = Rt_hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.5) +
  facet_wrap(~ region, ncol = 4) +
  coord_cartesian(ylim = c(0, 3)) +
  labs(
    title = sprintf("EpiEstim Rt by state, %s flu season", focus_season),
    subtitle = "Ribbon = 95% credible interval; dashed line at Rt = 1",
    x = NULL, y = expression(R[t])
  ) +
  theme(legend.position = "none")

ggsave("figures/m1_rt_by_state.png", p1,
       width = 10, height = 5, dpi = 150)

## ---------------------------------------------------------------------------
## Plot 2: Rt across seasons for a single state
## ---------------------------------------------------------------------------
focus_state <- "CA"
p2 <- main_results |>
  filter(region == focus_state) |>
  group_by(season) |>
  mutate(week_in_season = row_number()) |>
  ggplot(aes(week_in_season, Rt_mean, color = season, fill = season)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  geom_ribbon(aes(ymin = Rt_lo, ymax = Rt_hi), alpha = 0.12, color = NA) +
  geom_line(linewidth = 0.5) +
  coord_cartesian(ylim = c(0, 3)) +
  labs(
    title = sprintf("EpiEstim Rt across seasons, %s", focus_state),
    subtitle = "Aligned by week-of-season",
    x = "Week of season", y = expression(R[t]), color = "Season", fill = "Season"
  )

ggsave("figures/m1_rt_by_season.png", p2,
       width = 9, height = 4.5, dpi = 150)

## ---------------------------------------------------------------------------
## Sensitivity analysis: same data, three different SIs
## ---------------------------------------------------------------------------
message("Running SI sensitivity analysis (CA, 2017-2018) ...")
focus <- inc_data |> filter(region == "CA", season == "2017-2018")

sens <- map_dfr(names(si_settings), function(label) {
  cfg <- si_settings[[label]]
  si_d <- weekly_si(cfg$mean_d, cfg$sd_d)
  out <- run_epiestim(focus, si_distr = si_d)
  if (!is.null(out)) out$si_label <- label
  out
})

p3 <- ggplot(sens, aes(date, Rt_mean, color = si_label, fill = si_label)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  geom_ribbon(aes(ymin = Rt_lo, ymax = Rt_hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.5) +
  scale_color_manual(values = c(short = "#1f77b4", medium = "#2ca02c", long = "#d62728"),
                     name = "Serial interval") +
  scale_fill_manual(values = c(short = "#1f77b4", medium = "#2ca02c", long = "#d62728"),
                    guide = "none") +
  labs(
    title = "Sensitivity of Rt estimates to serial interval (CA, 2017-2018)",
    subtitle = "Longer SI -> larger Rt amplitude (more weight on early infections)",
    x = NULL, y = expression(R[t])
  )

ggsave("figures/m1_si_sensitivity.png", p3,
       width = 9, height = 4.5, dpi = 150)

cat("Method 1 (EpiEstim) done. Figures:\n",
    "  figures/m1_rt_by_state.png\n",
    "  figures/m1_rt_by_season.png\n",
    "  figures/m1_si_sensitivity.png\n",
    "Estimates saved to data/processed/rt_epiestim.csv\n", sep = "")
