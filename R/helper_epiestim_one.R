## Read I from arg, write Rt CSV (week, mean, lo, hi)
args <- commandArgs(trailingOnly = TRUE)
in_csv  <- args[1]; out_csv <- args[2]
suppressPackageStartupMessages({library(EpiEstim); library(readr)})
d <- read_csv(in_csv, show_col_types = FALSE)
weekly_si <- function(mean_d, sd_d, n_days = 28) {
  shape <- (mean_d / sd_d)^2; rate <- mean_d / sd_d^2
  pmf_d <- diff(pgamma(0:n_days, shape = shape, rate = rate))
  pmf_d <- pmf_d / sum(pmf_d)
  n_weeks <- ceiling(n_days / 7)
  pmf_w <- numeric(n_weeks + 1)
  for (i in seq_along(pmf_d)) {
    w_lag <- (i - 1) %/% 7 + 1
    pmf_w[w_lag + 1] <- pmf_w[w_lag + 1] + pmf_d[i]
  }
  pmf_w[1] <- 0; pmf_w / sum(pmf_w)
}
si <- weekly_si(3.6, 1.6)
window <- 2
res <- EpiEstim::estimate_R(
  incid = d$I, method = "non_parametric_si",
  config = EpiEstim::make_config(list(
    si_distr = si,
    t_start = seq(2, nrow(d) - window),
    t_end   = seq(2 + window, nrow(d)))))
out <- data.frame(week = res$R$t_end,
                  mean = res$R$`Mean(R)`,
                  lo   = res$R$`Quantile.0.025(R)`,
                  hi   = res$R$`Quantile.0.975(R)`)
write_csv(out, out_csv)
