## 04_pomp_mif2.R
## Method 2: Estimate Rt via a stochastic SEIR POMP model fit by mif2.
##
## Model:
##   dS/dt = -beta(t) * S * I / N
##   dE/dt =  beta(t) * S * I / N - sigma * E
##   dI/dt =  sigma * E - gamma * I
##   dR/dt =  gamma * I
##
##   reports_t ~ NegBin( rho * H_t, k )
## where H_t is the number of new I->R transitions in week t and beta(t) is
## a cubic B-spline of time. Then Rt = beta(t) * (S/N) / gamma.
##
## We fit (b1..bK, rho, k, eta=initial S fraction) by iterated filtering (mif2),
## holding sigma, gamma fixed at literature values for influenza.
##
## Outputs:
##   data/processed/rt_pomp.csv     -- weekly Rt with bootstrap CI
##   figures/m2_pomp_fit.png        -- observed vs simulated incidence
##   figures/m2_rt_pomp.png         -- Rt trajectory with CI
##   figures/m2_loglik_trace.png    -- mif2 log-likelihood traces

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(pomp)
})

theme_set(theme_minimal(base_size = 11) +
            theme(panel.grid.minor = element_blank()))
dir.create("figures", showWarnings = FALSE, recursive = TRUE)

set.seed(2026)

## ---------------------------------------------------------------------------
## Data: California 2019-2020 weekly ILI counts
## ---------------------------------------------------------------------------
focus_state  <- "CA"
focus_season <- "2019-2020"
N_pop        <- 39.5e6   # CA population (round number is fine)

d <- read_csv("data/processed/fluview_weekly.csv", show_col_types = FALSE) |>
  filter(region == focus_state, season == focus_season) |>
  arrange(week_end) |>
  mutate(
    week     = row_number(),
    reports  = round(num_ili)        # weekly reported ILI count
  ) |>
  select(week, week_end, reports)

cat(sprintf("Data: %s %s, %d weeks, total reports = %d\n",
            focus_state, focus_season, nrow(d), sum(d$reports)))

## ---------------------------------------------------------------------------
## C-snippets defining the SEIR step + measurement model
## ---------------------------------------------------------------------------
## beta(t) = exp( sum_k b_k * basis_k(t) ),  K cubic B-spline basis functions.
## Spline is built outside the C-snippet via pomp::bspline_basis() and passed
## through covariates.

K_basis <- 6

basis_mat <- bspline_basis(
  x      = seq_len(nrow(d)),
  nbasis = K_basis,
  degree = 3
)
covar_tbl <- as_tibble(basis_mat) |>
  set_names(sprintf("xi%d", seq_len(K_basis))) |>
  mutate(week = seq_len(nrow(d)))

seir_step <- Csnippet("
  double beta = 0;
  beta += b1*xi1 + b2*xi2 + b3*xi3 + b4*xi4 + b5*xi5 + b6*xi6;
  beta = exp(beta);

  double dt_step = dt;
  double rate_SE = beta * I / N;
  double rate_EI = sigma;
  double rate_IR = gamma;

  double trans_SE = rbinom(S, 1 - exp(-rate_SE * dt_step));
  double trans_EI = rbinom(E, 1 - exp(-rate_EI * dt_step));
  double trans_IR = rbinom(I, 1 - exp(-rate_IR * dt_step));

  S -= trans_SE;
  E += trans_SE - trans_EI;
  I += trans_EI - trans_IR;
  R += trans_IR;
  H += trans_IR;
")

rinit <- Csnippet("
  S = nearbyint(eta * N);
  E = 0;
  I = 10;
  R = nearbyint((1 - eta) * N) - I;
  H = 0;
")

dmeas <- Csnippet("
  double mu = rho * H + 1e-6;
  lik = dnbinom_mu(reports, k, mu, give_log);
")

rmeas <- Csnippet("
  double mu = rho * H + 1e-6;
  reports = rnbinom_mu(k, mu);
")

## Reset the H accumulator at each observation
acc <- "H"

## ---------------------------------------------------------------------------
## Build the pomp object
## ---------------------------------------------------------------------------
flu_pomp <- pomp(
  data        = d |> select(week, reports),
  times       = "week",
  t0          = 0,
  rprocess    = euler(seir_step, delta.t = 1/7),
  rinit       = rinit,
  rmeasure    = rmeas,
  dmeasure    = dmeas,
  accumvars   = acc,
  covar       = covariate_table(covar_tbl, times = "week"),
  statenames  = c("S", "E", "I", "R", "H"),
  paramnames  = c(sprintf("b%d", 1:K_basis),
                  "sigma", "gamma", "rho", "k", "eta", "N"),
  partrans = parameter_trans(
    log     = c("k"),
    logit   = c("rho", "eta")
  )
)

## ---------------------------------------------------------------------------
## Fixed + initial parameters
## ---------------------------------------------------------------------------
## sigma, gamma in 1/week. Influenza: latent ~ 1.5 d, infectious ~ 3 d.
sigma_fix <- 7 / 1.5
gamma_fix <- 7 / 3.0

## Initial guesses: log(beta) starts near log(gamma) so Rt ~ 1 at t=0.
## mif2 will move the b_k from there.
b_init <- log(gamma_fix)

p_init <- c(
  b1 = b_init, b2 = b_init, b3 = b_init,
  b4 = b_init, b5 = b_init, b6 = b_init,
  sigma = sigma_fix,
  gamma = gamma_fix,
  rho   = 0.01,     # reporting fraction
  k     = 10,       # NB overdispersion (larger = closer to Poisson)
  eta   = 0.95,
  N     = N_pop
)

## ---------------------------------------------------------------------------
## Quick sanity check: simulate from the prior and look at it
## ---------------------------------------------------------------------------
sim <- simulate(flu_pomp, params = p_init, nsim = 10, format = "data.frame")

p_sim <- ggplot() +
  geom_line(data = sim, aes(week, reports, group = .id), alpha = 0.3) +
  geom_line(data = d,   aes(week, reports), color = "red", linewidth = 0.7) +
  labs(title = "Sanity check: prior simulations (grey) vs observed (red)",
       subtitle = sprintf("%s %s — before mif2", focus_state, focus_season),
       x = "Week of season", y = "Reports")

ggsave("figures/m2_sanity_prior.png", p_sim,
       width = 8, height = 4, dpi = 150)

## ---------------------------------------------------------------------------
## mif2 — iterated filtering
## ---------------------------------------------------------------------------
## Random-walk SD on the spline coefficients + observation params.
rw_sd_spec <- rw_sd(
  b1 = 0.05, b2 = 0.05, b3 = 0.05, b4 = 0.05, b5 = 0.05, b6 = 0.05,
  rho = 0.02, k = 0.05, eta = ivp(0.02)
)

## Cooling: standard geometric, halve sd every 50 iterations
cooling_frac <- 0.5

## Speed knobs
NP_MIF  <- 1000   # particles per filter
NMIF    <- 200    # mif2 iterations
NREPS   <- 4      # parallel mif2 chains

cat(sprintf("\nRunning %d mif2 chains x %d iters x %d particles ...\n",
            NREPS, NMIF, NP_MIF))
t_start <- Sys.time()

mif_runs <- vector("list", NREPS)
for (r in seq_len(NREPS)) {
  cat(sprintf("  chain %d/%d ...\n", r, NREPS))
  mif_runs[[r]] <- mif2(
    flu_pomp,
    params         = p_init,
    Np             = NP_MIF,
    Nmif           = NMIF,
    cooling.fraction.50 = cooling_frac,
    rw.sd          = rw_sd_spec
  )
}

cat(sprintf("mif2 done in %.1f min\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

## Pick best-likelihood replicate
ll_finals <- sapply(mif_runs, logLik)
cat("Final log-likelihoods per chain:\n"); print(ll_finals)
best <- which.max(ll_finals)
mf   <- mif_runs[[best]]
p_mle <- coef(mf)
cat("Best chain:", best, " logLik =", ll_finals[best], "\n")
print(round(p_mle, 4))

## ---------------------------------------------------------------------------
## Diagnostics: log-lik trace per chain
## ---------------------------------------------------------------------------
trace_df <- map_dfr(seq_along(mif_runs), function(i) {
  tr <- traces(mif_runs[[i]])
  tibble(iter = seq_len(nrow(tr)),
         loglik = tr[, "loglik"],
         chain  = factor(i))
})

p_trace <- ggplot(trace_df, aes(iter, loglik, color = chain)) +
  geom_line() +
  labs(title = "mif2 log-likelihood traces",
       x = "mif2 iteration", y = "log-likelihood")

ggsave("figures/m2_loglik_trace.png", p_trace,
       width = 8, height = 4, dpi = 150)

## ---------------------------------------------------------------------------
## Posterior predictive: simulate from the MLE
## ---------------------------------------------------------------------------
sim_post <- simulate(flu_pomp, params = p_mle, nsim = 100, format = "data.frame")

post_summary <- sim_post |>
  group_by(week) |>
  summarise(
    med = median(reports),
    lo  = quantile(reports, 0.025),
    hi  = quantile(reports, 0.975),
    .groups = "drop"
  )

p_fit <- ggplot() +
  geom_ribbon(data = post_summary, aes(week, ymin = lo, ymax = hi),
              fill = "steelblue", alpha = 0.25) +
  geom_line(data = post_summary, aes(week, med),
            color = "steelblue", linewidth = 0.6) +
  geom_point(data = d, aes(week, reports), color = "black", size = 1.4) +
  labs(title = sprintf("POMP fit: %s %s", focus_state, focus_season),
       subtitle = "Points = observed weekly ILI; band = 95% posterior predictive",
       x = "Week of season", y = "Reports")

ggsave("figures/m2_pomp_fit.png", p_fit,
       width = 8, height = 4, dpi = 150)

## ---------------------------------------------------------------------------
## Recover Rt = beta(t) * S/N / gamma
## ---------------------------------------------------------------------------
## Use S(t) trajectory averaged across simulations.
S_traj <- sim_post |>
  group_by(week) |>
  summarise(S_mean = mean(S), .groups = "drop")

beta_t <- exp(as.numeric(basis_mat %*% p_mle[sprintf("b%d", 1:K_basis)]))
Rt_pomp <- tibble(
  week     = seq_along(beta_t),
  week_end = d$week_end,
  beta     = beta_t,
  S        = S_traj$S_mean,
  Rt_mean  = beta_t * (S_traj$S_mean / N_pop) / p_mle["gamma"]
)

## Bootstrap CI by resampling spline coefficients across mif chains
boot_Rt <- map_dfr(seq_along(mif_runs), function(i) {
  pp <- coef(mif_runs[[i]])
  bt <- exp(as.numeric(basis_mat %*% pp[sprintf("b%d", 1:K_basis)]))
  tibble(week = seq_along(bt),
         Rt   = bt * (S_traj$S_mean / N_pop) / pp["gamma"],
         chain = i)
})

ci <- boot_Rt |>
  group_by(week) |>
  summarise(Rt_lo = quantile(Rt, 0.025),
            Rt_hi = quantile(Rt, 0.975),
            .groups = "drop")

Rt_pomp <- Rt_pomp |> left_join(ci, by = "week")
write_csv(Rt_pomp, "data/processed/rt_pomp.csv")

p_rt <- ggplot(Rt_pomp, aes(week_end, Rt_mean)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  geom_ribbon(aes(ymin = Rt_lo, ymax = Rt_hi), fill = "darkorange", alpha = 0.2) +
  geom_line(color = "darkorange", linewidth = 0.7) +
  labs(title = sprintf("POMP + mif2 Rt estimate: %s %s", focus_state, focus_season),
       subtitle = "Cubic B-spline beta(t); ribbon = range across mif2 chains",
       x = NULL, y = expression(R[t]))

ggsave("figures/m2_rt_pomp.png", p_rt,
       width = 9, height = 4.5, dpi = 150)

cat("\nMethod 2 (POMP + mif2) done.\n")
cat("  data/processed/rt_pomp.csv\n")
cat("  figures/m2_pomp_fit.png\n")
cat("  figures/m2_rt_pomp.png\n")
cat("  figures/m2_loglik_trace.png\n")
