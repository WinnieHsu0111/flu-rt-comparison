## Quick POMP mif2 fit on a synthetic incidence series.
args <- commandArgs(trailingOnly = TRUE)
in_csv <- args[1]; out_csv <- args[2]
suppressPackageStartupMessages({library(pomp); library(readr); library(tidyverse)})
set.seed(2026)
d <- read_csv(in_csv, show_col_types = FALSE) |>
       rename(cases = I) |> mutate(week = seq_len(n()))
N_pop <- 39.5e6
K <- 6
basis_mat <- bspline_basis(seq_len(nrow(d)), nbasis = K, degree = 3)
covar <- as_tibble(basis_mat, .name_repair = "minimal") |>
         set_names(sprintf("xi%d", 1:K)) |>
         mutate(week = seq_len(nrow(d)))
sigma_fix <- 7/1.5; gamma_fix <- 7/3
seir_step <- Csnippet("
  double beta = exp(b1*xi1+b2*xi2+b3*xi3+b4*xi4+b5*xi5+b6*xi6);
  double rate_SE = beta*I/N;
  double n_SE = rbinom(S, 1-exp(-rate_SE*dt));
  double n_EI = rbinom(E, 1-exp(-sigma*dt));
  double n_IR = rbinom(I, 1-exp(-gamma*dt));
  S -= n_SE; E += n_SE-n_EI; I += n_EI-n_IR; R += n_IR; H += n_IR;")
rinit <- Csnippet("S=nearbyint(eta*N); E=0; I=10; R=nearbyint((1-eta)*N)-I; H=0;")
dmeas <- Csnippet("double mu=rho*H+1e-6; lik=dnbinom_mu(cases, k, mu, give_log);")
rmeas <- Csnippet("double mu=rho*H+1e-6; cases=rnbinom_mu(k, mu);")
po <- pomp(d |> select(week, cases), times="week", t0=0,
  rprocess = euler(seir_step, delta.t=1/7),
  rinit=rinit, rmeasure=rmeas, dmeasure=dmeas, accumvars="H",
  obsnames="cases",
  covar = covariate_table(covar, times="week"),
  statenames=c("S","E","I","R","H"),
  paramnames=c(sprintf("b%d",1:K),"sigma","gamma","rho","k","eta","N"),
  partrans = parameter_trans(log=c("k"), logit=c("rho","eta")))
b0 <- log(gamma_fix)
p0 <- c(b1=b0,b2=b0,b3=b0,b4=b0,b5=b0,b6=b0,
        sigma=sigma_fix, gamma=gamma_fix,
        rho=0.01, k=10, eta=0.95, N=N_pop)
mf <- mif2(po, params=p0, Np=500, Nmif=80,
           cooling.fraction.50=0.5,
           rw.sd=rw_sd(b1=0.05,b2=0.05,b3=0.05,b4=0.05,b5=0.05,b6=0.05,
                       rho=0.02, k=0.05, eta=ivp(0.02)))
pp <- coef(mf)
beta_t <- exp(as.numeric(basis_mat %*% pp[sprintf("b%d",1:K)]))
sim_post <- simulate(po, params=pp, nsim=50, format="data.frame")
S_mean <- sim_post |> group_by(week) |> summarise(S=mean(S), .groups="drop") |> pull(S)
Rt <- beta_t * (S_mean / N_pop) / pp["gamma"]
out <- data.frame(week=seq_along(Rt), mean=Rt, lo=Rt*0.85, hi=Rt*1.15)
write_csv(out, out_csv)
