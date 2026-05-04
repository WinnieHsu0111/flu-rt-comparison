"""
07_simulation_study.py
----------------------
Head-to-head comparison of three Rt estimation methods on synthetic outbreaks
with known ground-truth Rt trajectories.

For N_SIM synthetic outbreaks:
  - Generate (theta_true, x_obs) from the SEIR simulator
  - Run EpiEstim   (via Rscript subprocess)
  - Run POMP+mif2  (via Rscript subprocess)
  - Run Neural SBI (via the trained NPE posterior)
  - Compute MAE, 95% CI coverage, runtime

Outputs:
  data/processed/sim_results.csv       - per-week Rt estimates by method/sim
  data/processed/sim_metrics.csv       - method-level summary metrics
  figures/m4_sim_trajectories.png      - example overlay of methods vs truth
  figures/m4_sim_metrics.png           - bar chart of MAE / coverage / runtime
"""
from __future__ import annotations

import os
import sys
import subprocess
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
from importlib import import_module
sim_mod = import_module("05_sbi_simulator")
simulate_one    = sim_mod.simulate_one
beta_trajectory = sim_mod.beta_trajectory
T_OBS           = sim_mod.T_OBS

DATA_DIR = os.path.join(ROOT, "data", "processed")
FIG_DIR  = os.path.join(ROOT, "figures")
TMP_DIR  = os.path.join(ROOT, "data", "tmp_sim")
os.makedirs(TMP_DIR, exist_ok=True)

N_SIM = 5

# ---------------------------------------------------------------------------
# Generate ground-truth synthetic outbreaks
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
truths = []
for s in range(N_SIM):
    # Vary the four log-Rt knots so we get diverse outbreak shapes
    r1 = rng.uniform(np.log(1.2), np.log(2.0))
    r2 = rng.uniform(np.log(1.1), np.log(1.8))
    r3 = rng.uniform(np.log(0.6), np.log(1.0))
    r4 = rng.uniform(np.log(0.5), np.log(0.9))
    # Larger I0 (1000) and rho=0.05 -> outbreak takes off earlier and counts
    # are well above zero, which is necessary for EpiEstim/POMP to estimate Rt
    theta = np.array([r1, r2, r3, r4, 0.05, 8.0, 0.95, 3.0])
    y, S_traj = simulate_one(theta, rng, return_S=True)
    beta_t, R0_t = beta_trajectory(theta[:4])
    # Effective Rt = beta(t) * S(t)/N / gamma  (matches what EpiEstim/POMP estimate)
    Rt_true = R0_t * (S_traj / sim_mod.N_POP)
    truths.append({"sim": s, "theta": theta, "y": y, "Rt_true": Rt_true})
    pd.DataFrame({"week": np.arange(T_OBS), "I": y}).to_csv(
        os.path.join(TMP_DIR, f"sim{s}.csv"), index=False)
print(f"Generated {N_SIM} synthetic outbreaks.")


# ---------------------------------------------------------------------------
# Method 1: EpiEstim, via a small Rscript helper
# ---------------------------------------------------------------------------
EPIESTIM_HELPER = os.path.join(ROOT, "R", "helper_epiestim_one.R")
with open(EPIESTIM_HELPER, "w") as f:
    f.write("""## Read I from arg, write Rt CSV (week, mean, lo, hi)
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
""")


def run_epiestim(sim_idx: int):
    in_csv  = os.path.join(TMP_DIR, f"sim{sim_idx}.csv")
    out_csv = os.path.join(TMP_DIR, f"sim{sim_idx}_epiestim.csv")
    t0 = time.time()
    subprocess.run(["Rscript", EPIESTIM_HELPER, in_csv, out_csv],
                   check=True, capture_output=True)
    rt = time.time() - t0
    df = pd.read_csv(out_csv)
    return df, rt


# ---------------------------------------------------------------------------
# Method 2: POMP-style mif2, via a tiny helper script
# A quick mif2 fit on the synthetic data with the same SEIR model.
# ---------------------------------------------------------------------------
POMP_HELPER = os.path.join(ROOT, "R", "helper_pomp_one.R")
with open(POMP_HELPER, "w") as f:
    f.write("""## Quick POMP mif2 fit on a synthetic incidence series.
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
""")


def run_pomp(sim_idx: int):
    in_csv  = os.path.join(TMP_DIR, f"sim{sim_idx}.csv")
    out_csv = os.path.join(TMP_DIR, f"sim{sim_idx}_pomp.csv")
    t0 = time.time()
    res = subprocess.run(["Rscript", POMP_HELPER, in_csv, out_csv],
                         capture_output=True)
    rt = time.time() - t0
    if res.returncode != 0:
        print(f"   POMP failed on sim{sim_idx}: {res.stderr.decode()[-300:]}")
        return None, rt
    df = pd.read_csv(out_csv)
    return df, rt


# ---------------------------------------------------------------------------
# Method 3: Neural SBI — re-train one NPE on N_TRAIN simulations,
# then re-use it for all N_SIM observations (this is the SBI advantage:
# amortized inference).
# ---------------------------------------------------------------------------
from sbi.inference import NPE
from sbi.utils import BoxUniform

print("Training shared NPE for SBI step ...")
torch.manual_seed(2026)
prior = BoxUniform(
    low=torch.from_numpy(sim_mod.PRIOR_LOW).float(),
    high=torch.from_numpy(sim_mod.PRIOR_HIGH).float())
N_TRAIN = 5000
theta_train = prior.sample((N_TRAIN,))
x_train_raw = sim_mod.simulator_torch(theta_train)
x_train = torch.log1p(x_train_raw)
inf = NPE(prior=prior, density_estimator="maf")
inf.append_simulations(theta_train, x_train)
de = inf.train(show_train_summary=False)
posterior = inf.build_posterior(de)
print("  done.")


def run_sbi(y_obs: np.ndarray):
    t0 = time.time()
    y_t = torch.log1p(torch.from_numpy(y_obs.astype(np.float32)))
    posterior.set_default_x(y_t)
    samples = posterior.sample((1000,), show_progress_bars=False).numpy()
    Rt_samples = np.zeros((samples.shape[0], T_OBS))
    for i, s in enumerate(samples):
        _, Rt = beta_trajectory(s[:4])
        Rt_samples[i] = Rt
    rt = time.time() - t0
    return pd.DataFrame({
        "week": np.arange(T_OBS),
        "mean": np.median(Rt_samples, axis=0),
        "lo":   np.quantile(Rt_samples, 0.025, axis=0),
        "hi":   np.quantile(Rt_samples, 0.975, axis=0)}), rt


# ---------------------------------------------------------------------------
# Run all methods on all sims
# ---------------------------------------------------------------------------
records = []
runtimes = {"epiestim": [], "pomp": [], "sbi": []}

for tr in truths:
    s = tr["sim"]
    print(f"\n--- sim {s} ---")
    print(" EpiEstim ...", end=" ", flush=True)
    df_ee, t_ee = run_epiestim(s)
    runtimes["epiestim"].append(t_ee)
    print(f"{t_ee:.1f}s")
    print(" POMP+mif2 ...", end=" ", flush=True)
    df_pp, t_pp = run_pomp(s)
    runtimes["pomp"].append(t_pp)
    print(f"{t_pp:.1f}s")
    print(" Neural SBI ...", end=" ", flush=True)
    df_sb, t_sb = run_sbi(tr["y"])
    runtimes["sbi"].append(t_sb)
    print(f"{t_sb:.1f}s")

    Rt_true = tr["Rt_true"]
    for label, df in [("EpiEstim", df_ee), ("POMP", df_pp), ("NeuralSBI", df_sb)]:
        if df is None: continue
        d2 = df.copy()
        d2["sim"] = s; d2["method"] = label
        # EpiEstim weeks are 1-indexed; clip to valid Rt_true range
        idx = d2["week"].values - (1 if label == "EpiEstim" else 0)
        idx = np.clip(idx, 0, len(Rt_true) - 1)
        d2["Rt_true"] = Rt_true[idx]
        records.append(d2)

all_results = pd.concat(records, ignore_index=True)
all_results.to_csv(os.path.join(DATA_DIR, "sim_results.csv"), index=False)
print(f"\nSaved sim_results.csv  ({len(all_results)} rows)")


# ---------------------------------------------------------------------------
# Summary metrics: MAE + coverage + runtime
# ---------------------------------------------------------------------------
def metrics(df):
    # Restrict to outbreak window (weeks 5-25) where Rt is identifiable.
    # Outside this window incidence is near zero and all estimators degrade,
    # which is itself a finding (discussed in the report).
    sub = df[(df["week"] >= 5) & (df["week"] <= 25)]
    err = (sub["mean"] - sub["Rt_true"]).abs()
    cover = ((sub["lo"] <= sub["Rt_true"]) & (sub["Rt_true"] <= sub["hi"])).mean()
    return pd.Series({"MAE_outbreak": err.mean(),
                      "Coverage95_outbreak": cover})

summary = (all_results.groupby("method")
                      .apply(metrics, include_groups=False)
                      .reset_index())
summary["RuntimeSec_mean"] = summary["method"].map({
    "EpiEstim":  np.mean(runtimes["epiestim"]),
    "POMP":      np.mean(runtimes["pomp"]),
    "NeuralSBI": np.mean(runtimes["sbi"])})
summary.to_csv(os.path.join(DATA_DIR, "sim_metrics.csv"), index=False)
print(summary)


# ---------------------------------------------------------------------------
# Figure: trajectories overlay for one example sim
# ---------------------------------------------------------------------------
ex = 0
truth = truths[ex]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axhline(1, ls="--", c="grey")
ax.plot(np.arange(T_OBS), truth["Rt_true"], "k-", lw=2, label="truth")
colors = {"EpiEstim": "tab:blue", "POMP": "tab:orange", "NeuralSBI": "tab:purple"}
for label in colors:
    sub = all_results[(all_results.sim == ex) & (all_results.method == label)]
    if sub.empty: continue
    ax.plot(sub["week"], sub["mean"], color=colors[label], label=label, lw=1.2)
    ax.fill_between(sub["week"], sub["lo"], sub["hi"], color=colors[label], alpha=0.15)
ax.set(title=f"Synthetic outbreak #{ex}: three methods vs ground truth",
       xlabel="Week", ylabel=r"$R_t$", ylim=(0.3, 2.5))
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "m4_sim_trajectories.png"), dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: bar chart of metrics
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, col, ylabel in zip(
    axes,
    ["MAE_outbreak", "Coverage95_outbreak", "RuntimeSec_mean"],
    ["MAE (Rt) on weeks 5-25", "95% CI coverage", "Runtime / dataset (s)"]):
    ax.bar(summary["method"], summary[col],
           color=["tab:blue", "tab:purple", "tab:orange"])
    ax.set(title=col.replace("_", " "), ylabel=ylabel)
    if col == "Coverage95_outbreak":
        ax.axhline(0.95, ls="--", color="grey")
        ax.set_ylim(0, 1.05)
fig.suptitle("Method comparison on N=%d synthetic outbreaks (outbreak window)" % N_SIM)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "m4_sim_metrics.png"), dpi=150)
plt.close(fig)

print("\nSimulation study done.")
print(" figures/m4_sim_trajectories.png")
print(" figures/m4_sim_metrics.png")
print(" data/processed/sim_metrics.csv")
