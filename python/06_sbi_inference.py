"""
06_sbi_inference.py
-------------------
Fit a Neural Posterior Estimator (NPE) to the SEIR simulator and use it
to recover Rt(t) from real CA 2019-2020 ILI data.

Pipeline:
  1. Sample N=10000 (theta, x) pairs from prior x simulator.
  2. Train an NPE (masked autoregressive flow on log-counts).
  3. Sample posterior given (a) a synthetic test trajectory (validation)
     and (b) the real CA observation.
  4. Recover Rt(t) = exp(spline(theta_post)) and save figures.

Outputs (figures/):
  m3_sbi_validation.png   - posterior on synthetic data with known Rt
  m3_sbi_real.png         - posterior Rt for CA 2019-2020 + obs fit
Outputs (data/processed/):
  rt_sbi.csv              - posterior median + 95% CI Rt trajectory
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sbi.inference import NPE
from sbi.utils import BoxUniform

# Local module
import sys
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
sim_mod = import_module("05_sbi_simulator")
simulate_one     = sim_mod.simulate_one
simulator_torch  = sim_mod.simulator_torch
beta_trajectory  = sim_mod.beta_trajectory
PRIOR_LOW        = sim_mod.PRIOR_LOW
PRIOR_HIGH       = sim_mod.PRIOR_HIGH
PARAM_NAMES      = sim_mod.PARAM_NAMES
T_OBS            = sim_mod.T_OBS

torch.manual_seed(2026)
np.random.seed(2026)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR  = os.path.join(ROOT, "figures")
DATA_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Prior + simulation
# ---------------------------------------------------------------------------
prior = BoxUniform(
    low=torch.from_numpy(PRIOR_LOW).float(),
    high=torch.from_numpy(PRIOR_HIGH).float(),
)

N_SIMS = 10_000
print(f"Drawing {N_SIMS} prior samples and simulating ...")
theta = prior.sample((N_SIMS,))
# log1p transform of counts stabilises training
x_raw = simulator_torch(theta)
x = torch.log1p(x_raw)

print(f"  theta shape: {tuple(theta.shape)}, x shape: {tuple(x.shape)}")

# ---------------------------------------------------------------------------
# 2. Train NPE
# ---------------------------------------------------------------------------
print("Training NPE ...")
inference = NPE(prior=prior, density_estimator="maf")
inference = inference.append_simulations(theta, x)
density_estimator = inference.train(show_train_summary=True)
posterior = inference.build_posterior(density_estimator)
print("  done.")


# ---------------------------------------------------------------------------
# 3. Validation on synthetic data (known Rt)
# ---------------------------------------------------------------------------
true_theta = np.array([
    np.log(1.5),  # R1
    np.log(1.6),  # R2
    np.log(0.9),  # R3
    np.log(0.7),  # R4
    0.01,         # rho
    8.0,          # k
    0.95,         # eta
    1.0,          # log10 I0
])
rng = np.random.default_rng(42)
x_obs = sim_mod.simulate_one(true_theta, rng).astype(np.float32)
x_obs_t = torch.log1p(torch.from_numpy(x_obs))

print("Sampling posterior on synthetic obs ...")
posterior.set_default_x(x_obs_t)
samples = posterior.sample((2000,))
samples_np = samples.detach().cpu().numpy()

# recover Rt trajectory per posterior sample
weeks = np.arange(T_OBS)
Rt_samples = np.zeros((samples_np.shape[0], T_OBS))
for i, s in enumerate(samples_np):
    _, Rt = beta_trajectory(s[:4])
    Rt_samples[i] = Rt

Rt_lo  = np.quantile(Rt_samples, 0.025, axis=0)
Rt_med = np.quantile(Rt_samples, 0.5,   axis=0)
Rt_hi  = np.quantile(Rt_samples, 0.975, axis=0)

_, Rt_true = beta_trajectory(true_theta[:4])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

ax = axes[0]
ax.plot(weeks, x_obs, "k-", lw=1, label="observed (synthetic)")
# Posterior predictive: simulate from a thinned set of posterior samples
n_pp = 100
pp = np.zeros((n_pp, T_OBS))
for i, s in enumerate(samples_np[:n_pp]):
    pp[i] = sim_mod.simulate_one(s, rng)
ax.fill_between(weeks,
                np.quantile(pp, 0.025, axis=0),
                np.quantile(pp, 0.975, axis=0),
                alpha=0.25, color="purple", label="95% post. pred.")
ax.set(title="Synthetic data fit", xlabel="Week", ylabel="Reports")
ax.legend()

ax = axes[1]
ax.axhline(1, ls="--", c="grey")
ax.fill_between(weeks, Rt_lo, Rt_hi, alpha=0.25, color="purple", label="95% CI")
ax.plot(weeks, Rt_med, color="purple", lw=1.2, label="posterior median")
ax.plot(weeks, Rt_true, "k--", lw=1.2, label="true Rt")
ax.set(title="Recovered Rt vs truth", xlabel="Week", ylabel=r"$R_t$",
       ylim=(0.4, 2.5))
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "m3_sbi_validation.png"), dpi=150)
plt.close(fig)
print("  saved figures/m3_sbi_validation.png")


# ---------------------------------------------------------------------------
# 4. Apply to real CA 2019-2020 data
# ---------------------------------------------------------------------------
real_path = os.path.join(DATA_DIR, "fluview_weekly.csv")
real_df = pd.read_csv(real_path)
real_df = real_df[(real_df.region == "CA") & (real_df.season == "2019-2020")]
real_df = real_df.sort_values("week_end").reset_index(drop=True)
y_real = real_df.num_ili.values.astype(np.float32)

if len(y_real) > T_OBS:
    y_real = y_real[:T_OBS]
elif len(y_real) < T_OBS:
    y_real = np.concatenate([y_real, np.zeros(T_OBS - len(y_real))])

print("Sampling posterior on real CA 2019-2020 data ...")
y_real_t = torch.log1p(torch.from_numpy(y_real))
posterior.set_default_x(y_real_t)
samples_real = posterior.sample((2000,)).detach().cpu().numpy()

Rt_real_samples = np.zeros((samples_real.shape[0], T_OBS))
for i, s in enumerate(samples_real):
    _, Rt = beta_trajectory(s[:4])
    Rt_real_samples[i] = Rt

Rt_real_lo  = np.quantile(Rt_real_samples, 0.025, axis=0)
Rt_real_med = np.quantile(Rt_real_samples, 0.5,   axis=0)
Rt_real_hi  = np.quantile(Rt_real_samples, 0.975, axis=0)

# Posterior predictive on real data
n_pp = 100
pp_real = np.zeros((n_pp, T_OBS))
for i, s in enumerate(samples_real[:n_pp]):
    pp_real[i] = sim_mod.simulate_one(s, rng)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

ax = axes[0]
ax.plot(weeks, y_real, "k.-", lw=1, ms=3, label="observed")
ax.fill_between(weeks,
                np.quantile(pp_real, 0.025, axis=0),
                np.quantile(pp_real, 0.975, axis=0),
                alpha=0.25, color="purple", label="95% post. pred.")
ax.plot(weeks, np.quantile(pp_real, 0.5, axis=0), color="purple", lw=1)
ax.set(title="CA 2019-2020 fit", xlabel="Week of season", ylabel="Reports")
ax.legend()

ax = axes[1]
ax.axhline(1, ls="--", c="grey")
ax.fill_between(weeks, Rt_real_lo, Rt_real_hi, alpha=0.25, color="purple",
                label="95% CI")
ax.plot(weeks, Rt_real_med, color="purple", lw=1.2, label="posterior median")
ax.set(title=r"NPE-recovered $R_t$, CA 2019-2020",
       xlabel="Week of season", ylabel=r"$R_t$", ylim=(0.4, 2.5))
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "m3_sbi_real.png"), dpi=150)
plt.close(fig)
print("  saved figures/m3_sbi_real.png")


# ---------------------------------------------------------------------------
# 5. Persist Rt estimates
# ---------------------------------------------------------------------------
out = pd.DataFrame({
    "week": weeks,
    "week_end": real_df.week_end.values[:T_OBS],
    "Rt_mean": Rt_real_med,
    "Rt_lo": Rt_real_lo,
    "Rt_hi": Rt_real_hi,
})
out.to_csv(os.path.join(DATA_DIR, "rt_sbi.csv"), index=False)
print("  saved data/processed/rt_sbi.csv")

print("\nMethod 3 (Neural SBI) done.")
