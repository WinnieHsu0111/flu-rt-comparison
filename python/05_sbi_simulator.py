"""
05_sbi_simulator.py
-------------------
Stochastic SEIR simulator that mirrors the R/POMP model in 04_pomp_mif2.R,
but with a low-dimensional, time-varying R(t) parameterization that the
neural posterior estimator (NPE) in 06_sbi_inference.py will invert.

Parameter vector (8-dim):
    theta = [r1, r2, r3, r4, rho, k, eta, I0_log]
    - r1..r4   : log(Rt) at four spline knots over the season (cubic spline)
    - rho      : reporting fraction
    - k        : NB overdispersion
    - eta      : initial susceptible fraction
    - I0_log   : log10 of initial infected count

Output: weekly reported ILI counts of length T (= 52 weeks).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline

# ---- fixed disease parameters (in 1/week) ---------------------------------
SIGMA_FIX = 7.0 / 1.5   # 1 / 1.5d latent period
GAMMA_FIX = 7.0 / 3.0   # 1 / 3d infectious period
N_POP    = 39_500_000   # CA population
T_OBS    = 52           # weeks per season
DT       = 1 / 7        # daily Euler step within each weekly observation


def beta_trajectory(r_knots: np.ndarray, n_weeks: int = T_OBS) -> np.ndarray:
    """Cubic-spline interpolation of log(Rt) onto a weekly grid."""
    knot_x = np.linspace(0, n_weeks - 1, len(r_knots))
    spline = CubicSpline(knot_x, r_knots, bc_type="natural")
    log_Rt = spline(np.arange(n_weeks))
    Rt = np.exp(log_Rt)
    beta = Rt * GAMMA_FIX
    return beta, Rt


def simulate_one(theta: np.ndarray, rng: np.random.Generator,
                 return_S: bool = False):
    """Run one stochastic SEIR trajectory and return weekly reported counts.

    If return_S=True, also returns the weekly S(t) at the start of each week.
    """
    r1, r2, r3, r4, rho, k_disp, eta, I0_log = theta
    beta_weekly, _ = beta_trajectory(np.array([r1, r2, r3, r4]))

    S = int(eta * N_POP)
    I = max(int(round(10 ** I0_log)), 1)
    E = 0
    R = N_POP - S - I

    reports = np.zeros(T_OBS, dtype=np.int64)
    S_traj  = np.zeros(T_OBS, dtype=np.int64)

    for t in range(T_OBS):
        S_traj[t] = S
        H = 0
        beta_t = beta_weekly[t]
        for _ in range(7):
            rate_SE = beta_t * I / N_POP
            n_SE = rng.binomial(S, 1 - np.exp(-rate_SE * DT))
            n_EI = rng.binomial(E, 1 - np.exp(-SIGMA_FIX * DT))
            n_IR = rng.binomial(I, 1 - np.exp(-GAMMA_FIX * DT))
            S -= n_SE
            E += n_SE - n_EI
            I += n_EI - n_IR
            R += n_IR
            H += n_IR

        mu = max(rho * H, 1e-6)
        p = k_disp / (k_disp + mu)
        reports[t] = rng.negative_binomial(k_disp, p)

    if return_S:
        return reports, S_traj
    return reports


def simulator_torch(theta_batch: torch.Tensor) -> torch.Tensor:
    """Vectorized wrapper for `sbi`. Accepts a (B, 8) tensor, returns (B, T)."""
    theta_np = theta_batch.detach().cpu().numpy()
    rng = np.random.default_rng()
    out = np.zeros((theta_np.shape[0], T_OBS), dtype=np.float32)
    for i, th in enumerate(theta_np):
        out[i] = simulate_one(th, rng).astype(np.float32)
    return torch.from_numpy(out)


# ---- prior bounds (uniform) -----------------------------------------------
# r_k = log Rt, so Rt in [0.5, 3] -> log Rt in [log 0.5, log 3] ~ [-0.7, 1.1]
PRIOR_LOW  = np.array([-0.7, -0.7, -0.7, -0.7, 0.001, 1.0,  0.5, 0.0])
PRIOR_HIGH = np.array([ 1.1,  1.1,  1.1,  1.1, 0.05,  20.0, 0.99, 4.0])
PARAM_NAMES = ["log_R1", "log_R2", "log_R3", "log_R4",
               "rho", "k", "eta", "log10_I0"]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    theta = np.array([np.log(1.3), np.log(1.5), np.log(0.9), np.log(0.7),
                      0.01, 10.0, 0.95, 1.0])
    y = simulate_one(theta, rng)
    print("Sample simulated trajectory (first 12 weeks):", y[:12])
    print("Total reports:", y.sum())
