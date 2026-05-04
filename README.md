# Comparing Methods for Time-Varying Reproduction Number ($R_t$) Estimation

Likelihood-based vs simulation-based inference of $R_t$ for seasonal influenza in the United States.

This project pits three modern $R_t$-estimation methods against one another on the same data:

| Method | Framework | Implementation | Per-fit runtime |
|---|---|---|---|
| **EpiEstim** | Renewal equation, Bayesian | R · `EpiEstim` | < 1 s |
| **POMP + mif2** | Mechanistic SEIR, iterated filtering | R · `pomp` | ~ 3 s |
| **Neural SBI (NPE)** | Simulation-based, normalizing flow | Python · `sbi` | ~ 80 ms after training |

The methods are evaluated both on (i) a simulation study with known ground-truth $R_t$ and (ii) real CDC FluView data across multiple US states and seasons.

This is a methodological companion to research at the University of Michigan on `phylopomp` (likelihood) vs `PhyloDeep` (neural) phylodynamic inference. The same likelihood-vs-neural inference contrast is studied here in a simpler time-series regime.

## Headline figure

The simulation study cleanly separates the methods. POMP achieves the lowest mean absolute error when the model is correctly specified; NPE is roughly **30x faster** at inference time but pays for amortization with higher bias near prior boundaries; EpiEstim is the strongest model-free baseline.

![Method comparison on N=5 synthetic outbreaks](figures/m4_sim_metrics.png)

![One synthetic outbreak: three methods vs ground truth](figures/m4_sim_trajectories.png)

## Real-data application: California 2019–2020

EpiEstim resolves the abrupt April 2020 collapse of $R_t$ — a collateral effect of COVID-19 NPIs on flu transmission — that POMP's spline-based $\beta(t)$ smooths over.

![EpiEstim Rt for the 2019-2020 flu season across US states](figures/m1_rt_by_state.png)

## Repo layout

```
flu-rt-comparison/
├── R/
│   ├── 01_fetch_data.R         # pull CDC FluView via Delphi Epidata API
│   ├── 02_eda.R                # exploratory plots
│   ├── 03_epiestim.R           # Method 1
│   ├── 04_pomp_mif2.R          # Method 2
│   ├── helper_epiestim_one.R   # called from sim study
│   └── helper_pomp_one.R       # called from sim study
├── python/
│   ├── 05_sbi_simulator.py     # SEIR simulator + prior
│   ├── 06_sbi_inference.py     # Method 3
│   ├── 07_simulation_study.py  # head-to-head on synthetic outbreaks
│   └── requirements.txt
├── data/
│   ├── raw/                    # raw FluView pull
│   └── processed/              # cleaned + per-method Rt estimates
├── figures/
└── report/
    ├── report.qmd              # Quarto paper-style writeup
    ├── report.html             # rendered (4 MB, self-contained)
    └── references.bib
```

## Reproducing the analysis

```bash
# R packages
Rscript R/install_packages.R           # tidyverse, EpiEstim, pomp, ...

# Pull data + run R-side methods
Rscript R/01_fetch_data.R
Rscript R/02_eda.R
Rscript R/03_epiestim.R
Rscript R/04_pomp_mif2.R               # ~ 1 min

# Python env (Method 3 + simulation study)
python -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt
python python/06_sbi_inference.py      # ~ 5 min (trains NPE)
python python/07_simulation_study.py   # ~ 2 min

# Render report
quarto render report/report.qmd --to html
open report/report.html
```

## Key findings

1. **POMP + mif2 is most accurate on well-specified data** (MAE ≈ 0.09 Rt vs 0.39 for EpiEstim, 1.22 for NPE), but slowest per fit (~3 s).
2. **NPE delivers ~30x faster inference** than EpiEstim and ~40x faster than POMP through amortization, at the cost of bias near prior boundaries.
3. **EpiEstim is robust to serial-interval misspecification** — short / medium / long flu SI assumptions yield nearly identical $R_t$ trajectories.
4. **All three methods degrade in the low-incidence tail** of an outbreak: $R_t$ becomes weakly identifiable when $I_t \to 0$.

See `report/report.html` for the full writeup with figures, tables, and discussion.

## Related work

- **`phylopomp`** [@kingaa](https://github.com/kingaa/phylopomp) — likelihood-based phylodynamic inference (POMP on phylogenies)
- **`PhyloDeep`** [Voznica et al. 2022](https://www.nature.com/articles/s41467-022-31511-0) — neural phylodynamic inference

## License

MIT.
