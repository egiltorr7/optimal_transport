# Optimization Experiments — Project Overview

This project systematically compares two high-level strategies for optimal control:

- **Optimize-then-Discretize**: optimize in continuous space, then discretize the solution
- **Discretize-then-Optimize**: discretize the problem first, then optimize the resulting finite-dimensional system

Within each strategy, we independently swap:
- **Projection enforcement method** (how constraints are satisfied)
- **Proximal operator for the kinetic energy term** (how the KE prox step is computed)
- **Discretization scheme** (Euler, Runge-Kutta, symplectic, etc.)

---

## Folder Structure

```
project/
│
├── README.md                        ← you are here
├── CONVENTIONS.md                   ← coding standards and interface contracts
├── EXPERIMENTS.md                   ← how to configure and run experiments
│
├── main.m                           ← master runner
├── run_experiment.m                 ← single experiment runner
├── compare_results.m                ← comparison table + figures
├── setup_problem.m                  ← problem definition
│
├── config/                          ← one file per experiment configuration
│   ├── cfg_optfirst_projA_proxExact.m
│   ├── cfg_optfirst_projB_proxLinear.m
│   ├── cfg_discfirst_projA_proxExact.m
│   └── ...
│
├── pipelines/                       ← top-level strategy (outer loop)
│   ├── optimize_then_discretize.m
│   └── discretize_then_optimize.m
│
├── projection/                      ← swappable projection methods
│   ├── proj_penalty.m
│   ├── proj_lagrangian.m
│   ├── proj_riemannian.m
│   └── proj_splitting.m
│
├── prox/                            ← swappable prox operators (kinetic energy)
│   ├── prox_ke_exact.m
│   ├── prox_ke_linearized.m
│   ├── prox_ke_splitting.m
│   └── prox_ke_iterative.m
│
├── discretization/                  ← discretization schemes
│   ├── disc_euler.m
│   ├── disc_runge_kutta.m
│   └── disc_symplectic.m
│
├── results/
│   ├── raw/                         ← .mat files, one per experiment (auto-saved)
│   └── figures/                     ← exported plots
│
└── utils/
    ├── plot_comparison.m
    ├── compute_metrics.m
    └── save_result.m
```

---

## Quick Start

1. Define your problem once in `setup_problem.m`
2. Create config files in `config/` (copy an existing one and modify)
3. Add your configs to the list in `main.m`
4. Run `main.m` — results are saved to `results/raw/` and figures to `results/figures/`

See [EXPERIMENTS.md](EXPERIMENTS.md) for a full walkthrough.  
See [CONVENTIONS.md](CONVENTIONS.md) for interface contracts and coding standards.
