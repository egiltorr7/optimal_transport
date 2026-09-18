# Running & Presenting Experiments

This document explains how to configure experiments, run them, and generate clean results for presentations.

---

## 1. Defining a New Experiment

Each experiment is a single config file in `config/`. To add a new experiment:

1. Copy an existing config file
2. Change `cfg.name` to something unique and descriptive
3. Swap any function handles or hyperparameters you want to change
4. Add it to the list in `main.m`

That's it. No other files need to change.

```matlab
% config/cfg_discfirst_projRiemannian_proxSplit.m
function cfg = cfg_discfirst_projRiemannian_proxSplit()
    cfg.name       = 'discfirst_projRiemannian_proxSplit';
    cfg.pipeline   = @discretize_then_optimize;
    cfg.projection = @proj_riemannian;
    cfg.prox_ke    = @prox_ke_splitting;
    cfg.disc       = @disc_symplectic;

    cfg.prox_stepsize = 0.005;
    cfg.max_iter      = 1000;
    cfg.tol           = 1e-7;
end
```

---

## 2. Running Experiments

### Run everything

Open `main.m`, add your config to the list, and run.

```matlab
% main.m
configs = {
    cfg_optfirst_projPenalty_proxExact(),
    cfg_optfirst_projLagrangian_proxExact(),
    cfg_discfirst_projPenalty_proxExact(),
    cfg_discfirst_projRiemannian_proxSplit(),
    % add more here...
};

problem = setup_problem();

results = cell(numel(configs), 1);
for i = 1:numel(configs)
    cfg = configs{i};
    fprintf('[%d/%d] Running: %s\n', i, numel(configs), cfg.name);
    results{i} = run_experiment(cfg, problem);
    save_result(results{i});
end

compare_results(results);
```

### Run a single experiment

```matlab
cfg     = cfg_optfirst_projPenalty_proxExact();
problem = setup_problem();
result  = run_experiment(cfg, problem);
```

### Re-load a saved result

```matlab
data   = load('results/raw/optfirst_projPenalty_proxExact.mat');
result = data.result;
% result.cfg tells you exactly what produced this result
```

---

## 3. Available Methods

### Pipelines (`pipelines/`)

| Function | Description |
|---|---|
| `optimize_then_discretize` | Optimize in continuous space, discretize at the end |
| `discretize_then_optimize` | Discretize first, then optimize the discrete system |

### Projection Methods (`projection/`)

| Function | Description |
|---|---|
| `proj_penalty` | Quadratic penalty relaxation |
| `proj_lagrangian` | Augmented Lagrangian / ADMM |
| `proj_riemannian` | Riemannian gradient projection |
| `proj_splitting` | Operator splitting projection |

### Prox Operators — Kinetic Energy (`prox/`)

| Function | Description |
|---|---|
| `prox_ke_exact` | Exact closed-form prox |
| `prox_ke_linearized` | Linearized approximation |
| `prox_ke_splitting` | Splitting-based prox |
| `prox_ke_iterative` | Iterative inner solver |

### Discretization Schemes (`discretization/`)

| Function | Description |
|---|---|
| `disc_euler` | Forward Euler |
| `disc_runge_kutta` | 4th-order Runge-Kutta |
| `disc_symplectic` | Symplectic integrator (structure-preserving) |

---

## 4. Generating Comparison Results

`compare_results(results)` produces:

- A **summary table** printed to the console (and optionally saved as CSV)
- A **figure grid** saved to `results/figures/`

### Summary Table Format

```
Method                              Iters       Error      Time(s)
optfirst_projPenalty_proxExact        312    4.21e-07       1.832
optfirst_projLagrangian_proxExact     198    2.11e-07       2.104
discfirst_projPenalty_proxExact       445    8.73e-07       0.971
discfirst_projRiemannian_proxSplit    201    1.05e-07       3.412
```

### Figure Layout

Figures are organized as a grid: one column per method, one row per metric (trajectory, convergence curve, constraint violation, etc.).

---

## 5. Tips for Advisor Presentations

- **Always present the summary table first** — it gives a quick overview before diving into figures.
- **Group results by what you're varying** (e.g. "here we fix the pipeline and vary the projection method") rather than showing all combinations at once.
- **Each saved `.mat` file is self-contained** — the `result.cfg` field records exactly what was run, so results are always reproducible and auditable.
- **Add a `notes` field to cfg** for anything worth remembering:
  ```matlab
  cfg.notes = 'Penalty weight tuned by grid search; symplectic disc required smaller dt';
  ```

---

## 6. Adding a Completely New Method

To add a new projection method (same process for prox, disc, or pipeline):

1. Create `projection/proj_mymethod.m` following the interface in [CONVENTIONS.md](CONVENTIONS.md)
2. Create a config file `config/cfg_..._projMymethod_....m` using `@proj_mymethod`
3. Add the config to `main.m`
4. Run

**No existing files need to change.**
