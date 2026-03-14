# Coding Conventions & Interface Contracts

This document defines the rules every function in this project must follow.
**Read this before writing any new function.**

---

## Core Philosophy

> One function = one method. The config struct decides what runs — never hardcode strategy logic inside a function.

The entire project is built on one pattern: every swappable component follows a fixed interface so it can be dropped in anywhere without changing any other code.

---

## The Config Struct

Every experiment is fully described by a single `cfg` struct. It holds both **function handles** (what method to use) and **hyperparameters** (how to tune it). Nothing else should be needed to reproduce a run.

### Required Fields

| Field | Type | Description |
|---|---|---|
| `cfg.name` | `string` | Unique identifier for this experiment (used as filename) |
| `cfg.pipeline` | `function handle` | `@optimize_then_discretize` or `@discretize_then_optimize` |
| `cfg.projection` | `function handle` | e.g. `@proj_penalty` |
| `cfg.prox_ke` | `function handle` | e.g. `@prox_ke_exact` |
| `cfg.disc` | `function handle` | e.g. `@disc_euler` |
| `cfg.max_iter` | `int` | Maximum number of iterations |
| `cfg.tol` | `double` | Convergence tolerance |

Any method-specific hyperparameters (e.g. `cfg.penalty_weight`, `cfg.prox_stepsize`) are also stored here.

### Example Config File

```matlab
% config/cfg_optfirst_projPenalty_proxExact.m
function cfg = cfg_optfirst_projPenalty_proxExact()
    cfg.name           = 'optfirst_projPenalty_proxExact';
    cfg.pipeline       = @optimize_then_discretize;
    cfg.projection     = @proj_penalty;
    cfg.prox_ke        = @prox_ke_exact;
    cfg.disc           = @disc_euler;

    % Hyperparameters
    cfg.penalty_weight = 1e3;
    cfg.prox_stepsize  = 0.01;
    cfg.max_iter       = 500;
    cfg.tol            = 1e-6;
end
```

---

## Interface Contracts

Every function in `pipelines/`, `projection/`, `prox/`, and `discretization/` **must** follow these exact signatures. Do not deviate.

### Pipelines

```matlab
result = my_pipeline(cfg, problem)
```

**Inputs:**
- `cfg` — the full config struct
- `problem` — the problem struct (see Problem Struct below)

**Output:** a `result` struct containing at minimum:

| Field | Description |
|---|---|
| `result.trajectory` | Solution trajectory |
| `result.iters` | Number of iterations to convergence |
| `result.converged` | Boolean |
| `result.error` | Final convergence error |
| `result.walltime` | Wall-clock time (seconds) |
| `result.cfg` | Copy of the cfg used (for self-documentation) |

---

### Projection Methods

```matlab
x = proj_mymethod(x, problem, cfg)
```

**Inputs:**
- `x` — current iterate
- `problem` — problem struct
- `cfg` — config struct (access method-specific params here)

**Output:** `x` projected onto the feasible set.

---

### Proximal Operators (Kinetic Energy)

```matlab
x = prox_ke_mymethod(x, stepsize, problem)
```

**Inputs:**
- `x` — current iterate
- `stepsize` — proximal stepsize (scalar)
- `problem` — problem struct

**Output:** `x` after applying the prox operator.

---

### Discretization Schemes

```matlab
trajectory = disc_myscheme(x, problem)
```

**Inputs:**
- `x` — continuous solution or initial condition
- `problem` — problem struct

**Output:** discrete trajectory (matrix, time steps × state dimension).

---

## The Problem Struct

Define the problem **once** in `setup_problem.m`. All functions read from it; none modify it.

```matlab
function problem = setup_problem()
    problem.x0     = ...;   % initial condition
    problem.T      = ...;   % time horizon
    problem.dt     = ...;   % time step (for discretization)
    problem.n      = ...;   % state dimension
    % add any problem-specific data (mass matrix, potential, constraints, etc.)
end
```

---

## The Result Struct

Results are self-documenting: always save `cfg` inside `result`. This means any saved `.mat` file is fully reproducible — you can look at it months later and know exactly what produced it.

```matlab
result.cfg = cfg;   % always include this
```

---

## Naming Conventions

| Component | Prefix | Example |
|---|---|---|
| Config files | `cfg_` | `cfg_optfirst_projPenalty_proxExact.m` |
| Pipelines | none | `optimize_then_discretize.m` |
| Projection methods | `proj_` | `proj_lagrangian.m` |
| Prox operators | `prox_ke_` | `prox_ke_linearized.m` |
| Discretization | `disc_` | `disc_symplectic.m` |

Config file names should be descriptive enough to identify the full experiment:
`cfg_{pipeline}_{projection}_{prox}.m`

---

## Rules

1. **Never use `if/switch` to choose methods inside a function.** That logic belongs in the config. If you find yourself writing `if strcmp(cfg.method, 'A')`, stop — make two separate functions instead.

2. **Never hardcode hyperparameters.** All tunable values live in `cfg`. Functions read from `cfg`; they never define magic numbers.

3. **Never modify the `problem` struct.** It is read-only. If a method needs derived quantities, compute them locally.

4. **Always save `cfg` in `result`.** No exceptions.

5. **Interfaces are fixed.** If you need extra inputs for a new method, use fields already in `cfg` or `problem`. Do not change function signatures.
