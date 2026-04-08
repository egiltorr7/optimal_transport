# Problem Setup

## Continuous Formulation

We seek a time-evolving density `rho(t,x)` and momentum density `m(t,x)` on the
space-time domain `[0,1] × [0,1]` that solve the **dynamic optimal transport** problem:

```
min  ∫₀¹ ∫₀¹  m(t,x)² / rho(t,x)  dx dt
rho,m

subject to:
  ∂_t rho + ∂_x m = 0            (continuity equation)
  rho(0, x) = rho0(x)            (initial condition)
  rho(1, x) = rho1(x)            (terminal condition)
  m(t, 0) = m(t, 1) = 0          (no-flux at spatial boundaries)
  rho(t, x) ≥ 0                  (non-negativity)
```

The integrand `|m|²/rho` is the **kinetic energy density**; minimizing it corresponds
to the Benamou–Brenier formulation of the 2-Wasserstein distance between `rho0` and
`rho1`. When `rho0` and `rho1` are given probability densities, the optimal value of the
objective equals `W₂(rho0, rho1)²`.

## Regularization

An optional diffusion regularization controlled by parameter `vareps` modifies the
constraint to a **Fokker–Planck equation**:

```
∂_t rho + ∂_x m = vareps * ∂_xx rho
```

or equivalently in divergence form (replacing `m` with a corrected flux). In the code
this appears as a biharmonic term in the projection step (see `proj_div_biharmonic`),
which penalizes spatial non-smoothness of `rho` and connects to the **Schrödinger
bridge** problem at positive `vareps`.

As `vareps → 0` the problem reduces to unregularized optimal transport (Wasserstein-2).

## Variables

| Symbol | Meaning | Grid location |
|--------|---------|---------------|
| `rho(t,x)` | probability density | cell centers in both t and x |
| `m(t,x)` | momentum density (= rho * velocity) | cell centers in t, cell faces in x |
| `phi(t,x)` | potential (Lagrange multiplier for continuity eq.) | cell faces in t, cell centers in x |

## Boundary Conditions

- **Time boundaries**: `rho(0,·) = rho0` and `rho(1,·) = rho1` are prescribed (Dirichlet).
- **Spatial boundaries**: `m(t,0) = m(t,1) = 0` (homogeneous Dirichlet for momentum,
  equivalent to no-flux Neumann for density).

## Equivalent Dual Formulation

The Lagrangian of the problem introduces a potential `phi(t,x)` as a multiplier for
the continuity equation. The optimality conditions give:

```
m = rho * ∂_x phi
∂_t phi + ½|∂_x phi|² = 0    (Hamilton–Jacobi equation)
∂_t rho + ∂_x(rho ∂_x phi) = 0  (continuity equation)
```

This is the classical fluid mechanics interpretation: particles follow characteristics
of the velocity field `v = ∂_x phi`.
