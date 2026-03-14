# Algorithm: ADMM for Optimal Transport

## High-Level Structure

The discretized problem has the saddle-point form:

```
min_{rho, m}  f(rho, m)  +  g(rho_tilde, bx)
subject to:   (rho, m) = (rho_tilde, bx)
```

where:
- `f(rho, m)` = kinetic energy `∑ m²/rho * dx * dt` (defined at cell centers)
- `g(rho_tilde, bx)` = indicator of the set where the continuity equation holds
  (i.e., `g = 0` if `∂_t rho_tilde + ∂_x bx = 0`, else `g = +∞`)

The two auxiliary variables `(rho_tilde, bx)` store the "divergence-free" copy;
`(rho, m)` store the "kinetic-energy-feasible" copy. The algorithm alternates
between optimizing each part.

## ADMM Iteration

The update equations at iteration `k` are:

```
(1) Prox step:    (rho, m)  ← prox_{f/gamma}(rho_tilde + delta_rho/gamma,
                                               bx        + delta_mx/gamma)

(2) Proj step:    (rho_tilde, bx) ← proj_{div-free}(rho - delta_rho/gamma,
                                                      m   - delta_mx/gamma)

(3) Dual update:  delta_rho ← delta_rho - gamma*(rho - rho_tilde)
                  delta_mx  ← delta_mx  - gamma*(m   - bx)
```

Here `gamma > 0` is the ADMM penalty parameter (called `gamma` in the code).

## Step 1: Proximal Operator for Kinetic Energy

The proximal operator of `f(rho, m) = ∑ m²/(2*rho)` with parameter `sigma = 1/gamma`
is computed **pointwise** at each space-time cell center.

Given inputs `(r̃, m̃)`, we solve:

```
min_{r,m}  m²/(2r)  +  (gamma/2)*[(r - r̃)² + (m - m̃)²]
```

Optimality conditions give `m = r * m̃ / (r + sigma)`, and `r` satisfies the cubic:

```
r³ + (2σ - r̃)r² + (σ² - 2σr̃)r - σ(σr̃ + m̃²/2) = 0
```

The code calls `solve_cubic` for this. The real positive root is selected; any negative
density values are clamped to zero.

### Grid Interpolation in the Prox Step

The proximal step operates at space-time cell centers, while `rho` and `mx` live on the
staggered grid. The code therefore:

1. Interpolates inputs to cell centers via `interp_t_at_phi` and `interp_x_at_phi`.
2. Applies the pointwise prox at cell centers.
3. Interpolates results back to the staggered grid via `interp_t_at_rho` and `interp_x_at_m`.

## Step 2: Projection onto the Divergence-Free Set

Given `(mu, psi)`, find `(rho_out, m_out)` such that:

```
∂_t rho_out + ∂_x m_out = 0,  with  rho_out(0,·) = rho0,  rho_out(1,·) = rho1
m_out(t,0) = m_out(t,1) = 0
```

This is an orthogonal projection in the discrete L²-inner product. Introducing a
potential `phi`, the correction is:

```
rho_out = mu  - ∂_t phi  - vareps * ∂_xx phi
m_out   = psi - ∂_x phi
```

Substituting into the continuity equation gives the elliptic PDE for `phi`:

```
-∂_t(∂_t phi) - ∂_x(∂_x phi) + vareps * ∂_x(∂_x(∂_t phi)) =
    -(∂_t mu + ∂_x psi - vareps * ∂_xx mu)
```

which in the code reduces to:

```
(-Δ - vareps² * ∂_xxxx) phi = -(∂_t mu + ∂_x psi) + vareps * ∂_xx mu
```

where `Δ = ∂_tt + ∂_xx`. This is the **biharmonic-modified Poisson equation**.

### Fast Solve via DCT

The equation is solved spectrally using the 2D Discrete Cosine Transform (DCT) with
Neumann boundary conditions. In frequency space the operator is diagonal:

```
phi_hat(k,l) = RHS_hat(k,l) / (lambda_t(k) + lambda_x(l) - vareps² * lambda_x(l)²)
```

The zero-frequency mode `(k=0, l=0)` is set to zero (free gauge).

### Special Case: vareps = 0

When `vareps = 0`, the equation is just `-Δ phi = RHS`, the standard Poisson problem,
solved by DCT in `O(N log N)` time.

## Convergence Criterion

The code tracks the **iterative residual**:

```
residual_diff(iter) = sqrt(dt*dx * (||rho_tilde_new - rho_tilde||² + ||bx_new - bx||²))
```

This measures the change in the divergence-free variable between successive iterates.
Convergence is declared when this falls below a tolerance `opts.vareps` (or after
`opts.maxIter` iterations).

An optional **true error** can be computed if a reference solution `(rho_star, mx_star)`
is provided via `opts.rho_star` and `opts.mx_star`.

## Parameter Guidance

| Parameter | Typical value | Effect |
|-----------|---------------|--------|
| `gamma` | 1–100 | ADMM step size; larger values may converge faster but less stably |
| `vareps` | 1e-4 – 1e-2 | Diffusion regularization; smaller = closer to W₂ OT |
| `maxIter` | 500–5000 | Maximum iterations |

Convergence of the extra-gradient ADMM variant used here requires:

```
tau / gamma > ||A^T A||₂
```

where `A` is the staggered interpolation operator. Since `||A||₂ ≈ 1`, the ratio
`tau/gamma` should be comfortably above 1 (e.g., 1.5×) for robust convergence.
