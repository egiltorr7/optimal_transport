# Implementation Notes

## File Structure

```
1d/
├── sb1d_admm.m          Main solver function
├── solve_cubic.m        Analytic cubic root finder (vectorized)
├── mirt_dctn.m          N-D Discrete Cosine Transform
├── mirt_idctn.m         N-D Inverse DCT
├── test_operators.m     Sanity checks for staggered operators
├── test_adjoints.m      Adjoint consistency tests
├── check_convergence.m  Post-processing / convergence plots
└── documentation/       This folder
```

## Main Solver: `sb1d_admm.m`

```matlab
[rho, mx, outs] = sb1d_admm(rho0, rho1, opts)
```

**Inputs:**
- `rho0` — `(1 × nx)` initial density (must integrate to ~1)
- `rho1` — `(1 × nx)` terminal density (must integrate to ~1)
- `opts` — struct with fields:
  - `nt` — number of temporal slabs
  - `gamma` — ADMM penalty parameter
  - `maxIter` — maximum iterations
  - `vareps` — diffusion regularization coefficient
  - `rho_star`, `mx_star` — (optional) reference solution for error tracking

**Outputs:**
- `rho` — `(nt-1) × nx` density at interior time-faces
- `mx` — `nt × (nx-1)` momentum at temporal cell-centers, interior spatial faces
- `outs.residual_diff` — iteration history of the iterative residual
- `outs.true_error` — (if reference provided) iteration history of true error

## Key Internal Functions

### Interpolation Operators

All four interpolation operators are implemented as Toeplitz matrix-vector products:

```matlab
interp_t_at_phi(in, bc_start, bc_end)  % face → cell center in time, with BCs
interp_t_at_rho(in)                    % cell center → face in time
interp_x_at_phi(in, bc_in, bc_out)     % face → cell center in space, with BCs
interp_x_at_m(in)                      % cell center → face in space
```

The matrices are simple bidiagonal averaging (0.5, 0.5 pattern). Boundary conditions
are added by modifying the first/last rows: the BC value enters with weight 0.5.

**Adjoint note:** When computing `A^T v` as part of a gradient step, the BC arguments
must be **zero**, even if the forward operator uses nonzero BCs. This is because the
adjoint of the affine operator `A*x + b` is the linear part `A^T`, not `A^T(·+b)`.

### Derivative Operators

```matlab
deriv_t_at_phi(in, bc_start, bc_end)   % backward difference in time → face
deriv_t_at_rho(in)                     % forward difference in time → cell center
deriv_x_at_phi(in, bc_in, bc_out)      % backward difference in space → face
deriv_x_at_m(in)                       % forward difference in space → cell center
```

All differences are normalized by `dt` or `dx` respectively.

### `proj_div_biharmonic(mu, psi)`

Computes the orthogonal projection onto the set of `(rho, m)` satisfying the
discretized continuity equation:

1. Compute RHS: `f = -(D_t mu + D_x psi) + vareps * Δ_x mu`
   (where `Δ_x mu = D_x_at_phi(D_x_at_m(mu))`)
2. Solve the biharmonic-Poisson system spectrally: `phi = DCT_solve(f)`
3. Apply the correction:
   `rho_out = mu - D_t phi - vareps * Δ_x phi`
   `m_out   = psi - D_x phi`

The diffusion correction term `vareps * Δ_x mu` in the RHS accounts for the Fokker–Planck
structure; its boundary values use the Laplacian of `rho0` and `rho1`.

### `solve_cubic(a, b, c, d)`

Vectorized analytic cubic solver using Cardano's method. Handles the three cases:
- `delta > 0`: one real root (Cardano formula)
- `delta = 0`: double root
- `delta < 0`: three real roots (trigonometric method; largest real root is taken)

Returns the real root corresponding to the **positive density** solution of the
proximal problem.

## Initialization

```matlab
% Linear interpolation between rho0 and rho1
rho = (1-tt) .* rho0 + tt .* rho1;   % (nt-1) × nx
mx  = zeros(nt, nx-1);                % zero momentum
```

Dual variables start at zero. This is a natural "warm start" since linear interpolation
satisfies the continuity equation (with the trivial zero-momentum field for constant
total mass).

## DCT-Based Fast Solve

`mirt_dctn` / `mirt_idctn` compute the N-D Type-II DCT. The discrete Neumann Laplacian
on a uniform grid diagonalizes in the DCT basis. The eigenvalues are precomputed once
outside the iteration loop:

```matlab
lambda_x = (2 - 2*cos(pi*dx*(0:nx-1))) / dx^2;   % (1 × nx)
lambda_t = (2 - 2*cos(pi*dt*(0:nt-1))) / dt^2;   % (nt × 1)
lambda_lap_biharmonic = lambda_x + lambda_t - vareps^2 * lambda_x.^2;
```

The (0,0) eigenvalue is zero (constant mode); it is set to a safe value before division
and the corresponding coefficient is zeroed afterward (fixing the gauge).

## Typical Usage

```matlab
nx = 128; nt = 64;
x  = ((1:nx) - 0.5) / nx;

rho0 = normpdf(x, 0.3, 0.05); rho0 = rho0 / sum(rho0) * nx;
rho1 = normpdf(x, 0.7, 0.05); rho1 = rho1 / sum(rho1) * nx;

opts.nt      = nt;
opts.gamma   = 10;
opts.maxIter = 2000;
opts.vareps  = 1e-4;

[rho, mx, outs] = sb1d_admm(rho0, rho1, opts);

semilogy(outs.residual_diff);
xlabel('Iteration'); ylabel('Residual');
```

## Known Issues / Design Decisions

- **Negative density clamping**: densities below `1e-12` are set to zero and the
  corresponding momentum is also zeroed. This is a hard truncation, not a soft
  regularization; it can affect convergence near vacuum regions.

- **Stability condition**: the ratio `tau/gamma` should exceed `||A^T A||_2 ≈ 1`
  with comfortable margin (recommend 1.5×). Too-small margin causes oscillatory
  convergence stalling around 1e-7 residual.

- **Adjoint correctness**: interpolation adjoints must use zero BCs. Using nonzero
  BCs in the adjoint corrupts the gradient direction and causes convergence to an
  incorrect solution (see `ADJOINT_BUG_FIX.md`).
