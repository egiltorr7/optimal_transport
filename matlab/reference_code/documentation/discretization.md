# Discretization and Grids

## Staggered Grid Overview

The space-time domain `[0,1] × [0,1]` is discretized using a **staggered (MAC-type)
grid**. Different variables live at different grid locations. This is essential for the
structure-preserving properties of the discretization.

### Grid Parameters

```
nx   = number of spatial cells
nt   = number of temporal slabs (not counting boundaries)
dx   = 1/nx    (spatial cell width)
dt   = 1/nt    (temporal slab height)
```

### Spatial Grid Points

```
Cell centers (phi, rho): x_i = (i - 0.5)*dx,   i = 1, ..., nx
Cell faces   (m):        x_{i+1/2} = i*dx,      i = 0, ..., nx
```

The spatial face grid has `nx+1` points including the two boundary faces at `x=0` and
`x=1`. The momentum `m` is stored at the `nx-1` interior faces (`nxm = nx-1`), since
the boundary faces are fixed to zero (no-flux BC).

### Temporal Grid Points

```
Cell faces (rho, phi):   t_k = k*dt,        k = 0, 1, ..., nt
Cell centers (m):        t_{k+1/2} = (k+0.5)*dt,   k = 0, ..., nt-1
```

The density `rho` is stored at the `nt-1` interior time faces (`ntm = nt-1`), since
`rho(0,·) = rho0` and `rho(1,·) = rho1` are given boundary data.

### Variable Array Sizes

| Variable | Size | Meaning |
|----------|------|---------|
| `rho` | `(nt-1) × nx` | density at interior time faces, all spatial cell centers |
| `mx` | `nt × (nx-1)` | momentum at all temporal cell centers, interior spatial faces |
| `rho_tilde` | same as `rho` | dual variable for density |
| `bx` | same as `mx` | dual variable for momentum |

### Cell-Center Variables (for proximal step)

The kinetic energy `m²/rho` is evaluated at the space-time **cell centers** —
the intersection of temporal cell-center slabs and spatial cell centers:

| Variable | Size | Meaning |
|----------|------|---------|
| `rho` at cell center | `nt × nx` | density interpolated to cell centers |
| `mx` at cell center | `nt × nx` | momentum interpolated to cell centers |

These are obtained by averaging adjacent faces.

## Staggered Layout Diagram

```
Temporal direction (t) ↑

t=1 ───── rho1 (BC) ──────────────────────────────────
         |         |         |         |         |
         |  cell   |  cell   |  cell   |  cell   |     ← temporal cell centers
         |  (1,1)  |  (1,2)  |  (1,3)  |  (1,4)  |       mx lives here
         |         |         |         |         |
t_{3/2}  ─── rho ─── rho ─── rho ─── rho ─── rho ──    ← rho at interior time faces
         |         |         |         |         |
         |  cell   |  cell   |  cell   |  cell   |
         |  (0,1)  |  (0,2)  |  (0,3)  |  (0,4)  |
         |         |         |         |         |
t=0  ──── rho0 (BC) ──────────────────────────────────

         x_{1/2}  x_{3/2}  x_{5/2}  x_{7/2}  x_{9/2}   ← cell centers (rho)
        x_0      x_1      x_2      x_3      x_4      x_5 ← faces (mx, BC at x_0, x_5)
                           →  x  →
```

## Finite Difference Operators

All operators are first-order, using adjacent points. The four basic stencils are:

### `deriv_t_at_phi` — time derivative, result at time-face locations
Acts on `rho` (at interior time faces), adds BC at `t=0` and `t=1`:
```
(D_t rho)_k = (rho_k - rho_{k-1}) / dt
boundary:  rho_0 = rho0,  rho_{nt} = rho1
```

### `deriv_t_at_rho` — time derivative, result at time-cell-center locations
Acts on a field at time faces, outputs at time cell centers:
```
(D_t phi)_{k+1/2} = (phi_{k+1} - phi_k) / dt
```

### `deriv_x_at_phi` — spatial derivative, result at spatial-face locations
Acts on `rho` (at cell centers), adds BC at `x=0` and `x=1`:
```
(D_x rho)_{i+1/2} = (rho_{i+1} - rho_i) / dx
boundary:  rho_0 and rho_{nx} specified
```

### `deriv_x_at_m` — spatial derivative, result at spatial-cell-center locations
Acts on `mx` (at faces), outputs at cell centers:
```
(D_x m)_i = (m_{i+1/2} - m_{i-1/2}) / dx
```

## Interpolation Operators

Used to move variables between staggered locations for computing `m²/rho`.

### `interp_t_at_phi` — interpolate to time-face location
Averages adjacent time-cell-center values; adds half the BC values at the ends:
```
(I_t f)_k = 0.5*(f_{k-1/2} + f_{k+1/2})
boundary:  (I_t f)_0 = 0.5*bc_start + 0.5*f_{1/2}
           (I_t f)_{nt} = 0.5*f_{nt-1/2} + 0.5*bc_end
```

### `interp_t_at_rho` — interpolate to time-cell-center location
Averages adjacent time-face values:
```
(I_t f)_{k+1/2} = 0.5*(f_k + f_{k+1})
```

### `interp_x_at_phi` and `interp_x_at_m`
Analogous operators in the spatial direction.

## Divergence Constraint (Discrete)

The continuity equation `∂_t rho + ∂_x m = 0` is discretized as:

```
(D_t rho)_{k, i+1/2-centerx} + (D_x m)_{k+1/2-centert, i} = 0
```

More precisely, at each space-time cell the constraint is:
```
(rho_{k+1, i+1/2} - rho_{k, i+1/2})/dt + (m_{k+1/2, i+1} - m_{k+1/2, i})/dx = 0
```

This discrete divergence lives at the **cell-center** of each space-time cell, and is
computed by `deriv_t_at_phi(rho, rho0, rho1) + deriv_x_at_phi(mx, 0, 0)`.

## Discrete Laplacian Eigenvalues

The Poisson solve in the projection step (see `algorithm.md`) is done spectrally using
the DCT. The eigenvalues of the 2D negative Laplacian with Neumann BCs on a uniform
grid are:

```
lambda_x(k) = (2 - 2*cos(pi*dx*k)) / dx²,   k = 0, ..., nx-1
lambda_t(k) = (2 - 2*cos(pi*dt*k)) / dt²,   k = 0, ..., nt-1

lambda_lap(k,l) = lambda_x(l) + lambda_t(k)
```

For the biharmonic-modified projection, the eigenvalues become:
```
lambda_bih(k,l) = lambda_x(l) + lambda_t(k) - vareps² * lambda_x(l)²
```

The DCT-based fast solve has complexity `O(nx * nt * log(nx * nt))`.
