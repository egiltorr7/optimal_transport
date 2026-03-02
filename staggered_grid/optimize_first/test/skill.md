# Skill: Implementing ADMM for Schrödinger Bridge

## Overview

This skill guides implementation of a 1D Schrödinger Bridge solver using Linearized ADMM (Formulation 2, two-grid approach). Read `context.md` first for the full mathematical background.

---

## Implementation Guidelines

### Language and Libraries
- Use **Python** with `numpy`, `scipy`, and `matplotlib`
- Use `scipy.fft.dst` and `scipy.fft.dct` for the spectral solvers
- Structure the code as a single well-documented file `sb_admm_1d.py`

### Code Structure

```
sb_admm_1d.py
├── build_operators(Nt, Nx, dt, dx, eps)     # Build all discrete operators
├── build_rhs(rho0, rho1, Nt, Nx, dt, dx)   # Build d vector
├── prox_J(u, v, dt, dx, tau)               # Pointwise proximal of kinetic energy
├── solve_phi(F, Nt, Nx, dt, dx, eps, ...)  # Solve M_SB phi = F via DCT + Woodbury
├── project_C(p_rho, p_b, ...)              # Projection onto constraint set C
├── interpolate_to_staggered(rho, m, ...)   # Interpolation operators A_rho, A_m
├── admm_solve(rho0, rho1, Nt, Nx, eps, gamma, tau, max_iter, tol)
└── main()                                   # Test cases + plots
```

---

## Key Implementation Details

### 1. Operator Construction

Build $D_t$, $A_t$, $D_x$, $L_x$ as sparse matrices using `scipy.sparse`.

**$D_t$** of size $N_t \times (N_t-1)$:
```python
# First row: [1, 0, ..., 0] / dt
# Interior rows: standard [-1, 1] / dt  
# Last row: [0, ..., 0, -1] / dt
```

**$A_t$** of size $N_t \times (N_t-1)$:
```python
# First row: [1, 0, ..., 0] / 2
# Interior rows: standard [1, 1] / 2
# Last row: [0, ..., 0, 1] / 2
```

**$D_x$** of size $N_x \times (N_x-1)$: same structure as $D_t$ but with $\Delta x$.

**$L_x = D_x D_x^\top$** of size $N_x \times N_x$: Neumann Laplacian.

### 2. Interpolation Operators $A$

$A_\rho$ maps $\rho$ (collocated, $N_t \times N_x$) to $\bar\rho$ grid ($(N_t-1) \times N_x$):
- Average adjacent time levels: $(\rho^i + \rho^{i+1})/2$ for $i = 0,\ldots,N_t-2$
- This gives interior time nodes $n = 1,\ldots,N_t-1$

$A_m$ maps $m$ (collocated, $N_t \times N_x$) to $b$ grid ($N_t \times (N_x-1)$):
- Average adjacent space cells: $(m^j + m^{j+1})/2$ for $j = 0,\ldots,N_x-2$

### 3. Proximal Operator of $J$

For each grid point, solve:
$$\min_{\rho \geq 0, m} \frac{\Delta t \Delta x}{2\rho} m^2 + \frac{\tau}{2}(\rho - u)^2 + \frac{\tau}{2}(m - v)^2$$

Optimality in $m$: $m^* = \frac{\tau \rho v}{\tau \rho + \frac{\Delta t \Delta x}{2}}$... 

Actually the standard approach: let $\alpha = \Delta t \Delta x / (2\tau)$. Then the problem reduces to finding the largest real root $\rho^* > 0$ of:

$$\rho^3 - u\rho^2 - \alpha v^2 \frac{\rho}{\rho + \alpha}... $$

Use the explicit formula from Papadakis et al. (2014): optimality conditions give:
- If $u \leq 0$: $\rho^* = 0$, $m^* = 0$
- Else: $\rho^*$ is the largest real root of $f(\rho) = \rho^3 - u\rho^2 + \frac{|v|^2}{2\tau^2/(\Delta t \Delta x)} = 0$... 

**Recommended**: use `numpy.roots` or `numpy.polynomial.polynomial.polyroots` for the cubic, taking the largest real root. Or use the analytical Cardano formula for speed.

The cubic to solve is (after eliminating $m$):
$$2\tau \rho^3 - 2\tau u \rho^2 + \frac{\Delta t \Delta x}{2} v^2 \cdot \frac{1}{?}$$

**Safe approach**: implement as a scalar Newton iteration per point — fast and robust.

### 4. Solving $M_{\text{SB}} \phi = F$

This is the core computational step. Algorithm:

```
1. Reshape F to (Nt, Nx) matrix
2. Apply DCT-II in space (axis=1) to get F_hat of shape (Nt, Nx)
3. For each spatial mode k = 0,...,Nx-1:
   a. Get lambda_x_k (eigenvalue of L_x)
   b. Build Toeplitz eigenvalues: mu_l = lambda_t_l + eps^2 * lambda_x_k^2 * lambda_A_l + lambda_xx_k
      where lambda_xx_k is eigenvalue of D_xD_x^T
   c. Compute Woodbury rank-2 correction (2x2 system)
   d. Apply (T^k)^{-1} via DST-I + pointwise division + inverse DST-I
   e. Apply Woodbury correction
4. Apply inverse DCT-II in space
5. Reshape back to vector
```

**Eigenvalues:**

Spatial (DCT basis, $L_x = D_x D_x^\top$):
$$\lambda^x_k = \frac{2}{\Delta x^2}\left(\cos\left(\frac{\pi k}{N_x}\right) - 1\right), \quad k = 0,\ldots,N_x-1$$

Temporal DST-I eigenvalues of $T_D$ (Toeplitz part of $D_t D_t^\top$):
$$\lambda^t_l = \frac{4}{\Delta t^2}\sin^2\left(\frac{\pi l}{2 N_t}\right), \quad l = 1,\ldots,N_t$$

Temporal DST-I eigenvalues of $T_A$ (Toeplitz part of $A_t A_t^\top$):
$$\lambda^A_l = \frac{1}{2} + \frac{1}{2}\cos\left(\frac{\pi l}{N_t}\right), \quad l = 1,\ldots,N_t$$

**Woodbury correction per mode $k$:**

Corner values of the rank-2 correction:
$$\sigma^k_1 = -\frac{1}{\Delta t^2} - \frac{\varepsilon^2(\lambda^x_k)^2}{4} + \frac{\varepsilon\lambda^x_k}{\Delta t}$$
$$\sigma^k_{N_t} = -\frac{1}{\Delta t^2} - \frac{\varepsilon^2(\lambda^x_k)^2}{4} - \frac{\varepsilon\lambda^x_k}{\Delta t}$$

$U = [e_1, e_{N_t}]$, so $U^\top (T^k)^{-1} U$ is a $2\times 2$ matrix extracting rows/columns 1 and $N_t$ of $(T^k)^{-1}$.

To get column $j$ of $(T^k)^{-1}$: apply $(T^k)^{-1}$ to $e_j$ via DST.

### 5. Vectorization

**Avoid Python loops over grid points.** Use numpy broadcasting:
- The prox step should be vectorized over all $(i,j)$ simultaneously
- The DCT/DST steps operate on full arrays at once
- Only loop over spatial modes $k$ in the Woodbury step (or vectorize further)

### 6. Density Positivity

Enforce $\rho > 0$ after the prox step. Use a small floor: `rho = np.maximum(rho, 1e-10)`.

### 7. Initial Conditions

Initialize:
```python
rho = np.ones((Nt, Nx)) * 0.5  # or linear interpolation between rho0 and rho1
m = np.zeros((Nt, Nx))
rho_bar = np.zeros((Nt-1, Nx))
b = np.zeros((Nt, Nx-1))
delta_rho = np.zeros_like(rho_bar)
delta_b = np.zeros_like(b)
```

---

## Validation Checks

Before running ADMM, verify:

1. **Operator sizes**: print shapes of $D_t$, $A_t$, $D_x$, $L_x$ and confirm they match expected sizes
2. **$M_{\text{SB}}$ positive definiteness**: for small $N_t, N_x$, build $M_{\text{SB}}$ explicitly and check eigenvalues > 0
3. **Solver accuracy**: solve $M_{\text{SB}} \phi = F$ for random $F$ and verify $\|M_{\text{SB}}\phi - F\| < 10^{-10}$
4. **Prox correctness**: verify prox solution satisfies optimality conditions
5. **Constraint satisfaction**: after convergence, verify Fokker-Planck residual < tol

---

## Plotting

Produce the following plots after convergence:

1. **Density evolution**: heatmap of $\rho(t, x)$ (x-axis: space, y-axis: time)
2. **Momentum field**: heatmap of $m(t, x)$
3. **Marginals check**: plot $\rho(0, \cdot)$ and $\rho(1, \cdot)$ vs $\rho_0$ and $\rho_1$
4. **Convergence plot**: primal and dual residuals vs iteration number
5. **Snapshots**: $\rho(t, \cdot)$ at $t = 0, 0.25, 0.5, 0.75, 1.0$

---

## Common Pitfalls

1. **Kronecker product ordering**: be consistent — use (time, space) ordering throughout, i.e. time is the slow index, space is the fast index when vectorizing
2. **DST-I vs DST-II**: use DST type 1 for the Toeplitz temporal solve (homogeneous Dirichlet structure)
3. **DCT type**: use DCT type 2 for the spatial solve (Neumann BCs)
4. **Normalization**: `scipy.fft.dst` and `scipy.fft.dct` have normalization options — use `norm='ortho'` for clean inverses
5. **$\varepsilon = 0$ case**: should reduce exactly to OT — verify this first before testing $\varepsilon > 0$
