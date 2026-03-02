# Schrödinger Bridge Project — Claude Instructions

This project implements a 1D dynamic Schrödinger Bridge solver using Linearized ADMM.
Read this file before writing any code.

---

## Mathematical Context

### Problem

Solve the **dynamic Schrödinger Bridge** in 1D:

$$\min_{\rho, m} \int_0^1 \int_0^1 \frac{1}{2}\frac{m(t,x)^2}{\rho(t,x)} \, dx \, dt$$

subject to the **Fokker-Planck equation**:

$$\partial_t \rho + \partial_x m = \varepsilon \partial_{xx} \rho$$

with boundary conditions: $\rho(0,\cdot) = \rho_0$, $\rho(1,\cdot) = \rho_1$, Neumann in space.
When $\varepsilon = 0$ this reduces to **Benamou-Brenier** OT.

---

### Formulation 2: Two-grid Linearized ADMM

**Grid 1 (collocated):** $(\rho, m)$ at space-time cell centers
- $\rho_{ij}$: size $N_t \times N_x$
- $m_{ij}$: size $N_t \times N_x$

**Grid 2 (staggered):** $(\bar\rho, b)$ on staggered grid
- $\bar\rho_{ij}$: size $(N_t-1) \times N_x$ (interior time nodes, BCs absorbed into $d$)
- $b_{ij}$: size $N_t \times (N_x-1)$ (interior faces, Neumann BCs)

**ADMM constraint:** $A(\rho, m) - (\bar\rho, b) = 0$
- $A_\rho$: averages $\rho$ in time (collocated → staggered)
- $A_m$: averages $m$ in space (collocated → faces)

**Objective:** $J(\rho, m) = \Delta t \Delta x \sum_{i,j} \frac{1}{2} \frac{m_{ij}^2}{\rho_{ij}}$ ($+\infty$ if any $\rho_{ij} \leq 0$)

---

### Discrete Operators

For $N_t$ time intervals, $N_x$ space intervals, $\Delta t = 1/N_t$, $\Delta x = 1/N_x$.

**$D_t$** of size $N_t \times (N_t-1)$: first-difference in time, BCs in $d$.
**$A_t$** of size $N_t \times (N_t-1)$: time averaging.
**$D_x$** of size $N_x \times (N_x-1)$: first-difference in space.
**$L_x = D_x D_x^\top$**: Neumann Laplacian, size $N_x \times N_x$.

### Fokker-Planck Constraint

$$(I_x \otimes D_t - \varepsilon L_x \otimes A_t)\bar\rho + (D_x \otimes I_t) b = d$$

where $d$ encodes BCs: first block $-\rho_0/\Delta t$, last block $\rho_1/\Delta t$, zeros between.

---

### Linearized ADMM Iterations

With penalty $\gamma > 0$, proximal parameter $\tau > \gamma \|A\|^2$, dual $\delta$:

**Step 1 — $(\rho, m)$-update (pointwise proximal):**

$$(\rho^{k+1}, m^{k+1}) = \text{prox}_{J/\tau}\!\left((\rho^k, m^k) - \frac{\gamma}{\tau}A^\top\!\left(A(\rho^k,m^k) - (\bar\rho^k, b^k) + \frac{\delta^k}{\gamma}\right)\right)$$

Prox of kinetic energy: for each point $(u, v)$, find largest real root $\rho^* > 0$ of the cubic (Papadakis et al. 2014), then $m^* = \tau\rho^* v / (\tau\rho^* + \Delta t\Delta x/2)$. If $u \leq 0$: set $\rho^*=0, m^*=0$.

**Step 2 — $(\bar\rho, b)$-update (projection onto $C$):**

$$(\bar\rho^{k+1}, b^{k+1}) = \text{proj}_C\!\left(A(\rho^{k+1}, m^{k+1}) + \frac{\delta^k}{\gamma}\right)$$

Solve $M_{\text{SB}} \phi = F$ where:
$$M_{\text{SB}} = (I_x \otimes D_t - \varepsilon L_x \otimes A_t)(...)^\top + (D_x \otimes I_t)(D_x \otimes I_t)^\top$$

**Step 3 — Dual update:**

$$\delta^{k+1} = \delta^k + \gamma\left(A(\rho^{k+1}, m^{k+1}) - (\bar\rho^{k+1}, b^{k+1})\right)$$

---

### Solving $M_{\text{SB}}\phi = F$ Efficiently

Algorithm: DCT in space + DST-I + Woodbury rank-2 correction.

1. Reshape $F$ to $(N_t, N_x)$, apply DCT-II in space (axis=1)
2. For each spatial mode $k$:
   - Toeplitz eigenvalues: $\mu_l = \lambda^t_l + \varepsilon^2(\lambda^x_k)^2\lambda^A_l + \lambda^{xx}_k$
   - Apply $(T^k)^{-1}$ via DST-I + pointwise divide + inverse DST-I
   - Apply $2\times 2$ Woodbury correction
3. Apply inverse DCT-II in space

**Eigenvalues:**
- Spatial ($L_x$, DCT basis): $\lambda^x_k = \frac{2}{\Delta x^2}(\cos(\pi k/N_x)-1)$
- Temporal ($T_D$, DST-I): $\lambda^t_l = \frac{4}{\Delta t^2}\sin^2(\pi l/(2N_t))$
- Temporal ($T_A$, DST-I): $\lambda^A_l = \frac{1}{2}+\frac{1}{2}\cos(\pi l/N_t)$

**Woodbury corner values:**
$$\sigma^k_1 = -\frac{1}{\Delta t^2} - \frac{\varepsilon^2(\lambda^x_k)^2}{4} + \frac{\varepsilon\lambda^x_k}{\Delta t}, \quad \sigma^k_{N_t} = -\frac{1}{\Delta t^2} - \frac{\varepsilon^2(\lambda^x_k)^2}{4} - \frac{\varepsilon\lambda^x_k}{\Delta t}$$

---

## Implementation Guidelines

### Language & Libraries
- Python with `numpy`, `scipy`, `matplotlib`
- `scipy.fft.dst`, `scipy.fft.dct` with `norm='ortho'`
- Sparse operators via `scipy.sparse`
- **Use `python3`** to run scripts

### Code Structure (`sb_admm_1d.py`)
```
├── build_operators(Nt, Nx, dt, dx, eps)
├── build_rhs(rho0, rho1, Nt, Nx, dt, dx)
├── prox_J(u, v, dt, dx, tau)
├── solve_phi(F, Nt, Nx, dt, dx, eps, ...)
├── project_C(p_rho, p_b, ...)
├── interpolate_to_staggered(rho, m, ...)
├── admm_solve(rho0, rho1, Nt, Nx, eps, gamma, tau, max_iter, tol)
└── main()
```

### Key Rules
- **No Python loops over grid points** — use numpy vectorization
- Kronecker ordering: **time = slow index, space = fast index**
- Use **DST type 1** for temporal solve, **DCT type 2** for spatial solve
- Enforce $\rho > 0$: `rho = np.maximum(rho, 1e-10)` after prox
- $\varepsilon = 0$ must reproduce Benamou-Brenier OT exactly

### Parameters
- $N_t = N_x = 32$ for initial testing
- $\varepsilon \in \{0.0, 0.05, 0.1\}$
- $\gamma = 1.0$, $\tau = 2.0$ (tune if needed)
- Max iterations: 1000, tolerance: $10^{-5}$

### Initialization
```python
rho = np.ones((Nt, Nx)) * 0.5  # or linear interpolation
m = np.zeros((Nt, Nx))
rho_bar = np.zeros((Nt-1, Nx))
b = np.zeros((Nt, Nx-1))
delta_rho = np.zeros_like(rho_bar)
delta_b = np.zeros_like(b)
```

---

## Validation Checks (run before ADMM)

1. Print shapes of $D_t$, $A_t$, $D_x$, $L_x$
2. For small $N_t, N_x$: build $M_{\text{SB}}$ explicitly, verify eigenvalues > 0
3. Solve $M_{\text{SB}}\phi = F$ for random $F$, verify residual < $10^{-10}$
4. Verify prox satisfies optimality conditions
5. After convergence: verify Fokker-Planck residual < tol

---

## Test Cases

**Test 1 — Gaussian→Gaussian (OT, $\varepsilon=0$):**
$\rho_0 = \mathcal{N}(0.25, 0.05^2)$, $\rho_1 = \mathcal{N}(0.75, 0.05^2)$. Expected: linear interpolation.

**Test 2 — Schrödinger Bridge ($\varepsilon > 0$):**
Same marginals, $\varepsilon = 0.05, 0.1$. Expected: smoother, more diffuse than OT.

**Test 3 — Two-hump → one-hump:**
$\rho_0 = \frac{1}{2}\mathcal{N}(0.25,0.05^2)+\frac{1}{2}\mathcal{N}(0.75,0.05^2)$, $\rho_1 = \mathcal{N}(0.5,0.1^2)$

---

## Plots (save as PNG)

1. Heatmap of $\rho(t,x)$
2. Heatmap of $m(t,x)$
3. Marginals: $\rho(0,\cdot)$ and $\rho(1,\cdot)$ vs $\rho_0$, $\rho_1$
4. Convergence: primal + dual residuals vs iteration
5. Snapshots: $\rho(t,\cdot)$ at $t=0, 0.25, 0.5, 0.75, 1.0$

---

## Convergence Diagnostics (print every 50 iters)

1. Primal residual: $\|A(\rho^k, m^k) - (\bar\rho^k, b^k)\|$
2. Dual residual: $\|\gamma A^\top((\bar\rho^k, b^k) - (\bar\rho^{k-1}, b^{k-1}))\|$
3. Objective value: $J(\rho^k, m^k)$
4. Constraint violation: $\|(I_x \otimes D_t - \varepsilon L_x \otimes A_t)\bar\rho^k + (D_x \otimes I_t)b^k - d\|$

---

## Common Pitfalls

1. Kronecker ordering: keep time as slow index throughout
2. DST-I vs DST-II: use DST-I for temporal Toeplitz solve
3. DCT type: use DCT-II for spatial solve (Neumann BCs)
4. Use `norm='ortho'` for clean inverses
5. Proximal cubic: most delicate step — take care with positivity
6. Woodbury $2\times 2$ system: verify corner assembly carefully
7. Test $M_{\text{SB}}$ solver independently before plugging into ADMM

---

## Existing Files

- `sbp.py` — original 1D SB demo, equal std devs $\mathcal{N}(-1.5,0.5^2)\to\mathcal{N}(1.5,0.5^2)$
- `sbp_admm_1d.py` — ADMM-based 1D solver
- `sbp_admm.py` — ADMM solver (2D or general)
- `sbp_gaussian_comparison.py` — OT vs SB comparison with full analytics
- `context.md`, `skill.md`, `prompt.md` — source docs for this CLAUDE.md
