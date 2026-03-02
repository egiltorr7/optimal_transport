# Mathematical Context: ADMM for Schrödinger Bridge (1D)

## Problem

We want to solve the **dynamic Schrödinger Bridge** problem in 1D:

$$\min_{\rho, m} \int_0^1 \int_0^1 \frac{1}{2}\frac{m(t,x)^2}{\rho(t,x)} \, dx \, dt$$

subject to the **Fokker-Planck equation**:

$$\partial_t \rho + \partial_x m = \varepsilon \partial_{xx} \rho \quad \text{on } (0,1)^2$$

with boundary conditions:
- $\rho(0, \cdot) = \rho_0$ (given initial density)
- $\rho(1, \cdot) = \rho_1$ (given final density)
- Neumann BCs in space: $\partial_x \rho \cdot n = 0$ at $x=0,1$

When $\varepsilon = 0$ this reduces to the **Benamou-Brenier** optimal transport problem.

---

## Formulation 2: Two-grid Linearized ADMM

### Variable splitting

We use **two grids**:

**Grid 1 (collocated):** $(\rho, m)$ live at space-time cell centers — for easy prox of $J$:
- $\rho_{ij} \approx \rho(t_{i+\frac{1}{2}}, x_{j+\frac{1}{2}})$, size $N_t \times N_x$
- $m_{ij} \approx m(t_{i+\frac{1}{2}}, x_{j+\frac{1}{2}})$, size $N_t \times N_x$

**Grid 2 (staggered):** $(\bar\rho, b)$ live on staggered grid — for clean Fokker-Planck discretization:
- $\bar\rho_{ij} \approx \rho(t_i, x_{j+\frac{1}{2}})$, integer times × cell centers, size $(N_t-1) \times N_x$ (interior time nodes only, BCs absorbed into $d$)
- $b_{ij} \approx m(t_{i+\frac{1}{2}}, x_j)$, half times × cell faces, size $N_t \times (N_x-1)$ (interior faces only due to Neumann BCs)

### ADMM constraint

$$A\begin{pmatrix}\rho \\ m\end{pmatrix} - \begin{pmatrix}\bar\rho \\ b\end{pmatrix} = 0$$

where $A$ interpolates from collocated $\to$ staggered grid:
- $A_\rho$: averages $\rho$ in time (cell centers $\to$ integer times)
- $A_m$: averages $m$ in space (cell centers $\to$ cell faces)

### Objective

$$J(\rho, m) = \Delta t \Delta x \sum_{i,j} \frac{1}{2} \frac{m_{ij}^2}{\rho_{ij}}$$

(with convention $J = +\infty$ if any $\rho_{ij} \leq 0$)

---

## Discrete Operators

For $N_t$ time intervals and $N_x$ space intervals, $\Delta t = 1/N_t$, $\Delta x = 1/N_x$.

### $D_t$: time difference, size $N_t \times (N_t - 1)$

Acts on interior time nodes $n = 1, \ldots, N_t-1$ of $\bar\rho$. Boundary values $\bar\rho^0 = \rho_0$, $\bar\rho^{N_t} = \rho_1$ are absorbed into $d$.

For $N_t = 4$:
```
D_t = (1/dt) * [ 1  0  0 ]
                [-1  1  0 ]
                [ 0 -1  1 ]
                [ 0  0 -1 ]
```
First row: $+\bar\rho^1$ (since $\bar\rho^0 = \rho_0$ moved to RHS)
Last row: $-\bar\rho^{N_t-1}$ (since $\bar\rho^{N_t} = \rho_1$ moved to RHS)

### $A_t$: time averaging, size $N_t \times (N_t - 1)$

Averages interior $\bar\rho$ from integer times to half times. For $N_t = 4$:
```
A_t = (1/2) * [ 1  0  0 ]
               [ 1  1  0 ]
               [ 0  1  1 ]
               [ 0  0  1 ]
```

### $D_x$: space divergence, size $N_x \times (N_x - 1)$

Acts on interior faces of $b$. For $N_x = 4$:
```
D_x = (1/dx) * [ 1  0  0 ]
                [-1  1  0 ]
                [ 0 -1  1 ]
                [ 0  0 -1 ]
```

### $L_x = D_x D_x^\top$: discrete Laplacian, size $N_x \times N_x$

With Neumann BCs (corner entries $-1$ instead of $-2$):
```
L_x = (1/dx^2) * [-1  1  0  0]
                  [ 1 -2  1  0]
                  [ 0  1 -2  1]
                  [ 0  0  1 -1]
```

---

## Fokker-Planck Constraint

The discrete Fokker-Planck equation enforced at half times × cell centers:

$$(I_x \otimes D_t - \varepsilon L_x \otimes A_t)\bar\rho + (D_x \otimes I_t) b = d$$

where $d$ encodes the boundary conditions:

$$d = \frac{1}{\Delta t} \text{vec} \begin{pmatrix} -\rho_0 \\ 0 \\ \vdots \\ 0 \\ \rho_1 \end{pmatrix}$$

(first block $-\rho_0/\Delta t$, last block $\rho_1/\Delta t$, zeros in between)

The constraint set is:
$$C = \{(\bar\rho, b) : (I_x \otimes D_t - \varepsilon L_x \otimes A_t)\bar\rho + (D_x \otimes I_t)b = d\}$$

---

## Linearized ADMM Iterations

With penalty parameter $\gamma > 0$ and proximal parameter $\tau > \gamma \|A\|^2$, dual variable $\delta$ (on staggered grid):

### Step 1: $(\rho, m)$-update (pointwise proximal)

$$(\rho^{k+1}, m^{k+1}) = \text{prox}_{J/\tau}\left((\rho^k, m^k) - \frac{\gamma}{\tau}A^\top\left(A(\rho^k,m^k) - (\bar\rho^k, b^k) + \frac{\delta^k}{\gamma}\right)\right)$$

The argument is a known vector — let $u = \rho^k - \frac{\gamma}{\tau}[A_\rho^\top r_\rho]$ and $v = m^k - \frac{\gamma}{\tau}[A_m^\top r_m]$ where $r = A(\rho^k, m^k) - (\bar\rho^k, b^k) + \delta^k/\gamma$.

Then for each $(i,j)$, solve the **proximal of kinetic energy** (cubic root problem):

$$(\rho^{k+1}_{ij}, m^{k+1}_{ij}) = \arg\min_{\rho \geq 0, m} \frac{\Delta t \Delta x}{2}\frac{m^2}{\rho} + \frac{\tau}{2}(\rho - u_{ij})^2 + \frac{\tau}{2}(m - v_{ij})^2$$

**Closed form solution:** given $(u, v)$, solve for $\rho > 0$:

$$\rho^3 - u\rho^2 + \frac{\Delta t \Delta x}{2\tau} v^2 \cdot \frac{1}{???}$$

Actually the prox reduces to finding the largest real root of:

$$2\tau \rho^3 - 2\tau u \rho^2 + \Delta t \Delta x \cdot v^2 / 2 = 0$$

Wait — let me be more careful. Optimality in $m$ gives:

$$m = \frac{\tau \rho v}{\tau \rho + \Delta t \Delta x / 2}$$

Substituting back and optimizing over $\rho$ gives a **cubic equation** in $\rho$:

$$2\tau \rho^3 - 2\tau u \rho^2 - \frac{\Delta t \Delta x \cdot v^2}{2(\tau \rho + \Delta t \Delta x/2)} \cdot \tau = 0$$

In practice: find the **largest real root** of the cubic, then recover $m$ from the formula above. If $u \leq 0$, set $\rho = 0, m = 0$.

### Step 2: $(\bar\rho, b)$-update (projection onto $C$)

$$(\bar\rho^{k+1}, b^{k+1}) = \text{proj}_C\left(A(\rho^{k+1}, m^{k+1}) + \frac{\delta^k}{\gamma}\right)$$

Let $p_{\bar\rho} = A_\rho \rho^{k+1} + \delta^k_{\bar\rho}/\gamma$ and $p_b = A_m m^{k+1} + \delta^k_b/\gamma$.

KKT conditions give $\bar\rho = p_{\bar\rho} - (I_x \otimes D_t - \varepsilon L_x \otimes A_t)^\top \phi$ and $b = p_b - (D_x \otimes I_t)^\top \phi$, where $\phi$ solves:

$$M_{\text{SB}} \phi = d - (I_x \otimes D_t - \varepsilon L_x \otimes A_t) p_{\bar\rho} - (D_x \otimes I_t) p_b$$

with:

$$M_{\text{SB}} = (I_x \otimes D_t - \varepsilon L_x \otimes A_t)(I_x \otimes D_t - \varepsilon L_x \otimes A_t)^\top + (D_x \otimes I_t)(D_x \otimes I_t)^\top$$

$$= I_x \otimes D_t D_t^\top - \varepsilon L_x \otimes (A_t D_t^\top + D_t A_t^\top) + \varepsilon^2 L_x^2 \otimes A_t A_t^\top + D_x D_x^\top \otimes I_t$$

### Step 3: Dual update

$$\delta^{k+1} = \delta^k + \gamma\left(A(\rho^{k+1}, m^{k+1}) - (\bar\rho^{k+1}, b^{k+1})\right)$$

---

## Solving $M_{\text{SB}}\phi = F$ efficiently

### Structure of $M_{\text{SB}}$

Write:

$$M_{\text{SB}} = I_x \otimes T_D + \varepsilon^2 L_x^2 \otimes T_A + D_x D_x^\top \otimes I_t + \text{rank-2 correction in time}$$

where $T_D$, $T_A$ are the **Toeplitz parts** of $D_t D_t^\top$, $A_t A_t^\top$:

$$T_D = \frac{1}{\Delta t^2}\begin{pmatrix} 2 & -1 & & \\ -1 & 2 & -1 & \\ & \ddots & \ddots & \ddots \\ & & -1 & 2 \end{pmatrix}, \quad T_A = \frac{1}{4}\begin{pmatrix} 2 & 1 & & \\ 1 & 2 & 1 & \\ & \ddots & \ddots & \ddots \\ & & 1 & 2 \end{pmatrix}$$

The boundary corrections are **rank-2 diagonal** (nonzero only at corners $(1,1)$ and $(N_t, N_t)$):

$$E_D = D_t D_t^\top - T_D = -\frac{1}{\Delta t^2}(e_1 e_1^\top + e_{N_t} e_{N_t}^\top)$$
$$E_A = A_t A_t^\top - T_A = -\frac{1}{4}(e_1 e_1^\top + e_{N_t} e_{N_t}^\top)$$
$$\text{cross} = A_t D_t^\top + D_t A_t^\top = \frac{1}{\Delta t}(e_1 e_1^\top - e_{N_t} e_{N_t}^\top)$$

### Algorithm: DST in time + Woodbury rank-2 correction

For each spatial mode $k = 0, \ldots, N_x - 1$ (after applying DCT in space):

The system decouples into independent $N_t \times N_t$ systems:

$$M^k \hat\phi_k = \hat F_k$$

where $M^k = T^k + U \Sigma^k U^\top$ with:
- $T^k$: Toeplitz, diagonalized by **DST-I** with eigenvalues $\lambda^t_l + \varepsilon^2(\lambda^x_k)^2 \lambda^A_l + \lambda^{xx}_k$
- $U = [e_1, e_{N_t}] \in \mathbb{R}^{N_t \times 2}$
- $\Sigma^k = \text{diag}(\sigma^k_1, \sigma^k_{N_t})$ with corner values

Apply **Woodbury formula**:

$$(M^k)^{-1} = (T^k)^{-1} - (T^k)^{-1}U\left((\Sigma^k)^{-1} + U^\top(T^k)^{-1}U\right)^{-1}U^\top(T^k)^{-1}$$

Cost per mode: $O(N_t \log N_t)$ for DST + $O(1)$ for $2\times 2$ solve.
Total cost: $O(N_x N_t \log N_t)$ after DCT in space.

### Eigenvalues

Spatial eigenvalues of $L_x = D_x D_x^\top$ (DCT basis):
$$\lambda^x_k = \frac{2}{\Delta x^2}\left(\cos\left(\frac{\pi k}{N_x}\right) - 1\right), \quad k = 0,\ldots,N_x-1$$

Temporal eigenvalues of $T_D$ (DST-I basis):
$$\lambda^t_l = \frac{2}{\Delta t^2}\left(\cos\left(\frac{\pi l}{N_t}\right) - 1\right) + \frac{2}{\Delta t^2}, \quad l = 1,\ldots,N_t$$

Wait — DST-I eigenvalues for the tridiagonal $(2,-1,-1)$ matrix:
$$\lambda^t_l = \frac{4}{\Delta t^2}\sin^2\left(\frac{\pi l}{2 N_t}\right), \quad l = 1,\ldots,N_t$$

Temporal eigenvalues of $T_A$ (same DST-I basis):
$$\lambda^A_l = \frac{1}{2} + \frac{1}{2}\cos\left(\frac{\pi l}{N_t}\right), \quad l = 1,\ldots,N_t$$

---

## Parameters

- $N_t$: number of time intervals (e.g. 32, 64)
- $N_x$: number of space intervals (e.g. 32, 64)
- $\varepsilon$: diffusion coefficient (0 = pure OT, try 0.01, 0.1, 0.5)
- $\gamma$: ADMM penalty parameter (try 1.0)
- $\tau$: proximal parameter, must satisfy $\tau > \gamma \|A\|^2$ (try $\tau = 2\gamma$)
- Max iterations: 500-1000
- Convergence tolerance: $10^{-6}$ on primal/dual residuals

---

## Test Cases (1D)

### Test 1: Gaussian to Gaussian (OT, $\varepsilon = 0$)
$$\rho_0(x) = \mathcal{N}(0.25, 0.05^2), \quad \rho_1(x) = \mathcal{N}(0.75, 0.05^2)$$
Expected: straight-line transport, linear interpolation of Gaussians.

### Test 2: Schrödinger Bridge ($\varepsilon > 0$)
Same marginals as Test 1 but with $\varepsilon = 0.05, 0.1$.
Expected: smoother, more diffuse interpolation than OT.

### Test 3: Two-hump to one-hump
$$\rho_0(x) = \frac{1}{2}\mathcal{N}(0.25, 0.05^2) + \frac{1}{2}\mathcal{N}(0.75, 0.05^2), \quad \rho_1(x) = \mathcal{N}(0.5, 0.1^2)$$

---

## Convergence Diagnostics

Track at each iteration:
1. **Primal residual**: $\|A(\rho^k, m^k) - (\bar\rho^k, b^k)\|$
2. **Dual residual**: $\|\gamma A^\top((\bar\rho^k, b^k) - (\bar\rho^{k-1}, b^{k-1}))\|$
3. **Objective value**: $J(\rho^k, m^k)$
4. **Constraint violation**: $\|(I_x \otimes D_t - \varepsilon L_x \otimes A_t)\bar\rho^k + (D_x \otimes I_t)b^k - d\|$
