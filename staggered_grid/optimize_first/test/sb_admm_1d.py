"""
sb_admm_1d.py — 1D Dynamic Schrödinger Bridge via Linearized ADMM.

Solves:
    min_{rho, m}  dt*dx * sum_ij  m_ij^2 / (2*rho_ij)
    s.t. Fokker-Planck:  d_t rho + d_x m = eps * d_xx rho
         rho(0,.) = rho0,  rho(1,.) = rho1,  Neumann in x

Uses the two-grid linearized ADMM from Benamou, Brenier, Papadakis et al.
"""

import numpy as np
from scipy.fft import dct, idct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from operators import (
    build_all_operators, build_rhs,
    apply_Arho, apply_Arho_T, apply_Am, apply_Am_T,
)


# ---------------------------------------------------------------------------
# Proximal operator for kinetic energy  J(rho, m) = dt*dx * sum m^2/(2*rho)
# ---------------------------------------------------------------------------

def _cubic_largest_positive_root(a3, a2, a1, a0):
    """
    Vectorized: find the largest real root > 0 of
        a3*x^3 + a2*x^2 + a1*x + a0 = 0.

    Arrays a3, a2, a1, a0 broadcast to the same shape.
    Returns an array of the same shape.
    """
    a3, a2, a1, a0 = np.broadcast_arrays(
        np.asarray(a3, float), np.asarray(a2, float),
        np.asarray(a1, float), np.asarray(a0, float))
    shape = a3.shape

    # Monic form: x^3 + B*x^2 + C*x + D = 0
    B = a2 / a3
    C = a1 / a3
    D = a0 / a3

    # Depress via x = t - B/3
    p = C - B**2 / 3.0
    q = D - B*C / 3.0 + 2.0*B**3 / 27.0

    disc = -(4.0*p**3 + 27.0*q**2)   # > 0  ↔  three distinct real roots

    root = np.empty(shape)

    # --- case: three real roots (discriminant ≥ 0) -------------------------
    mask3 = disc >= 0.0
    if mask3.any():
        pm, qm, Bm = p[mask3], q[mask3], B[mask3]

        # safe amplitude;  p ≤ 0 when disc ≥ 0
        amp = 2.0 * np.sqrt(np.maximum(-pm / 3.0, 0.0))

        # argument for arccos; guard against division by zero (p≈0)
        arg = np.where(amp > 1e-14, 3.0*qm / (pm * amp), 0.0)
        theta = np.arccos(np.clip(arg, -1.0, 1.0)) / 3.0

        r0 = amp * np.cos(theta)           - Bm / 3.0
        r1 = amp * np.cos(theta + 2*np.pi/3) - Bm / 3.0
        r2 = amp * np.cos(theta + 4*np.pi/3) - Bm / 3.0

        root[mask3] = np.maximum(np.maximum(r0, r1), r2)

    # --- case: one real root (discriminant < 0) ----------------------------
    mask1 = ~mask3
    if mask1.any():
        pm, qm, Bm = p[mask1], q[mask1], B[mask1]

        Delta = (qm / 2.0)**2 + (pm / 3.0)**3   # Δ > 0 here
        sqrt_D = np.sqrt(np.maximum(Delta, 0.0))

        t1 = -qm / 2.0 + sqrt_D
        t2 = -qm / 2.0 - sqrt_D

        # Real cube roots (sign-preserving)
        cbrt1 = np.sign(t1) * np.abs(t1)**(1.0/3.0)
        cbrt2 = np.sign(t2) * np.abs(t2)**(1.0/3.0)

        root[mask1] = cbrt1 + cbrt2 - Bm / 3.0

    return root


def prox_J(u, v, dt, dx, tau):
    """
    Proximal operator of J/tau (kinetic energy), applied pointwise.

    J(rho, m) = dt*dx * sum_ij  m_ij^2 / (2*rho_ij)

    For each cell:
      (rho*, m*) = argmin_{rho>0, m}  (dt*dx)/(2*tau) * m^2/rho
                                    + (1/2)(rho-u)^2 + (1/2)(m-v)^2

    rho* is the largest positive root of the cubic:
        (rho - u)(tau*rho + w)^2 = (w*tau/2)*v^2,   w = dt*dx
    then  m* = tau*rho* * v / (tau*rho* + w)

    If u ≤ 0:  rho* = 0, m* = 0  (outside the domain of J).

    Parameters
    ----------
    u, v : ndarray, shape (Nt, Nx)  — input rho, m
    dt, dx, tau : float

    Returns
    -------
    rho_star, m_star : ndarray, shape (Nt, Nx)
    """
    w = dt * dx   # cell area

    # Coefficients: a3*rho^3 + a2*rho^2 + a1*rho + a0 = 0
    #   a3 = tau^2
    #   a2 = tau*(2*w - u*tau)
    #   a1 = w*(w - 2*u*tau)
    #   a0 = -u*w^2 - (w*tau/2)*v^2
    a3 = tau**2 * np.ones_like(u)
    a2 = tau * (2.0*w - u*tau)
    a1 = w * (w - 2.0*u*tau)
    a0 = -u*w**2 - 0.5*w*tau*v**2

    rho_star = _cubic_largest_positive_root(a3, a2, a1, a0)
    rho_star = np.maximum(rho_star, 0.0)

    # Where u ≤ 0: rho* = 0, m* = 0
    bad = u <= 0.0
    rho_star[bad] = 0.0

    # m* from optimality in m
    m_star = tau * rho_star * v / (tau * rho_star + w + 1e-300)
    m_star[bad] = 0.0

    # Safety floor on rho
    rho_star = np.maximum(rho_star, 1e-10)

    return rho_star, m_star


# ---------------------------------------------------------------------------
# Solver for  M_SB * phi = F   via DCT-II (space) + tridiagonal (time)
# ---------------------------------------------------------------------------

def solve_phi(F, Nt, Nx, dt, dx, eps):
    """
    Solve  M_SB * phi = F  via DCT-II diagonalisation in space.

    M_SB = FP_rhobar @ FP_rhobar.T + FP_b @ FP_b.T   (Nt*Nx square)

    DCT-II in space decouples the (Nt*Nx) system into Nx independent
    Nt × Nt tridiagonal systems, one per spatial DCT-II mode k.

    For each mode k:
        T^k = (D_t - eps*lk*A_t)(D_t - eps*lk*A_t)^T + lk*I_Nt
    where lk = (2/dx^2)*(1 - cos(pi*k/Nx)) >= 0 is the kth eigenvalue of L_x.

    T^k is symmetric tridiagonal with:
        interior diagonal:  alpha = 2/dt^2 + eps^2*lk^2/2 + lk
        off-diagonal:       beta  = -1/dt^2 + eps^2*lk^2/4
        corner (0,0):       1/dt^2 - eps*lk/dt + eps^2*lk^2/4 + lk
        corner (Nt-1,Nt-1): 1/dt^2 + eps*lk/dt + eps^2*lk^2/4 + lk

    T^0 (k=0) is the Neumann Laplacian D_t D_t^T — singular (null = const).
    Its RHS is always consistent; we return the minimum-norm solution.

    Parameters
    ----------
    F   : ndarray, shape (Nt, Nx)
    Nt, Nx, dt, dx, eps : problem parameters

    Returns
    -------
    phi : ndarray, shape (Nt, Nx)
    """
    from scipy.linalg import solve_banded

    # ---- 1. DCT-II in space -----------------------------------------------
    F_hat = dct(F, type=2, axis=1, norm='ortho')   # shape (Nt, Nx)
    phi_hat = np.empty_like(F_hat)

    # Spatial eigenvalues of L_x = D_x D_x^T (positive):
    k_arr = np.arange(Nx, dtype=float)
    lam_Lx = (2.0 / dx**2) * (1.0 - np.cos(np.pi * k_arr / Nx))  # >= 0

    # ---- 2. Solve each tridiagonal system T^k phi_k = F_hat[:,k] ----------
    for k in range(Nx):
        lk = lam_Lx[k]
        f_col = F_hat[:, k]

        # Tridiagonal entries
        alpha = 2.0/dt**2 + 0.5*eps**2*lk**2 + lk    # interior diagonal
        beta  = -1.0/dt**2 + 0.25*eps**2*lk**2        # off-diagonal

        main_diag = np.full(Nt, alpha)
        main_diag[0]  = 1.0/dt**2 - eps*lk/dt + 0.25*eps**2*lk**2 + lk
        main_diag[-1] = 1.0/dt**2 + eps*lk/dt + 0.25*eps**2*lk**2 + lk
        off_diag = np.full(Nt - 1, beta)

        if lk < 1e-14:
            # k=0: T^0 = D_t D_t^T is singular (null space = constants).
            # Project F_hat[:,0] onto range and return minimum-norm solution.
            T0 = (np.diag(main_diag)
                  + np.diag(off_diag, 1)
                  + np.diag(off_diag, -1))
            # Minimum-norm solution via lstsq
            sol, _, _, _ = np.linalg.lstsq(T0, f_col, rcond=None)
            # Subtract constant so solution is orthogonal to null space (1)
            sol -= sol.mean()
            phi_hat[:, k] = sol
        else:
            # T^k is symmetric positive definite: use banded solver O(Nt)
            # solve_banded format: ab[0, 1:] = upper, ab[1, :] = main, ab[2, :-1] = lower
            ab = np.zeros((3, Nt))
            ab[0, 1:]  = off_diag   # upper diagonal
            ab[1, :]   = main_diag  # main diagonal
            ab[2, :-1] = off_diag   # lower diagonal
            phi_hat[:, k] = solve_banded((1, 1), ab, f_col)

    # ---- 3. Inverse DCT-II in space ----------------------------------------
    phi = idct(phi_hat, type=2, axis=1, norm='ortho')
    return phi


# ---------------------------------------------------------------------------
# Projection onto the Fokker-Planck constraint set C
# ---------------------------------------------------------------------------

def project_C(p_rho, p_b, ops, d, Nt, Nx, dt, dx, eps):
    """
    Project (p_rho, p_b) onto C = {(rho_bar, b) : FP constraint holds}.

    The constraint is  FP_rhobar * vec(rho_bar) + FP_b * vec(b) = d.
    Projection solves the KKT system:
        M_SB * phi = FP_rhobar * vec(p_rho) + FP_b * vec(p_b) - d
    then
        rho_bar = p_rho - FP_rhobar^T * phi  (reshaped)
        b       = p_b   - FP_b^T     * phi  (reshaped)

    Parameters
    ----------
    p_rho : ndarray, shape (Nt-1, Nx)
    p_b   : ndarray, shape (Nt,   Nx-1)
    ops   : dict from build_all_operators
    d     : ndarray, shape (Nt*Nx,)
    Nt, Nx, dt, dx, eps : problem parameters

    Returns
    -------
    rho_bar : ndarray, shape (Nt-1, Nx)
    b       : ndarray, shape (Nt,   Nx-1)
    """
    FP_rhobar = ops['FP_rhobar']   # (Nt*Nx, (Nt-1)*Nx)
    FP_b      = ops['FP_b']        # (Nt*Nx,  Nt*(Nx-1))

    vec_p_rho = p_rho.ravel()
    vec_p_b   = p_b.ravel()

    # RHS for phi: residual of constraint at (p_rho, p_b)
    rhs = FP_rhobar @ vec_p_rho + FP_b @ vec_p_b - d   # shape (Nt*Nx,)

    # Reshape to (Nt, Nx) for the DCT/DST solver
    F = rhs.reshape(Nt, Nx)
    phi_arr = solve_phi(F, Nt, Nx, dt, dx, eps)          # shape (Nt, Nx)
    phi_vec = phi_arr.ravel()                              # shape (Nt*Nx,)

    # Update
    rho_bar = p_rho - (FP_rhobar.T @ phi_vec).reshape(Nt-1, Nx)
    b       = p_b   - (FP_b.T      @ phi_vec).reshape(Nt,   Nx-1)
    return rho_bar, b


# ---------------------------------------------------------------------------
# Interpolation: collocated (rho, m) -> staggered (rho_bar, b)
# ---------------------------------------------------------------------------

def interpolate_to_staggered(rho, m):
    """
    Compute the ADMM constraint variable  A(rho, m) = (rho_bar, b).

    rho_bar = A_rho * rho  (time averaging, shape Nt-1 × Nx)
    b       = A_m   * m    (space averaging, shape Nt × Nx-1)
    """
    return apply_Arho(rho), apply_Am(m)


# ---------------------------------------------------------------------------
# Full linearized ADMM solver
# ---------------------------------------------------------------------------

def admm_solve(rho0, rho1, Nt, Nx, eps, gamma=1.0, tau=5.0,
               max_iter=2000, tol=1e-4):
    """
    Solve the 1D Schrödinger Bridge via linearized ADMM.

    Parameters
    ----------
    rho0, rho1 : ndarray, shape (Nx,)  — boundary marginals (normalised)
    Nt, Nx     : int — number of time / space cells
    eps        : float — diffusion coefficient (0 → Benamou-Brenier OT)
    gamma      : float — ADMM penalty
    tau        : float — proximal parameter (must satisfy tau > gamma*||A||^2)
    max_iter   : int
    tol        : float — convergence tolerance on primal residual ||A(rho,m)-(rho_bar,b)||.
                         The dual residual converges at O(1/k) and is printed but not
                         used as stopping criterion.

    Returns
    -------
    rho  : ndarray, shape (Nt, Nx)
    m    : ndarray, shape (Nt, Nx)
    info : dict with 'primal_res', 'dual_res', 'obj', 'fp_viol' histories
    """
    dt = 1.0 / Nt
    dx = 1.0 / Nx

    # Build operators
    ops = build_all_operators(Nt, Nx, dt, dx, eps)
    d   = build_rhs(rho0, rho1, Nt, Nx, dt)

    # Initialise primal variables: displacement interpolation.
    # Linear interpolation of two separated distributions gives a BIMODAL
    # density at interior times — a terrible starting point for ADMM.
    # Instead we shift rho0 along the interpolated centre-of-mass trajectory
    # so each slice is unimodal and close to the optimal OT/SB path.
    t_arr = np.linspace(0.0, 1.0, Nt)
    x_arr = np.linspace(dx / 2, 1.0 - dx / 2, Nx)
    mu0   = float(np.sum(x_arr * rho0) * dx)   # centre of mass of rho0
    mu1   = float(np.sum(x_arr * rho1) * dx)   # centre of mass of rho1
    rho   = np.empty((Nt, Nx))
    for i, t in enumerate(t_arr):
        shift = t * (mu1 - mu0)                # x-shift for this time slice
        rho[i] = np.interp(x_arr - shift, x_arr, rho0, left=0.0, right=0.0)
        rho[i] = np.maximum(rho[i], 1e-10)
        rho[i] /= rho[i].sum() * dx            # normalise to unit mass
    # Initial momentum: constant-velocity advection  m = v*rho,  v = mu1-mu0
    m = (mu1 - mu0) * rho
    rho_bar   = apply_Arho(rho)            # (Nt-1, Nx)
    b         = apply_Am(m)                # (Nt,   Nx-1)
    delta_rho = np.zeros_like(rho_bar)
    delta_b   = np.zeros_like(b)

    # History
    primal_hist, dual_hist, obj_hist, fp_hist = [], [], [], []

    FP_rhobar = ops['FP_rhobar']
    FP_b      = ops['FP_b']

    for it in range(max_iter):

        # ---- Step 1: (rho, m) proximal update ----------------------------
        # Gradient of the augmented-Lagrangian w.r.t. (rho, m):
        #   grad = A^T( A(rho,m) - (rho_bar, b) + delta/gamma )
        rho_bar_A, b_A = interpolate_to_staggered(rho, m)

        res_rho = rho_bar_A - rho_bar + delta_rho / gamma
        res_b   = b_A       - b       + delta_b   / gamma

        # A^T applied to residuals
        AT_res_rho = apply_Arho_T(res_rho, Nt)   # (Nt, Nx)
        AT_res_b   = apply_Am_T(res_b, Nx)        # (Nt, Nx)

        # Gradient descent step before proximal
        u = rho - (gamma / tau) * AT_res_rho
        v = m   - (gamma / tau) * AT_res_b

        rho_new, m_new = prox_J(u, v, dt, dx, tau)

        # ---- Step 2: (rho_bar, b) projection update ----------------------
        rho_bar_prev = rho_bar.copy()
        b_prev       = b.copy()

        rho_bar_A2, b_A2 = interpolate_to_staggered(rho_new, m_new)
        p_rho = rho_bar_A2 + delta_rho / gamma
        p_b   = b_A2       + delta_b   / gamma

        rho_bar, b = project_C(p_rho, p_b, ops, d, Nt, Nx, dt, dx, eps)

        # ---- Step 3: dual update -----------------------------------------
        rho_bar_A3, b_A3 = interpolate_to_staggered(rho_new, m_new)
        delta_rho = delta_rho + gamma * (rho_bar_A3 - rho_bar)
        delta_b   = delta_b   + gamma * (b_A3       - b)

        # ---- Convergence diagnostics ------------------------------------
        # Normalize residuals by sqrt(N_elements) for grid-size independence.
        N_primal = (Nt - 1) * Nx + Nt * (Nx - 1)
        N_dual   = 2 * Nt * Nx
        primal_rho = np.linalg.norm(rho_bar_A3 - rho_bar)
        primal_b   = np.linalg.norm(b_A3 - b)
        primal_res = np.sqrt(primal_rho**2 + primal_b**2) / np.sqrt(N_primal)

        dual_rho   = gamma * np.linalg.norm(apply_Arho_T(rho_bar - rho_bar_prev, Nt))
        dual_b_    = gamma * np.linalg.norm(apply_Am_T(b - b_prev, Nx))
        dual_res   = np.sqrt(dual_rho**2 + dual_b_**2) / np.sqrt(N_dual)

        # Objective (kinetic energy at current rho_new, m_new)
        with np.errstate(divide='ignore', invalid='ignore'):
            kin = np.where(rho_new > 0,
                           m_new**2 / (2.0 * rho_new), 0.0)
        obj = dt * dx * np.sum(kin)

        # Fokker-Planck violation
        fp_viol = np.linalg.norm(
            FP_rhobar @ rho_bar.ravel() + FP_b @ b.ravel() - d)

        primal_hist.append(primal_res)
        dual_hist.append(dual_res)
        obj_hist.append(obj)
        fp_hist.append(fp_viol)

        rho, m = rho_new, m_new

        if (it + 1) % 50 == 0:
            print(f"  iter {it+1:4d}:  primal={primal_res:.3e}  "
                  f"dual={dual_res:.3e}  obj={obj:.6f}  "
                  f"FP_viol={fp_viol:.3e}")

        if primal_res < tol:
            print(f"  Converged at iteration {it+1} "
                  f"(primal={primal_res:.2e}, dual={dual_res:.2e}).")
            break

    info = dict(
        primal_res=np.array(primal_hist),
        dual_res=np.array(dual_hist),
        obj=np.array(obj_hist),
        fp_viol=np.array(fp_hist),
    )
    return rho, m, info


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_plots(rho, m, info, rho0, rho1, Nt, Nx, eps, prefix='sb'):
    """Save 5 diagnostic plots as PNG files."""
    dt = 1.0 / Nt
    dx = 1.0 / Nx
    t_arr = np.linspace(dt/2, 1.0 - dt/2, Nt)
    x_arr = np.linspace(dx/2, 1.0 - dx/2, Nx)

    # 1. Heatmap of rho
    fig, ax = plt.subplots()
    im = ax.imshow(rho.T, origin='lower', aspect='auto',
                   extent=[0, 1, 0, 1], cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_title(f'rho(t,x), eps={eps}')
    fig.savefig(f'{prefix}_rho.png', dpi=100)
    plt.close(fig)

    # 2. Heatmap of m
    fig, ax = plt.subplots()
    im = ax.imshow(m.T, origin='lower', aspect='auto',
                   extent=[0, 1, 0, 1], cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_title(f'm(t,x), eps={eps}')
    fig.savefig(f'{prefix}_m.png', dpi=100)
    plt.close(fig)

    # 3. Marginals
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x_arr, rho[0], label='rho(0,x)')
    axes[0].plot(x_arr, rho0,   '--', label='rho0 (target)')
    axes[0].legend(); axes[0].set_title('Initial marginal')
    axes[1].plot(x_arr, rho[-1], label='rho(1,x)')
    axes[1].plot(x_arr, rho1,    '--', label='rho1 (target)')
    axes[1].legend(); axes[1].set_title('Final marginal')
    fig.savefig(f'{prefix}_marginals.png', dpi=100)
    plt.close(fig)

    # 4. Convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(info['primal_res'], label='primal')
    axes[0].semilogy(info['dual_res'],   label='dual')
    axes[0].legend(); axes[0].set_xlabel('iteration')
    axes[0].set_title('Residuals')
    axes[1].plot(info['obj'])
    axes[1].set_xlabel('iteration')
    axes[1].set_title('Objective J(rho, m)')
    fig.savefig(f'{prefix}_convergence.png', dpi=100)
    plt.close(fig)

    # 5. Snapshots of rho at t=0, 0.25, 0.5, 0.75, 1.0
    snap_t = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots()
    for s in snap_t:
        idx = int(round(s * (Nt - 1)))
        idx = np.clip(idx, 0, Nt - 1)
        ax.plot(x_arr, rho[idx], label=f't={s}')
    ax.legend(); ax.set_xlabel('x')
    ax.set_title(f'Snapshots rho(t,x), eps={eps}')
    fig.savefig(f'{prefix}_snapshots.png', dpi=100)
    plt.close(fig)

    print(f"  Plots saved with prefix '{prefix}'.")


# ---------------------------------------------------------------------------
# Main — test cases
# ---------------------------------------------------------------------------

def gaussian(x, mu, sigma):
    g = np.exp(-0.5 * ((x - mu) / sigma)**2)
    return g / (g.sum() * (x[1] - x[0]))   # normalised


def main():
    # sigma=0.05 needs ~6 cells/sigma → Nx=64 gives dx=1/64, sigma/dx≈3.2
    # For very accurate OT use Nx=128; Nx=64 is a practical compromise.
    Nt, Nx = 64, 64
    x = np.linspace(1.0/(2*Nx), 1.0 - 1.0/(2*Nx), Nx)

    # ---- Test 1: Gaussian→Gaussian, eps=0 (Benamou-Brenier OT) -----------
    print("=" * 60)
    print("Test 1: Gaussian→Gaussian, eps=0 (Benamou-Brenier OT)")
    print("  Analytical J* = 0.5*(0.75-0.25)^2 = 0.125  [exact for uniform shift]")
    rho0 = gaussian(x, 0.25, 0.05)
    rho1 = gaussian(x, 0.75, 0.05)
    rho, m, info = admm_solve(rho0, rho1, Nt, Nx, eps=0.0,
                              gamma=1.0, tau=5.0, max_iter=3000, tol=1e-4)
    print(f"  Final objective: {info['obj'][-1]:.4f}  (expect → 0.125)")
    save_plots(rho, m, info, rho0, rho1, Nt, Nx, eps=0.0,
               prefix='test1_OT')

    # ---- Test 2: Same marginals, eps=0.05 (Schrödinger Bridge) -----------
    print("=" * 60)
    print("Test 2: Gaussian→Gaussian, eps=0.05 (Schrödinger Bridge)")
    print("  Expect: smoother, wider paths than OT; J_SB > J_OT")
    rho, m, info = admm_solve(rho0, rho1, Nt, Nx, eps=0.05,
                              gamma=1.0, tau=5.0, max_iter=3000, tol=1e-4)
    print(f"  Final objective: {info['obj'][-1]:.4f}")
    save_plots(rho, m, info, rho0, rho1, Nt, Nx, eps=0.05,
               prefix='test2_SB05')

    # ---- Test 3: Two-hump → one-hump, eps=0.05 ---------------------------
    print("=" * 60)
    print("Test 3: Two-hump→one-hump, eps=0.05")
    rho0_2 = 0.5*gaussian(x, 0.25, 0.05) + 0.5*gaussian(x, 0.75, 0.05)
    rho1_2 = gaussian(x, 0.5, 0.1)
    rho, m, info = admm_solve(rho0_2, rho1_2, Nt, Nx, eps=0.05,
                              gamma=1.0, tau=5.0, max_iter=3000, tol=1e-4)
    print(f"  Final objective: {info['obj'][-1]:.4f}")
    save_plots(rho, m, info, rho0_2, rho1_2, Nt, Nx, eps=0.05,
               prefix='test3_twohump')


if __name__ == '__main__':
    main()
