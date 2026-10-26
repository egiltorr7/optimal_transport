"""Zel'dovich-approximation realizations for the MAK / cosmological reconstruction
problem (candidate_3d_problems.tex Sec. 3.1), built so that the box has no-flux
walls: the displacement potential is synthesized in the cosine (Neumann) basis,
so it is even about every wall and Psi = -grad Phi has zero normal component ON
the wall.  Before shell crossing (I + grad Psi positive definite everywhere) no
particle crosses a wall and the no-flux problem is exact.  After shell crossing
particles near a wall can cross it; they are mirrored back by cic_deposit, which
is exactly the density of the mirror-symmetric flow restricted to the box (the
mirror particle enters where the particle leaves).  Plain numpy / scipy.

Units: lengths in Mpc/h, wavenumbers in h/Mpc.  rho_0 == 1 on the Lagrangian
grid q (cell centres); rho_1 is the CIC deposit of the displaced particles
x = q + D * Psi(q), normalized to mean 1.  D is the linear growth factor
(D = 1 today).  Zel'dovich trajectories are straight lines in (q, D), which is
why D in [0, 1] is the "time" of the dynamic OT problem.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import idctn


# --- linear power spectrum: BBKS transfer function, sigma_8 normalized --------
def bbks_transfer(k, gamma=0.21):
    q = k / gamma + 1e-30
    return np.log(1 + 2.34 * q) / (2.34 * q) * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25


def linear_power(k, sigma8=0.8, ns=0.96, gamma=0.21):
    """P(k) in (Mpc/h)^3, normalized so that the top-hat rms in R = 8 Mpc/h is sigma8."""
    kk = np.logspace(-4, 2, 4000)
    p = kk**ns * bbks_transfer(kk, gamma) ** 2
    x = kk * 8.0
    w = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    s2 = np.trapz(kk**2 * p * w**2, kk) / (2 * np.pi**2)
    A = sigma8**2 / s2
    return A * np.asarray(k) ** ns * bbks_transfer(np.asarray(k), gamma) ** 2


# --- Neumann (cosine-basis) Gaussian random field -------------------------------
def neumann_field(N, L, spectrum, rng, D=3):
    """Gaussian random field on an N^D cell-centred grid in [0,L]^D whose DCT-II
    coefficients have variance N^D P(k)/L^D with k = pi |m| / L.  In the interior
    its statistics are those of a stationary field with power spectrum P; it is
    even-symmetric about every wall.  Returns (field, k_grid)."""
    m = [np.arange(N)] * D
    K = np.sqrt(sum(g**2 for g in np.meshgrid(*m, indexing="ij"))) * np.pi / L
    coeff = rng.standard_normal(K.shape) * np.sqrt(N**D * spectrum(K) / L**D)
    coeff[(0,) * D] = 0.0
    return idctn(coeff, type=2, norm="ortho"), K


def zeldovich_displacement(N, L, rng, sigma8=0.8, D=3):
    """Linear overdensity delta and displacement Psi = -grad Phi, grad^2 Phi = delta,
    on the N^D Lagrangian grid, from a Neumann-basis realization.  Psi has zero
    normal component at the walls.  Shapes: delta (N,)*D, Psi (N,)*D + (D,)."""
    P = lambda k: linear_power(k, sigma8=sigma8)
    # synthesize Phi directly: P_Phi = P / k^4
    def spec_phi(k):
        out = np.zeros_like(k)
        nz = k > 0
        out[nz] = P(k[nz]) / k[nz] ** 4
        return out
    phi, K = neumann_field(N, L, spec_phi, rng, D)
    h = L / N
    # delta = grad^2 Phi, Psi = -grad Phi, both with mirror padding (Neumann symmetry)
    pad = np.pad(phi, 1, mode="symmetric")
    sl = [slice(1, -1)] * D
    Psi = np.empty(phi.shape + (D,))
    delta = np.zeros_like(phi)
    for d in range(D):
        up = list(sl); dn = list(sl)
        up[d] = slice(2, None); dn[d] = slice(0, -2)
        Psi[..., d] = -(pad[tuple(up)] - pad[tuple(dn)]) / (2 * h)
        delta += (pad[tuple(up)] - 2 * phi + pad[tuple(dn)]) / h**2
    return delta, Psi


def cic_deposit(x, L, n, D=3):
    """Cloud-in-cell mass assignment of particles x (M, D) in [0, L]^D onto an n^D
    cell-centred grid, with reflecting walls.  Returns density normalized to mean 1."""
    x = np.asarray(x, dtype=float)
    # reflect anything that crossed a wall (only possible through discretization error)
    x = np.where(x < 0, -x, x)
    x = np.where(x > L, 2 * L - x, x)
    h = L / n
    u = x / h - 0.5                       # cell-centre coordinates
    i0 = np.floor(u).astype(int)
    f = u - i0
    rho = np.zeros((n,) * D)
    for corner in range(2**D):
        idx = []
        w = np.ones(len(x))
        for d in range(D):
            bit = (corner >> d) & 1
            ii = i0[:, d] + bit
            w = w * (f[:, d] if bit else 1 - f[:, d])
            ii = np.where(ii < 0, -ii - 1, ii)          # reflect at the walls
            ii = np.where(ii > n - 1, 2 * n - 1 - ii, ii)
            idx.append(ii)
        np.add.at(rho, tuple(idx), w)
    return rho / rho.mean()


def jacobian_min_eig(Psi, h):
    """Smallest eigenvalue of I + grad Psi on the grid (<= 0 means shell crossing)."""
    D = Psi.shape[-1]
    J = np.empty(Psi.shape[:-1] + (D, D))
    for a in range(D):
        for b in range(D):
            J[..., a, b] = np.gradient(Psi[..., a], h, axis=b) + (1.0 if a == b else 0.0)
    return np.linalg.eigvalsh(0.5 * (J + np.swapaxes(J, -1, -2))).min(axis=-1)


def mak_problem(N=128, L=512.0, n=96, D_growth=1.0, sigma8=0.8, seed=1, D=3):
    """The MAK problem on a grid: rho_0 == 1 on the n^D grid, rho_1 = CIC deposit of
    N^D Zel'dovich particles displaced by D_growth * Psi.  Returns a dict."""
    rng = np.random.default_rng(seed)
    delta, Psi = zeldovich_displacement(N, L, rng, sigma8=sigma8, D=D)
    hq = L / N
    q = np.stack(np.meshgrid(*[(np.arange(N) + 0.5) * hq] * D, indexing="ij"), -1)
    x = (q + D_growth * Psi).reshape(-1, D)
    rho1 = cic_deposit(x, L, n, D)
    return dict(L=L, N=N, n=n, D_growth=D_growth, q=q, Psi=D_growth * Psi, delta=D_growth * delta,
                rho0=np.ones((n,) * D), rho1=rho1,
                lam_min=jacobian_min_eig(D_growth * Psi, hq))
