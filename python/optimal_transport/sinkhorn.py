"""Dynamic Schrodinger bridge via Hopf-Cole / Sinkhorn iteration, with a
Neumann (reflecting-wall) heat kernel.

Port of matlab/shared/1d/pipelines/sinkhorn_hopf_cole.m and
matlab/shared/1d/utils/precomp_heat_neumann.m.

Unlike analytical_sb_gaussian (which assumes free-space Brownian motion on
all of R, only accurate while the density's spread stays small relative to
the domain), this solves the exact discrete-marginal bridge problem on the
actual bounded domain -- the right reference once vareps is large enough
that the density can "feel" the domain boundary.

Hopf-Cole decomposition:  rho(t,x) = phi(t,x) * psi(t,x)
  d_t phi =  eps * Delta phi   (forward heat equation)
 -d_t psi =  eps * Delta psi   (backward heat equation)
with marginal conditions phi(0)*psi(0)=rho0, phi(1)*psi(1)=rho1.

Both marginals are tracked: SinkhornFit carries `error` (left -- the lagging
one, which normally decides) and `error_right`, and convergence now requires
BOTH below tol. See sinkhorn_fit step 5 for why the right one holds by
construction and why it is still worth measuring.

Sinkhorn loop applies the FULL-time heat kernel H_T in one shot per
iteration (not nt small steps): phi(0), psi(1) are narrow (each equals a
marginal divided by a slowly-varying factor), so a single application of
the exact semigroup to a narrow function is accurate even for large eps --
step-by-step propagation would compound truncation error instead.

Fitting (finding phi(0), psi(T)) is separated from evaluating the
trajectory at specific times: the heat semigroup is exact and spectral
(exp(-eps*lambda_x*t)), so phi/psi can be evaluated at *any* real t
directly -- no fixed time grid is needed. This lets a single fit (ideally
on a fine spatial grid) serve as a reference for comparison against
multiple, coarser LADMM grids: see study_utils.py, which fits once per
vareps on a fine grid and then brings rho to each test resolution.

That spatial reduction is NOT a free choice -- the solver reproduces its
marginals exactly at t=0 and t=1 (baked into A), so a reference discretizing
the same continuous marginal differently leaves an irreducible endpoint error
floor. Exact subsampling cannot be arranged either: cell-centred grids at nx
and 2nx share no point at all. study_utils.resample_rho_spectral therefore
sums THIS grid's own DCT-II series (its basis is cos(j*pi*x), so it evaluates
at any x, exactly as apply_time evaluates at any t) at the coarse cell
centres, matching the solver's marginals to ~1e-15. See
docs/sinkhorn_reference.tex for the derivation, the impossibility argument,
and the O(h^2/dt) failure that block-averaging used to cause.

The momentum returned by sinkhorn_eval is m = 2*vareps*phi*grad(psi), which
follows from the Hopf-Cole pair plus the FP constraint -- derived in
docs/sinkhorn_reference.tex Sec 2.1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct


def _apply_neumann(phi: np.ndarray, decay: np.ndarray) -> np.ndarray:
    out = idct(decay * dct(phi, type=2, norm="ortho"), type=2, norm="ortho")
    return np.maximum(out, 0.0)


@dataclass
class HeatNeumann:
    """Heat semigroup under Neumann BCs (reflected Brownian motion on [0,L]),
    diagonalized by the DCT-II: H_t[phi]_hat(k) = exp(-eps*lambda_x(k)*t).
    """

    lambda_x: np.ndarray
    vareps: float

    def apply_full(self, phi: np.ndarray) -> np.ndarray:
        """Apply H_T for the full time horizon T=1."""
        decay = np.exp(-self.vareps * self.lambda_x)
        return _apply_neumann(phi, decay)

    def apply_time(self, phi: np.ndarray, t: float) -> np.ndarray:
        """Apply H_t for arbitrary t."""
        decay = np.exp(-self.vareps * self.lambda_x * t)
        return _apply_neumann(phi, decay)


@dataclass
class SinkhornFit:
    """Converged Hopf-Cole factors phi(0), psi(T) on problem's spatial grid,
    plus the heat kernel -- enough to evaluate rho(t,x)/m(t,x) at any t.
    """

    phi_0: np.ndarray
    psi_T: np.ndarray
    heat: HeatNeumann
    dx: float
    errors: np.ndarray  # (iters,) left-marginal L2 error per iteration
    iters: int
    error: float        # final LEFT-marginal error (the lagging one; see below)
    converged: bool
    walltime: float
    errors_right: np.ndarray = None  # (iters,) right-marginal L2 error per iteration
    error_right: float = 0.0          # final right-marginal error


def sinkhorn_fit(problem, vareps: float, max_iter: int, tol: float) -> SinkhornFit:
    """Run the Sinkhorn iteration to find phi(0), psi(T). Only uses
    problem's spatial grid (nx, dx, rho0, rho1, lambda_x) -- nt/dt are
    irrelevant here, since evaluation at specific times happens separately.
    """
    dx = problem.dx
    rho0, rho1 = problem.rho0, problem.rho1
    heat = HeatNeumann(lambda_x=problem.lambda_x, vareps=vareps)

    psi_0 = np.ones(problem.nx)
    errors = []
    errors_right = []
    converged = False

    t_start = time.perf_counter()
    phi_0 = psi_0
    psi_T = psi_0
    for _ in range(max_iter):
        # 1. Left marginal: phi(0) = rho0 / psi(0)
        phi_0 = rho0 / np.maximum(psi_0, 1e-300)
        # 2. Forward: phi(T) = H_T[phi(0)]
        phi_T = heat.apply_full(phi_0)
        # 3. Right marginal: psi(T) = rho1 / phi(T)
        psi_T = rho1 / np.maximum(phi_T, 1e-300)
        # 4. Backward: psi(0) = H_T[psi(T)]
        psi_0 = heat.apply_full(psi_T)

        # 5. Convergence. The LEFT marginal is the lagging one and is what
        # normally decides: step 1 imposed phi_0*psi_0 = rho0 exactly, then step
        # 4 replaced psi_0 and broke it. The RIGHT marginal was imposed in step 3
        # and nothing since has touched phi_T or psi_T, so it holds by
        # construction -- measured at 1e-16 or exactly 0 across
        # vareps = 1e-2..1e2. It is checked anyway because it is free (phi_T is
        # already in hand) and it is not guaranteed: where phi_T underflows, the
        # maximum(., 1e-300) guard above silently yields the wrong psi_T, and the
        # left-marginal error alone would not reveal it.
        err = np.sqrt(dx) * np.linalg.norm(phi_0 * psi_0 - rho0)
        err_right = np.sqrt(dx) * np.linalg.norm(phi_T * psi_T - rho1)
        errors.append(err)
        errors_right.append(err_right)
        if max(err, err_right) < tol:
            converged = True
            break
    walltime = time.perf_counter() - t_start

    return SinkhornFit(
        phi_0=phi_0,
        psi_T=psi_T,
        heat=heat,
        dx=dx,
        errors=np.array(errors),
        iters=len(errors),
        error=errors[-1],
        converged=converged,
        walltime=walltime,
        errors_right=np.array(errors_right),
        error_right=errors_right[-1],
    )


def sinkhorn_eval(fit: SinkhornFit, t_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate rho(t,x) and m(t,x_stag) at arbitrary times t_values (any
    array of reals in [0,1]), on fit's (fine) spatial grid.

    Returns (rho, mx): rho shape (len(t_values), nx), mx shape
    (len(t_values), nx-1) on the staggered x-grid.
    """
    nx = fit.phi_0.shape[0]
    nxm = nx - 1
    dx = fit.dx
    vareps = fit.heat.vareps

    rho_out = np.zeros((len(t_values), nx))
    mx_out = np.zeros((len(t_values), nxm))
    for i, t in enumerate(t_values):
        phi_t = fit.heat.apply_time(fit.phi_0, t)
        psi_t = fit.heat.apply_time(fit.psi_T, 1.0 - t)
        rho_out[i] = phi_t * psi_t

        phi_stag = 0.5 * (phi_t[:nxm] + phi_t[1:nx])
        dpsi_stag = (psi_t[1:nx] - psi_t[:nxm]) / dx
        mx_out[i] = 2.0 * vareps * phi_stag * dpsi_stag

    return rho_out, mx_out
