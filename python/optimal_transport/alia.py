"""ALiA -- Adaptive Linearized ADMM (Jang, Sun, Yin & Ryu, arXiv:2602.15000),
Algorithm 1 with Subroutine 1.

Their problem (1) is
    min  f1(x) + f2(x) + g1(y) + g2(y)   s.t.  A x + B y = c
with f2, g2 differentiable convex and f1, g1 closed convex proper. This
project's problem is that with

    their f1 = our f1   the FP-feasibility indicator (closed convex proper --
                        an indicator is explicitly in scope, which is why this
                        paper applies here at all)
    their f2 = 0        no differentiable part on x
    their g1 = our f2   the kinetic energy m^2/(2 rho)
    their g2 = 0
    their B  = -I,      their c absorbs A_fn's affine constant (it cancels
                        anyway -- see delta_u below)

so Theorem 2.1's "f2, g2 convex and locally smooth" holds trivially and its
conclusion (point convergence to a saddle point of L) applies.

WHAT THIS IS NOT. ALiA is *not* another step-size rule for ladmm.py's
iteration -- it is a different algorithm. Both blocks get a linearized
proximal step at the same stepsize gamma_{k+1}, so in particular the
y-subproblem is no longer solved exactly against the augmented Lagrangian the
way ladmm.py's solve_y does. Comparisons against ladmm.py/ladmm_reordered.py
are therefore between algorithms, not between step-size rules for one
algorithm.

With f2 = g2 = 0 the subroutine simplifies sharply, and the simplifications
are worth recording because they remove most of its machinery:

    ell_x = L_x = ell_y = L_y = 0   (the 0/0 = 0 convention, since the
                                     gradient differences vanish identically)
    delta_x = delta_y = 0
    Gamma_x = (1-2 eps) / (2 a_{k+1} sqrt((4-8 eps) sigma lambda^A_{k+1}))
              -- gamma_k cancels out of Gamma_x entirely, and Gamma_x = +inf
              whenever lambda^A_{k+1} <= 0 (the square root is not real)
    b_{k+1} = 1 exactly, since B = -I makes ||B^T du|| = ||du||

NORMS. The paper is written in the Euclidean inner product. This project works
in the N (staggered) / M (cell-centre) weighted metrics, and At_fn is the
*weighted* adjoint, not the Euclidean transpose (docs/nonuniform_grid_norms.tex
Sec 4). Every norm and inner product below therefore goes through opts.inner_fn,
which must be the weighted one -- a_k, b_k are meant to estimate singular values
of A and B, and lambda^A is a normalized inner product, so both are only correct
in the metric that makes At_fn an adjoint. Passing a Euclidean inner_fn with a
weighted At_fn would silently mis-scale the stepsize.

THREE DIFFERENT EPSILONS, so the parameter here is named alia_eps rather than
eps. Do not conflate:

    alia_eps   ALiA's own epsilon (Subroutine 1). A small constant used to
               establish *point* convergence, constrained by
               0 < alia_eps < min(1/2, 1/(4 sigma)). It enters the stepsize
               bound only through an 8*sigma*alia_eps term, so it is a
               theoretical device, not a performance knob.
    vareps     this project's Schrodinger-bridge diffusion parameter, the
               physics. Nothing to do with ALiA; it reaches the solver only
               through the FP projection that prox_f1 closes over.
    eps_abs,
    eps_rel    the stopping tolerances, same meaning as everywhere else here.

Duck-typed on State like the other solvers.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .state import State


@dataclass
class AliaOpts:
    gamma0: float          # initial stepsize (their gamma_0 > 0); the point of
    # the method is that this is the *only* step-size input and it self-corrects
    sigma: float = 1.0     # primal-dual stepsize ratio (their sigma > 0)
    alia_eps: float = 1e-3  # ALiA's own epsilon -- NOT the SB vareps, and not
    # eps_abs/eps_rel; see the THREE DIFFERENT EPSILONS note above. Required
    # 0 < alia_eps < min(1/2, 1/(4 sigma)).
    max_iter: int = 40_000
    eps_abs: float = 1e-10
    eps_rel: float = 1e-15
    inner_fn: Callable[[State, State], float] | None = None  # REQUIRED, no
    # default: every norm and inner product in the subroutine must be the
    # weighted one (see the NORMS note above). There is deliberately no
    # Euclidean fallback -- pairing one with a weighted At_fn mis-scales the
    # stepsize silently, so alia_solve raises instead of guessing.
    grad_f2: Callable[[State], State] | None = None  # None = f2 identically 0
    grad_g2: Callable[[State], State] | None = None  # None = g2 identically 0
    x_ref: State | None = None
    delta_ref: State | None = None   # compared against u (their multiplier);
    # note our delta = -u, see the sign note in alia_solve
    delta_mask: State | None = None


@dataclass
class AliaInfo:
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray        # ||u^{k+1} - u^k||
    D: np.ndarray         # ||A x^k + B y^k - c||, the constraint violation
    P: np.ndarray         # ||x^{k+1}-x^k||/gamma_k, a stationarity proxy
    gamma: np.ndarray     # (iters,) stepsize chosen for iteration k
    lamA: np.ndarray
    lamB: np.ndarray
    a: np.ndarray         # (iters,) singular-value estimate for A
    b: np.ndarray         # (iters,) ditto for B (== 1 exactly when B = -I)
    which: np.ndarray     # (iters,) which of the 4 terms attained the min in
    # gamma_{k+1}: 0 = (3/2)gamma_k, 1 = the a/b bound, 2 = Gamma_x, 3 = Gamma_y
    err_x: np.ndarray
    err_delta: np.ndarray
    iters: int
    converged: bool
    walltime: float


def alia_solve(
    prox_f1: Callable[[State, float], State],
    prox_g1: Callable[[State, float], State],
    A_fn: Callable[[State], State],
    At_fn: Callable[[State], State],
    B_fn: Callable[[State], State],
    Bt_fn: Callable[[State], State],
    b: State,
    x0: State,
    y0: State,
    opts: AliaOpts,
) -> tuple[State, State, State, AliaInfo]:
    """
    prox_f1(v, step) -> argmin f1 + (1/(2*step))||.-v||^2
    prox_g1(v, step) -> argmin g1 + (1/(2*step))||.-v||^2
    A_fn(x) -> A*x (may be affine; the constant cancels, see delta_u)
    At_fn, Bt_fn -> the *weighted* adjoints of A's linear part and of B
    b -> the constraint constant (their c, up to A_fn's affine part)

    Returns (x, y, u, info). Sign note: their Lagrangian carries +<u, Ax+By-c>
    where this project's delta convention carries -<delta, ...>, so
    u = -delta. err_delta below compares -u against opts.delta_ref.
    """
    # 0 is ALLOWED (note the non-strict lower bound). The paper's own
    # discussion: "A small constant eps>0 ... is used to establish the point
    # convergence result of Theorem 2.1, but we observe in the experiments of
    # Section 4 that eps=0 works just as well empirically." What eps>0 buys is
    # *point* convergence -- the whole sequence converging to one saddle point
    # -- not convergence as such. Nothing in Subroutine 1 degenerates at 0: eps
    # enters only via (1-2 eps), (2-4 eps)/3 and 4-lamA-lamB-8 sigma eps, which
    # at eps=0 are 1, 2/3 and (since |lambda| <= 1) at least 2. The upper bound
    # exists precisely to keep those positive. Since eps only ever shrinks
    # gamma, eps=0 gives the largest steps the subroutine permits.
    if not (0.0 <= opts.alia_eps < min(0.5, 1.0 / (4.0 * opts.sigma))):
        raise ValueError(
            f"alia_eps={opts.alia_eps!r} must satisfy "
            f"0 <= alia_eps < min(1/2, 1/(4*sigma)) = "
            f"{min(0.5, 1.0 / (4.0 * opts.sigma)):g} for sigma={opts.sigma!r}. "
            f"(This is ALiA's epsilon, not the SB vareps. 0 is permitted and "
            f"forfeits only the point-convergence guarantee of their Thm 2.1.)"
        )
    if opts.inner_fn is None:
        raise ValueError(
            "AliaOpts.inner_fn is required: a_k/b_k are singular-value estimates "
            "and lambda^A/lambda^B are normalized inner products, so both are only "
            "correct in the metric that makes At_fn an adjoint. Pass the weighted "
            "inner product (pipeline_nu.discretize_then_optimize_nu_alia does)."
        )
    inner = opts.inner_fn
    nrm = lambda v: float(np.sqrt(max(inner(v, v), 0.0)))
    sigma, eps = opts.sigma, opts.alia_eps

    # (x^0,y^0,u^0) = (x^{-1},y^{-1},u^{-1}) per Algorithm 1 line 1, so the
    # first iteration's differences vanish and lambda^A_1 = lambda^B_1 = 0.
    x, y = x0, y0
    u = type(b).zeros_like(b)
    x_prev, y_prev = x0, y0
    gamma = opts.gamma0

    n = opts.max_iter
    H = {k: np.zeros(n) for k in
         ("dx", "dy", "dz", "D", "P", "gamma", "lamA", "lamB", "a", "bb", "which")}
    err_x_h = np.full(n, np.nan)
    err_d_h = np.full(n, np.nan)
    n_iters, converged = n, False

    t_start = time.perf_counter()
    for t in range(n):
        Ax, By = A_fn(x), B_fn(y)
        resid = Ax + By - b                     # A x^k + B y^k - c
        H["D"][t] = nrm(resid)

        # --- Subroutine 1: direction. The affine part of A_fn cancels in the
        #     difference A(x^k)-A(x^{k-1}), and `resid` already carries it once,
        #     so no explicit constant is needed anywhere. ---
        dAx = Ax - A_fn(x_prev)
        dBy = By - B_fn(y_prev)
        du = resid + dAx * 2.0 + dBy * 2.0
        du_n = nrm(du)
        Atdu, Btdu = At_fn(du), Bt_fn(du)
        Atdu_n, Btdu_n = nrm(Atdu), nrm(Btdu)
        a_k = Atdu_n / du_n if du_n > 0 else 0.0     # 0/0 = 0 convention
        b_k = Btdu_n / du_n if du_n > 0 else 0.0

        dxv, dyv = x - x_prev, y - y_prev
        def _lam(gt, gt_n, dv, s):
            den = (gt_n**2 / (16.0 * s**2) + 4.0 * s**2 * inner(dv, dv)) if s > 0 else 0.0
            return inner(gt, dv) / den if den > 0 else 0.0
        lamA = _lam(Atdu, Atdu_n, dxv, a_k)
        lamB = _lam(Btdu, Btdu_n, dyv, b_k)

        # --- curvature estimates. f2 = g2 = 0 here => all four are 0 and
        #     delta_x = delta_y = 0; kept general so a smooth part could be
        #     added later without re-deriving the subroutine. ---
        def _curv(grad, v, v_prev):
            if grad is None:
                return 0.0, 0.0
            d = v_prev - v
            dn2 = inner(d, d)
            if dn2 <= 0:
                return 0.0, 0.0
            gd = grad(v_prev) - grad(v)
            return inner(gd, d) / dn2, float(np.sqrt(inner(gd, gd) / dn2))
        ell_x, L_x = _curv(opts.grad_f2, x, x_prev)
        ell_y, L_y = _curv(opts.grad_g2, y, y_prev)

        def _Gamma(ell, L, s, lam):
            delta_k = gamma**2 * L**2 - 2.0 * gamma * ell
            rad = (gamma * ell)**2 + (2.0 - 4.0 * eps) / 3.0 * (
                delta_k + 6.0 * sigma * s**2 * gamma**2 * lam)
            if rad < 0.0:
                return np.inf                      # "square root not real-valued"
            den = gamma * ell + np.sqrt(rad)
            return np.inf if den <= 0.0 else (1.0 - 2.0 * eps) / 2.0 * gamma / den
        Gx = _Gamma(ell_x, L_x, a_k, lamA)
        Gy = _Gamma(ell_y, L_y, b_k, lamB)

        ab = a_k**2 + b_k**2
        # 4 - lamA - lamB - 8*sigma*eps > 0 always: |lambda| <= 1 by AM-GM on the
        # denominator, and eps < 1/(4 sigma) gives 8*sigma*eps < 2.
        cap = (np.sqrt((4.0 - lamA - lamB - 8.0 * sigma * eps) / (32.0 * sigma * ab))
               if ab > 0 else np.inf)
        cands = np.array([1.5 * gamma, cap, Gx, Gy])
        gamma = float(np.min(cands))
        H["which"][t] = int(np.argmin(cands))
        if not np.isfinite(gamma) or gamma <= 0:
            raise FloatingPointError(
                f"ALiA stepsize became {gamma!r} at iteration {t}; candidates "
                f"(1.5*gamma_k, ab-bound, Gamma_x, Gamma_y) = {cands}"
            )
        H["gamma"][t] = gamma
        H["lamA"][t], H["lamB"][t] = lamA, lamB
        H["a"][t], H["bb"][t] = a_k, b_k

        # --- Algorithm 1 lines 4-6 ---
        u_prev = u
        u = u + du * (sigma * gamma)
        Atu = At_fn(u)
        gx = Atu if opts.grad_f2 is None else Atu + opts.grad_f2(x)
        gy = Bt_fn(u) if opts.grad_g2 is None else Bt_fn(u) + opts.grad_g2(y)
        x_new = prox_f1(x - gx * gamma, gamma)
        y_new = prox_g1(y - gy * gamma, gamma)

        H["dx"][t] = nrm(x_new - x)
        H["dy"][t] = nrm(y_new - y)
        H["dz"][t] = nrm(u - u_prev)
        H["P"][t] = H["dx"][t] / gamma
        if opts.x_ref is not None:
            err_x_h[t] = nrm(x_new - opts.x_ref)
        if opts.delta_ref is not None:
            diff = (u * -1.0) - opts.delta_ref      # our delta = -u
            if opts.delta_mask is not None:
                diff = type(diff)(*(getattr(diff, f) * getattr(opts.delta_mask, f)
                                    for f in diff.__dataclass_fields__))
            err_d_h[t] = nrm(diff)

        x_prev, y_prev = x, y
        x, y = x_new, y_new

        scale = max(nrm(A_fn(x)), nrm(y))
        if (H["D"][t] <= opts.eps_abs + opts.eps_rel * scale
                and H["P"][t] <= opts.eps_abs + opts.eps_rel * max(scale, 1.0)):
            n_iters, converged = t + 1, True
            break

    info = AliaInfo(
        dx=H["dx"][:n_iters], dy=H["dy"][:n_iters], dz=H["dz"][:n_iters],
        D=H["D"][:n_iters], P=H["P"][:n_iters], gamma=H["gamma"][:n_iters],
        lamA=H["lamA"][:n_iters], lamB=H["lamB"][:n_iters],
        a=H["a"][:n_iters], b=H["bb"][:n_iters], which=H["which"][:n_iters],
        err_x=err_x_h[:n_iters], err_delta=err_d_h[:n_iters],
        iters=n_iters, converged=converged, walltime=time.perf_counter() - t_start,
    )
    return x, y, u, info
