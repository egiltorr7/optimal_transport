"""Linearized ADMM with the cycle cut after the x-update instead of after the
dual update -- i.e. one iteration *starts* at the y-update.

See docs/adaptive_pdhg_goldstein.tex. Parts I-III show this reordering makes
the iteration literally Goldstein-Li-Yuan's PDHG Algorithm 1, in their own
ordering, for any step-size sequence; Part V writes out the adaptive rule
implemented here as GlyAdaptiveStepSize.

Solves the same problem as ladmm.py:
    min  f1(x) + f2(y)   s.t.  A*x + B*y = b
and is a deliberate sibling of it, not a mode flag on it -- ladmm.py's loop
is left untouched so existing results stay reproducible. The two differ only
in where one iteration is declared to end:

    ladmm.py        r, x, y, delta      (residual against the *stored* y)
    here            y, r, delta, x      (residual against the *fresh* y)

Consequences, all from the doc:

  - The x-update's bracket becomes gamma*r - delta^{k+1} == delta^k -
    2*delta^{k+1}, Goldstein-Li-Yuan's own fixed (2,-1) extrapolation, for
    any gamma sequence. In ladmm.py's cut the same quantity carries a
    (1+rho, -rho) coefficient with rho = gamma_k/gamma_{k-1}, so the two
    schemes coincide only while gamma is constant (doc Part VI).
  - No y0 is needed: a cycle takes (x^k, delta^k) and makes its own y.
  - One A_fn evaluation per iteration instead of two -- A(x^{k+1}) is formed
    once for the residual and carried into the next cycle's y-update.
  - The residual r^k = A x^k + B y^{k+1} - b carries mixed indices by
    construction (doc Part I): the y it measures against is the one this
    cycle just produced. Its single-index cousin D^{k+1} = A x^{k+1} +
    B y^{k+1} - b is what the stopping rule and the step-size rule use.

Duck-typed on State exactly like ladmm.py (works for any (rho, mx, ...)
dataclass exposing __add__/__sub__/__mul__/.norm()/.zeros_like()).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .state import State


@dataclass
class GlyAdaptiveStepSize:
    """Goldstein, Li & Yuan, "Adaptive Primal-Dual Splitting Methods for
    Statistical Learning and Image Processing", NIPS 2015, Algorithm 1 --
    translated to LADMM's (gamma, tau) by docs/adaptive_pdhg_goldstein.tex
    Part V.

    NOTE ON SOURCE: constants here follow the NIPS 2015 paper's Algorithm 1
    (alpha_0 = eta = 0.95, hard-coded imbalance factor 2). They are NOT the
    ones in the longer five-author tech report arXiv:1305.0546 ("Adaptive
    PDHG Methods for Saddle-Point Problems", Goldstein/Li/Yuan/Esser/
    Baraniuk), whose Algorithm 2 uses alpha_0 = 0.5, Delta = 1.5 and an
    extra residual pre-scale s. The two are different write-ups of related
    schemes; do not mix their constants.

    Their rule moves the two PDHG steps in *opposite* directions, but since
    theta_k = gamma_k and sigma_k = 1/tau_k, in these variables it scales
    gamma and tau by the *same* factor:

        ||p|| > 2||d||  ->  gamma, tau *= 1/(1-alpha);  alpha *= eta
        2||p|| < ||d||  ->  gamma, tau *= (1-alpha);    alpha *= eta

    with p = D^{k+1} (the constraint violation) and
    q = tau*(x^k - x^{k+1}) - gamma*A^T r^k, both measured in norm_fn's
    (grid-weighted l2) norm -- their Algorithm 1 writes ||p||, ||d||, so l2
    is the faithful choice here.

    Because both scale together, gamma_k/tau_k is invariant, so the LADMM
    stability condition tau > gamma*||A||^2 holds at every iteration iff it
    holds at k=0. That is what makes alpha_0 = 0.95 safe to use here even
    though it multiplies the steps by 20 on its first firing: the
    aggressiveness lands entirely on the common scale, never on the ratio.

    NOT IMPLEMENTED: their Algorithm 1 line 4, the backtracking check
    (their eq. 14) with the tau <- tau/2, sigma <- sigma/2 fallback, which
    in their design is mandatory because line 1 starts from deliberately
    "large" tau_0, sigma_0. Starting instead from a stability-satisfying
    (gamma_0, tau_0) -- as this project's configs do -- the ratio never
    moves, so no backtracking is needed to stay convergent; it would only
    be needed to *exploit* a deliberately over-large ratio.

    Trigger direction, worth stating because it is easy to get backwards:
    the steps increase when the *constraint violation* dominates, which is
    also what Boyd et al. (2011) Sec 3.4.1 does for plain ADMM (their primal
    residual is the constraint violation).
    """
    omega0: float = 0.95   # their alpha_0 (renamed: alpha is LadmmOpts'
    # over-relaxation). Their Algorithm 1 line 1 sets alpha_0 = eta = 0.95.
    eta: float = 0.95      # adaptivity-level decay; omega -> 0 so the
    # adaptation switches itself off, which is what makes the perturbation
    # summable in their convergence proof
    threshold: float = 2.0  # their hard-coded imbalance factor 2: adapt only
    # when one residual exceeds the other by more than this
    check_every: int = 1   # how often (in iterations) to check and adapt


@dataclass
class ReorderedLadmmOpts:
    gamma: float     # penalty (> 0); initial value if step_size is set
    tau: float       # proximal step for the x-update (> 0, tau > gamma*||A||^2);
    # initial value if step_size is set
    max_iter: int
    eps_abs: float
    eps_rel: float
    step_size: GlyAdaptiveStepSize | None = None  # None = fixed gamma/tau
    norm_fn: Callable[[State], float] | None = None       # default: State.norm
    dual_norm_fn: Callable[[State], float] | None = None  # default: norm_fn
    x_ref: State | None = None       # optional ground truth, diagnostic only
    delta_ref: State | None = None
    delta_mask: State | None = None
    # No `alpha`: ladmm.py's over-relaxation is not implemented here. Every
    # run in this project uses alpha=1, which is the iteration below; the
    # reordered cycle does not carry y^k across the boundary, so the
    # over-relaxed z_hat = alpha*A x + (1-alpha)*(b - B y) has no y^k to
    # refer to without reintroducing one.


@dataclass
class ReorderedLadmmInfo:
    """Field names match ladmm.py's LadmmInfo so the existing scripts and
    plotting helpers consume either one unchanged. Two notes on meaning:

      D  = ||A x^k + B y^k - b||, the constraint violation. Under the
           correspondence this is *exactly* Goldstein-Li-Yuan's primal
           residual p_k (doc Part III), so no separate `p` is stored.
      P  = ||tau*(x^{k-1}-x^k) - gamma*A^T r^{k-1}||, their dual residual
           q_k. This is NOT the same expression as ladmm.py's own P (which
           is tau*(x^{k-1}-x^k) + gamma*A^T(D^{k-1}-D^k)); both vanish at a
           fixed point but they are different constructions.
    """
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    D: np.ndarray
    P: np.ndarray
    scale_D: np.ndarray
    scale_P: np.ndarray
    err_x: np.ndarray
    err_delta: np.ndarray
    tau: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray  # (iters,) adaptivity level; all-nan if step_size is None
    iters: int
    converged: bool
    walltime: float


def ladmm_reordered_solve(
    prox_f1: Callable[[State, float], State],
    solve_y: Callable[[State, State, float], State],
    A_fn: Callable[[State], State],
    At_fn: Callable[[State], State],
    B_fn: Callable[[State], State],
    b: State,
    x0: State,
    opts: ReorderedLadmmOpts,
) -> tuple[State, State, State, ReorderedLadmmInfo]:
    """Same callables as ladmm.py's ladmm_solve, minus y0 (unused here).

    prox_f1(v, step) -> x_new           prox of f1, step = 1/tau
    solve_y(delta, z_hat, gamma) -> y   exact y-subproblem; gamma passed
                                        explicitly so step_size can adapt it
    A_fn(x) -> A*x (affine),  At_fn(v) -> A^T*v (linear part only),
    B_fn(y) -> B*y

    Returns (x, y, delta, info). y is this iteration's fresh y-subproblem
    solution, produced by the cycle rather than carried across it.
    """
    norm_fn = opts.norm_fn if opts.norm_fn is not None else (lambda v: v.norm())
    dual_norm_fn = opts.dual_norm_fn if opts.dual_norm_fn is not None else norm_fn

    tau, gamma = opts.tau, opts.gamma
    omega = opts.step_size.omega0 if opts.step_size is not None else np.nan

    x = x0
    delta = type(b).zeros_like(b)
    Ax = A_fn(x)  # A x^k, carried across the cycle boundary (one A_fn/iter)
    # Placeholder y so a max_iter==0 call still returns a usable state; the
    # loop's first iteration overwrites it. Matches pdhg.py's same guard.
    y = solve_y(delta, Ax, gamma)
    y_seen = False

    n = opts.max_iter
    dx_hist = np.zeros(n)
    dy_hist = np.full(n, np.nan)
    dz_hist = np.zeros(n)
    D_hist = np.zeros(n)
    P_hist = np.zeros(n)
    scale_D_hist = np.zeros(n)
    scale_P_hist = np.zeros(n)
    err_x_hist = np.full(n, np.nan)
    err_delta_hist = np.full(n, np.nan)
    tau_hist = np.zeros(n)
    gamma_hist = np.zeros(n)
    omega_hist = np.full(n, np.nan)
    n_iters = n
    converged = False

    t_start = time.perf_counter()
    for t in range(n):
        tau_hist[t] = tau
        gamma_hist[t] = gamma
        omega_hist[t] = omega
        x_prev, y_prev, delta_prev = x, y, delta

        # --- (1) y-subproblem (exact), from the *stored* x and delta ---
        y = solve_y(delta, Ax, gamma)
        By = B_fn(y)
        # nan (not inf) on the first iteration: there is no previous y to
        # difference against, and nan keeps it out of plots and min/max.
        dy_hist[t] = norm_fn(y - y_prev) if y_seen else np.nan
        y_seen = True

        # --- (2) residual, mixed indices by construction (see module doc) ---
        r = Ax + By - b

        # --- (3) dual update ---
        delta = delta - r * gamma
        dz_hist[t] = norm_fn(delta - delta_prev)

        # --- (4) x-subproblem (linearized). The bracket gamma*r - delta^{k+1}
        #         collapses to delta^k - 2*delta^{k+1} -- Goldstein-Li-Yuan's
        #         own fixed (2,-1) extrapolation (doc Part III). ---
        g = At_fn(delta_prev - delta * 2.0)
        x = prox_f1(x_prev - g * (1.0 / tau), 1.0 / tau)
        dx_hist[t] = norm_fn(x - x_prev)
        if opts.x_ref is not None:
            err_x_hist[t] = norm_fn(x - opts.x_ref)
        if opts.delta_ref is not None:
            diff = delta - opts.delta_ref
            if opts.delta_mask is not None:
                diff = type(diff)(*(getattr(diff, f) * getattr(opts.delta_mask, f)
                                    for f in diff.__dataclass_fields__))
            err_delta_hist[t] = norm_fn(diff)

        # --- residuals. Ax is reused next iteration, so A_fn runs once. ---
        Ax = A_fn(x)
        D = Ax + By - b                                  # D^{k+1}, == their p
        P = (x_prev - x) * tau - At_fn(r) * gamma        # their q
        D_hist[t] = norm_fn(D)
        P_hist[t] = norm_fn(P)

        # --- convergence, same rule and scales as ladmm.py ---
        scale_D = max(norm_fn(Ax), norm_fn(y))
        scale_P = dual_norm_fn(At_fn(delta))
        scale_D_hist[t] = scale_D
        scale_P_hist[t] = scale_P
        if (D_hist[t] <= opts.eps_abs + opts.eps_rel * scale_D
                and P_hist[t] <= opts.eps_abs + opts.eps_rel * scale_P):
            n_iters = t + 1
            converged = True
            break

        # --- step-size adaptation (Goldstein-Li-Yuan Algorithm 2, doc Part V).
        #     Both steps scale by the same factor, so gamma/tau -- and with it
        #     the tau > gamma*||A||^2 margin -- is preserved exactly. ---
        ss = opts.step_size
        if ss is not None and t % ss.check_every == 0:
            p_res, q_res = D_hist[t], P_hist[t]
            if p_res > ss.threshold * q_res:          # their Alg 1 line 7
                scale = 1.0 / (1.0 - omega)
            elif ss.threshold * p_res < q_res:        # their Alg 1 line 6
                scale = 1.0 - omega
            else:                                     # their Alg 1 line 8
                scale = None
            if scale is not None:
                gamma *= scale
                tau *= scale
                omega *= ss.eta

    info = ReorderedLadmmInfo(
        dx=dx_hist[:n_iters], dy=dy_hist[:n_iters], dz=dz_hist[:n_iters],
        D=D_hist[:n_iters], P=P_hist[:n_iters],
        scale_D=scale_D_hist[:n_iters], scale_P=scale_P_hist[:n_iters],
        err_x=err_x_hist[:n_iters], err_delta=err_delta_hist[:n_iters],
        tau=tau_hist[:n_iters], gamma=gamma_hist[:n_iters],
        omega=omega_hist[:n_iters],
        iters=n_iters, converged=converged,
        walltime=time.perf_counter() - t_start,
    )
    return x, y, delta, info
