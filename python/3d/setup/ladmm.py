"""Over-relaxed linearized ADMM, following Fang, He, Liu, Yuan (2015).

Port of matlab/shared/1d/utils/ladmm_solve.m. The generic struct-arithmetic
helpers there (s_add/s_sub/s_scale/s_zeros/s_norm) exist only to work around
MATLAB structs lacking operator overloading; State's __add__/__sub__/__mul__/
norm() do that job here, so the loop below reads like the actual math.

Solves:
    min   f1(x) + f2(y)
    s.t.  A*x + B*y = b,   x in X, y in Y

The x-subproblem is linearized around x^t (the quadratic penalty is replaced
by its first-order Taylor expansion plus a proximal term), giving a gradient
step followed by a proximal operator. The y-subproblem is solved exactly.
Convergence requires tau > gamma * ||A||^2.

gamma and tau are fixed for the whole run. Adaptive step sizes live in
ladmm_reordered.py instead: the residual-balancing rule is derived for that
module's (y, r, delta, x) cycle, which is Goldstein-Li-Yuan's PDHG exactly
(docs/adaptive_pdhg_goldstein.tex), and does not carry over to this cycle
unchanged -- once the step size moves, the two are different algorithms.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np

from .state import State


@dataclass
class LadmmOpts:
    gamma: float     # penalty parameter (> 0)
    tau: float       # proximal penalty for the x-update (> 0, tau > gamma*||A||^2)
    max_iter: int
    eps_abs: float   # absolute floor for the D/P stopping criterion
    eps_rel: float   # relative tolerance for the D/P stopping criterion
    alpha: float = 1.0  # over-relaxation (1.0 = standard, (1,2) = over-relaxed)
    norm_fn: Callable[[State], float] | None = None  # default: State.norm
    dual_norm_fn: Callable[[State], float] | None = None  # default: norm_fn
    # dual_norm_fn measures delta (and At_fn(delta)) specifically. delta is a
    # KKT multiplier, i.e. ~ d f2(y)/dy -- a *gradient* w.r.t. the state, not
    # a state itself, kept as a separate slot in case its correct norm ever
    # diverges from norm_fn's. See pipeline.py: with rho/mx stored as genuine
    # densities, the two currently work out to the same formula.
    x_ref: State | None = None  # optional ground-truth state (same grid as x)
    # for tracking ||x^k - x_ref|| via norm_fn -- unlike D/P/dx/dy/dz (which
    # are all purely internal algorithmic quantities), this is the actual
    # distance to the true solution, when one is available (e.g. the
    # analytical or fine-grid Sinkhorn reference). Not part of the stopping
    # rule; purely diagnostic/comparative (e.g. for choosing gamma/tau).
    delta_ref: State | None = None  # optional ground-truth dual state (same
    # grid as delta/y), for tracking ||delta^k - delta_ref|| via norm_fn.
    # The true KKT condition is -delta* in the SUBdifferential d f2(y*) (see
    # pipeline.py's solve_y: y-update solves argmin f2(y) +
    # gamma/2||y-(z_hat-delta/gamma)||^2, optimality gives
    # d f2(y*) + gamma*(y*-z_hat) + delta* ~ 0, and at the fixed point
    # y*=z_hat so the middle term vanishes) -- f2 is smooth (df2 a singleton
    # = the classical gradient) wherever rho>0, so delta_ref can be computed
    # in closed form there, same as x_ref. See gamma_sweep.py. Wherever the
    # reference rho is at/near machine-zero, f2 is not differentiable (the
    # subdifferential is a set, not a point, at that boundary of its domain)
    # and dividing by the near-zero float rho loses all precision anyway --
    # delta_mask below excludes those points rather than comparing against
    # an arbitrary/unreliable single value.
    delta_mask: State | None = None  # optional 0/1 mask (same grid as delta),
    # multiplied into (delta - delta_ref) before the err_delta norm -- zeros
    # out points where delta_ref is not reliably defined (see delta_ref).
    x_refs: dict[str, State] | None = None  # optional NAMED ground-truth states
    # (same grid as x), each tracked per iteration exactly as x_ref is. x_ref
    # holds one reference; this holds several, so a single solve can be scored
    # against every reference available at once -- e.g. the same-grid and the
    # fine-grid Sinkhorn references, which differ by the reference's own
    # O(dx^2) and would otherwise need one solve each. Independent of x_ref;
    # both may be set. Purely diagnostic, never part of the stopping rule.
    rho_norm_fn: Callable[[State], float] | None = None  # optional rho-only
    # norm. When given, x_refs are ALSO tracked in rho alone, since norm_fn
    # mixes rho and mx and the density error is often the quantity of interest.


@dataclass
class LadmmInfo:
    dx: np.ndarray  # (iters,) ||x^{k+1} - x^k||
    dy: np.ndarray  # (iters,) ||y^{k+1} - y^k||
    dz: np.ndarray  # (iters,) ||delta^{k+1} - delta^k||
    D: np.ndarray   # (iters,) ||D^k||,  D^k = A x^k + B y^k - b  (constraint violation)
    P: np.ndarray   # (iters,) ||P^k||,  P^k = tau*(x^{k-1}-x^k) + gamma*A^T*(D^{k-1}-D^k)
    scale_D: np.ndarray  # (iters,) max(norm_fn(A_fn(x^k)), norm_fn(y^k)) -- D's stopping scale
    scale_P: np.ndarray  # (iters,) dual_norm_fn(At_fn(delta^k)) -- P's stopping scale
    err_x: np.ndarray  # (iters,) ||x^k - opts.x_ref||, nan-filled if x_ref is None
    err_delta: np.ndarray  # (iters,) ||delta^k - opts.delta_ref||, nan-filled if delta_ref is None
    err_x_refs: dict  # {name: (iters,)} ||x^k - x_refs[name]|| via norm_fn; {} if unused
    err_rho_refs: dict  # {name: (iters,)} the same in rho alone, via
    # rho_norm_fn; {} unless BOTH x_refs and rho_norm_fn are given
    tau: np.ndarray    # (iters,) tau used at iteration k (constant == opts.tau)
    gamma: np.ndarray  # (iters,) gamma used at iteration k (constant == opts.gamma)
    iters: int
    converged: bool
    walltime: float


def ladmm_solve(
    prox_f1: Callable[[State, float], State],
    solve_y: Callable[[State, State, float], State],
    A_fn: Callable[[State], State],
    At_fn: Callable[[State], State],
    B_fn: Callable[[State], State],
    b: State,
    x0: State,
    y0: State,
    opts: LadmmOpts,
) -> tuple[State, State, State, LadmmInfo]:
    """
    prox_f1(v, step) -> x_new   prox of f1: argmin f1(x) + (1/(2*step))||x-v||^2
                                 (step in the denominator -- large step means
                                 the prox stays close to minimizing f1 alone)
    solve_y(delta, z_hat, gamma) -> y_new  exact y-subproblem solve. gamma is
                                 passed explicitly (not closed over) because
                                 ladmm_reordered.py's sibling solver may adapt
                                 it mid-run and shares this callable's
                                 signature; a closure capturing gamma at
                                 construction time would silently keep using
                                 its initial value there.
    A_fn(x) -> A*x,  At_fn(v) -> A^T*v (linear part only),  B_fn(y) -> B*y

    Verified against a toy quadratic problem with a known closed-form fixed
    point (scratchpad validate_ladmm.py): x^{t+1} = prox_f1(x^t - g/tau,
    1/tau) correctly converges to the optimum under this convention.

    Tracks five residual histories (see LadmmInfo): the iterate-to-iterate
    change of each variable (dx, dy, dz), the constraint-violation "dual
    residual" D^k = A x^k + B y^k - b, and the "primal residual"
    P^k = tau*(x^{k-1}-x^k) + gamma*A^T*(D^{k-1}-D^k).

    Convergence is checked on D and P jointly (Boyd-style relative + absolute
    stopping rule, adapted to this problem's A_fn/At_fn/delta):
        eps_D = eps_abs + eps_rel * max(norm_fn(A_fn(x)), norm_fn(y))
        eps_P = eps_abs + eps_rel * dual_norm_fn(At_fn(delta))
        stop when D^k <= eps_D and P^k <= eps_P
    This is grid-independent given norm_fn/dual_norm_fn are themselves
    grid-independent (density-scale) norms: the relative term is a
    dimensionless ratio that cancels any common resolution-dependent scaling
    in numerator/denominator, and the eps_abs floor is meaningful across
    resolutions specifically because the norms converge to fixed physical
    quantities rather than drifting with grid size. delta gets its own norm
    slot (dual_norm_fn) since it's a gradient w.r.t. the state rather than a
    state itself -- see pipeline.py for the derivation of why, with rho/mx
    stored as densities, it currently works out to the same formula as
    norm_fn (this was not the case under the old mass-per-cell convention,
    where the two needed opposite dx corrections).
    """
    # Duck-typed rather than hardcoded to the 1D State import below: State
    # is only actually referenced here for type hints (all annotations are
    # strings, via `from __future__ import annotations`), so this also
    # works unchanged for state_2d.State2D or any other (rho, ...)
    # dataclass exposing the same .norm()/.zeros_like() protocol.
    norm_fn = opts.norm_fn if opts.norm_fn is not None else (lambda v: v.norm())
    dual_norm_fn = opts.dual_norm_fn if opts.dual_norm_fn is not None else norm_fn

    # Local copies; fixed for the whole run here (see the module docstring --
    # adaptive step sizes live in ladmm_reordered.py). opts is left untouched.
    tau, gamma = opts.tau, opts.gamma

    x, y = x0, y0
    delta = type(b).zeros_like(b)
    By = B_fn(y0)  # cache B*y to avoid calling B_fn twice per iteration
    D_prev = A_fn(x0) + By - b  # D^0, at the initial point

    dx_hist = np.zeros(opts.max_iter)
    dy_hist = np.zeros(opts.max_iter)
    dz_hist = np.zeros(opts.max_iter)
    D_hist = np.zeros(opts.max_iter)
    P_hist = np.zeros(opts.max_iter)
    scale_D_hist = np.zeros(opts.max_iter)
    scale_P_hist = np.zeros(opts.max_iter)
    err_x_hist = np.full(opts.max_iter, np.nan)
    err_delta_hist = np.full(opts.max_iter, np.nan)
    x_refs = opts.x_refs or {}
    rho_norm_fn = opts.rho_norm_fn
    err_x_refs_hist = {nm: np.full(opts.max_iter, np.nan) for nm in x_refs}
    err_rho_refs_hist = ({nm: np.full(opts.max_iter, np.nan) for nm in x_refs}
                         if rho_norm_fn is not None else {})
    tau_hist = np.zeros(opts.max_iter)
    gamma_hist = np.zeros(opts.max_iter)
    n_iters = opts.max_iter
    converged = False

    t_start = time.perf_counter()
    for t in range(opts.max_iter):
        tau_hist[t] = tau
        gamma_hist[t] = gamma
        x_prev, y_prev, By_prev, delta_prev = x, y, By, delta

        # --- x-subproblem (linearized) ---
        r = A_fn(x) + By_prev - b
        g = At_fn(gamma * r - delta)             # gradient w.r.t. x
        x = prox_f1(x - g * (1.0 / tau), 1.0 / tau)
        dx_hist[t] = norm_fn(x - x_prev)
        if opts.x_ref is not None:
            err_x_hist[t] = norm_fn(x - opts.x_ref)
        for nm, x_r in x_refs.items():
            diff_r = x - x_r
            err_x_refs_hist[nm][t] = norm_fn(diff_r)
            if rho_norm_fn is not None:
                err_rho_refs_hist[nm][t] = rho_norm_fn(diff_r)

        # --- over-relaxation ---
        Ax = A_fn(x)
        z_hat = Ax * opts.alpha + (b - By_prev) * (1.0 - opts.alpha)

        # --- y-subproblem (exact) ---
        y = solve_y(delta, z_hat, gamma)
        By = B_fn(y)
        dy_hist[t] = norm_fn(y - y_prev)

        # --- constraint violation D^k = A x^k + B y^k - b (uses the true Ax,
        #     not the over-relaxed z_hat, so it's exact even when alpha!=1) ---
        D = Ax + By - b
        D_hist[t] = norm_fn(D)

        # --- dual update ---
        delta = delta - (z_hat + By - b) * gamma
        dz_hist[t] = norm_fn(delta - delta_prev)
        if opts.delta_ref is not None:
            diff = delta - opts.delta_ref
            if opts.delta_mask is not None:
                # Field-wise, via dataclasses.fields, rather than the 1D
                # `State(diff.rho * mask.rho, diff.mx * mask.mx)` the 1D/2D
                # copies of this module hardcode: that form silently drops
                # my/mz here. Same duck-typing spirit as the rest of the loop
                # -- works for State, State2D and State3D alike.
                diff = type(diff)(
                    *[getattr(diff, f.name) * getattr(opts.delta_mask, f.name)
                      for f in fields(diff)]
                )
            err_delta_hist[t] = norm_fn(diff)

        # --- "primal residual" P^k ---
        P = (x_prev - x) * tau + At_fn(D_prev - D) * gamma
        P_hist[t] = norm_fn(P)
        D_prev = D

        # --- convergence on D and P jointly (relative + absolute) ---
        scale_D = max(norm_fn(Ax), norm_fn(y))
        scale_P = dual_norm_fn(At_fn(delta))
        scale_D_hist[t] = scale_D
        scale_P_hist[t] = scale_P
        eps_D = opts.eps_abs + opts.eps_rel * scale_D
        eps_P = opts.eps_abs + opts.eps_rel * scale_P
        if D_hist[t] <= eps_D and P_hist[t] <= eps_P:
            n_iters = t + 1
            converged = True
            break

    info = LadmmInfo(
        dx=dx_hist[:n_iters],
        dy=dy_hist[:n_iters],
        dz=dz_hist[:n_iters],
        D=D_hist[:n_iters],
        P=P_hist[:n_iters],
        scale_D=scale_D_hist[:n_iters],
        scale_P=scale_P_hist[:n_iters],
        err_x=err_x_hist[:n_iters],
        err_delta=err_delta_hist[:n_iters],
        err_x_refs={nm: v[:n_iters] for nm, v in err_x_refs_hist.items()},
        err_rho_refs={nm: v[:n_iters] for nm, v in err_rho_refs_hist.items()},
        tau=tau_hist[:n_iters],
        gamma=gamma_hist[:n_iters],
        iters=n_iters,
        converged=converged,
        walltime=time.perf_counter() - t_start,
    )
    return x, y, delta, info
