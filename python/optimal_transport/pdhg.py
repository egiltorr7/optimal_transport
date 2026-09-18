"""Adaptive PDHG (Goldstein, Li & Yuan 2015), mapped onto this project's
f1/f2/A problem. See docs/adaptive_pdhg_goldstein.tex for the full
derivation, the notation map (their symbols -> ours, renamed only to avoid
clashing with ladmm.py's own gamma/tau/alpha/delta/D/P), and the verified
Moreau-decomposition identity this relies on to get prox_{f2*} from
prox_f2 (== prox_ke_cc_nu) without ever writing f2* down.

Solves the same min f1(x) + f2(A(x)) problem as ladmm.py/pipeline_nu.py,
but as a genuinely different two-variable (x, w) iteration -- no gamma,
delta, z_hat, or over-relaxation, and only one call into prox_f2 per
iteration (buried inside the Moreau step, whose *input* to prox_f2 is also
this iteration's KE-optimal state y -- returned as a bonus, not
recomputed). Deliberately a sibling of ladmm.py, not a mode flag on it: the
loop bodies don't share enough structure to make a shared abstraction
worthwhile, and keeping them separate means neither risks the other's bugs.

Duck-typed on State exactly like ladmm.py (works for any (rho, mx, ...)
dataclass exposing __add__/__sub__/__mul__/.norm()).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .state import State


@dataclass
class AdaptivePdhgOpts:
    kappa0: float = 0.99  # primal step (their tau_0); kappa0*sigma0 < 1 suffices
    sigma0: float = 0.99  # dual step (their sigma_0) -- unconditionally, since
    # ||A||_N^2 <= 1 exactly (nonuniform_grid_norms.tex Sec 6) for any grid, so
    # unlike their general setting we need no backtracking / no per-grid
    # estimate of rho(A^T A) to pick a safe kappa0, sigma0.
    omega0: float = 0.5   # adaptivity level (their alpha_0 -- renamed, clashes
    # with LadmmOpts.alpha, an unrelated over-relaxation parameter)
    Delta: float = 1.5    # imbalance threshold
    eta: float = 0.95     # adaptivity-level decay
    s: float = 1.0        # residual pre-scale (their s; 1.0 unless the two
    # state components are on very different physical scales)
    max_iter: int = 40_000
    eps_abs: float = 1e-10
    eps_rel: float = 1e-8
    norm_fn: Callable[[State], float] | None = None  # default: State.norm;
    # used for both p and d below -- see docs Sec 7 for why one shape-adaptive
    # norm_fn (as pipeline_nu.py's already is) suffices for both spaces.


@dataclass
class AdaptivePdhgInfo:
    dx: np.ndarray     # (iters,) ||x^{k+1}-x^k||
    dw: np.ndarray     # (iters,) ||w^{k+1}-w^k||
    p: np.ndarray      # (iters,) primal residual (lowercase -- Goldstein-Li-Yuan's
    # own p_k, unrelated to LadmmInfo's uppercase P)
    d: np.ndarray      # (iters,) dual residual (their d_k, unrelated to LadmmInfo's D)
    kappa: np.ndarray  # (iters,) primal step used at iteration k
    sigma: np.ndarray  # (iters,) dual step used at iteration k
    omega: np.ndarray  # (iters,) adaptivity level at iteration k
    iters: int
    converged: bool
    walltime: float


def adaptive_pdhg_solve(
    prox_f1: Callable[[State], State],
    prox_f2: Callable[[State, float], State],
    A_fn: Callable[[State], State],
    At_fn: Callable[[State], State],
    x0: State,
    w0: State,
    opts: AdaptivePdhgOpts,
) -> tuple[State, State, State, AdaptivePdhgInfo]:
    """
    prox_f1(v) -> x_new  N-weighted prox of f1 (an indicator -- step-independent,
                         hence no step argument, unlike ladmm.py's prox_f1(v,step))
    prox_f2(v, step) -> y_new  M-weighted prox of f2 (== prox_ke_cc_nu); called
                         here only inside the Moreau step (docs Sec 5-6), on
                         prox_f2(v/sigma, 1/sigma) -- its result IS this
                         iteration's KE-optimal state, returned as `y` below.
    A_fn(x) -> A(x)  affine (BCs baked in); At_fn(w) -> A^T*w, linear (no
                      constant -- see docs Sec 2 for why this asymmetry means
                      A_fn's differences must be taken by evaluating it twice,
                      never by evaluating it once on a state-difference).

    See docs/adaptive_pdhg_goldstein.tex Sec 7 for the derivation of every
    line below; this is a direct transcription, not a fresh derivation.
    """
    norm_fn = opts.norm_fn if opts.norm_fn is not None else (lambda v: v.norm())

    kappa, sigma, omega = opts.kappa0, opts.sigma0, opts.omega0

    x, w = x0, w0
    Ax = A_fn(x0)
    y = prox_f2(w0 * (1.0 / sigma), 1.0 / sigma)  # placeholder KE state, in case
    # max_iter==0; overwritten in the loop's first iteration otherwise.

    dx_hist = np.zeros(opts.max_iter)
    dw_hist = np.zeros(opts.max_iter)
    p_hist = np.zeros(opts.max_iter)
    d_hist = np.zeros(opts.max_iter)
    kappa_hist = np.zeros(opts.max_iter)
    sigma_hist = np.zeros(opts.max_iter)
    omega_hist = np.zeros(opts.max_iter)
    n_iters = opts.max_iter
    converged = False

    t_start = time.perf_counter()
    for t in range(opts.max_iter):
        kappa_hist[t] = kappa
        sigma_hist[t] = sigma
        omega_hist[t] = omega
        x_prev, w_prev, Ax_prev = x, w, Ax

        # --- x-update (eq. x in the doc) ---
        x = prox_f1(x_prev - At_fn(w_prev) * kappa)
        dx_hist[t] = norm_fn(x - x_prev)

        # --- extrapolate, then w-update via Moreau (eq. w) ---
        x_bar = x * 2.0 - x_prev
        Ax = A_fn(x)              # for the residual below (two evals, see docstring)
        Ax_bar = A_fn(x_bar)      # the extrapolated forward map, feeds v
        v = w_prev + Ax_bar * sigma
        y = prox_f2(v * (1.0 / sigma), 1.0 / sigma)
        w = v - y * sigma
        dw_hist[t] = norm_fn(w - w_prev)

        # --- residuals ---
        p = norm_fn((x_prev - x) * (1.0 / kappa) - At_fn(w_prev - w))
        d = norm_fn((w_prev - w) * (1.0 / sigma) - (Ax_prev - Ax))
        p_hist[t] = p
        d_hist[t] = d

        # --- convergence (relative + absolute, same style as ladmm.py) ---
        scale_p = norm_fn(At_fn(w))
        scale_d = max(norm_fn(Ax), norm_fn(y))
        eps_p = opts.eps_abs + opts.eps_rel * scale_p
        eps_d = opts.eps_abs + opts.eps_rel * scale_d
        if p <= eps_p and d <= eps_d:
            n_iters = t + 1
            converged = True
            break

        # --- step-size adaptation (Algorithm 2, renamed symbols) ---
        if p > opts.s * opts.Delta * d:
            kappa = kappa / (1.0 - omega)
            sigma = sigma * (1.0 - omega)
            omega = omega * opts.eta
        elif p < (opts.s / opts.Delta) * d:
            kappa = kappa * (1.0 - omega)
            sigma = sigma / (1.0 - omega)
            omega = omega * opts.eta
        # else: leave kappa, sigma, omega unchanged

    info = AdaptivePdhgInfo(
        dx=dx_hist[:n_iters], dw=dw_hist[:n_iters], p=p_hist[:n_iters], d=d_hist[:n_iters],
        kappa=kappa_hist[:n_iters], sigma=sigma_hist[:n_iters], omega=omega_hist[:n_iters],
        iters=n_iters, converged=converged, walltime=time.perf_counter() - t_start,
    )
    return x, w, y, info
