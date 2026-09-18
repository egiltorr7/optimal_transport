"""LADMM with Malitsky-Pock's primal-dual linesearch.

Malitsky & Pock, "A first-order primal-dual algorithm with linesearch",
SIAM J. Optim. 28(1) 411-432, 2018 (arXiv:1608.08883), Algorithm 1, applied to
this project's *dual* saddle point and then rewritten as LADMM. The derivation
is in docs/malitsky_linesearch_ladmm.tex; the short version:

  their g  = f_2^*      their f = f_1^*  (so their f^* = f_1)
  their K  = -A^T       their x = -delta        their y = x_ladmm
  their tau_k = gamma_k                 their theta_k = gamma_k/gamma_{k-1}
  their beta ties the two steps:  tau_ladmm_k = 1/(beta * gamma_k)

The key structural fact, and the reason this needs NO reordering and NO lag:
Malitsky's extrapolation coefficient theta_k = tau_k/tau_{k-1} is *exactly* the
(1+rho_k, -rho_k) coefficient that ladmm.py's own unrotated (r, x, y, delta)
cycle already produces, with rho_k = gamma_k/gamma_{k-1} (see
docs/adaptive_pdhg_goldstein.tex Part VI, where that ratio was first identified
-- there as an obstruction to matching Goldstein-Li-Yuan's fixed (2,-1), here as
precisely the right thing). So ladmm.py's existing gradient bracket
`gamma*r - delta` IS Malitsky's extrapolated v-bar, with no auxiliary variable
and no stored delta^{k-1}: the extrapolation is implicit in the bracket.
Verified numerically against a direct transcription of their Algorithm 1
(scratch check): iterates agree to ~1e-15 and the step sequences agree exactly
until the iterates hit rounding noise.

HOW THIS DIFFERS FROM THE OTHER ADAPTIVE RULES HERE, in one line each:

  ladmm_reordered.GlyAdaptiveStepSize  scales gamma and tau by the SAME factor,
      so gamma/tau is invariant -- it moves the common scale and can never
      change the ratio.
  this module                          holds gamma*tau = 1/beta invariant and
      moves gamma, so gamma/tau = beta*gamma^2 DOES change. In (log gamma,
      log tau) coordinates the two rules move along orthogonal directions.

That matters because the ratio is where this problem's remaining headroom is:
the stability bound tau > gamma*||A||^2 becomes beta*gamma^2 < 1/||A||^2, and
Malitsky's linesearch test enforces the *directional* version
sqrt(beta)*gamma*||A dx|| <= kappa*||dx||, which can admit steps the worst-case
bound forbids whenever dx is not aligned with A's leading singular vector.

Duck-typed on State like the other solvers.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .state import State


@dataclass
class MalitskyLinesearch:
    """Malitsky-Pock Algorithm 1's linesearch parameters, renamed where their
    symbol already means something else here (same discipline as the docs):

        their beta   -> beta    no clash. Ties the steps: tau = 1/(beta*gamma).
        their mu     -> mu      no clash. Backtracking factor, in (0,1).
        their delta  -> kappa   CLASHES with this project's multiplier delta,
                                so renamed. Linesearch tolerance, in (0,1);
                                closer to 1 is more aggressive.
        their tau_k  -> gamma_k it *is* the LADMM penalty (see module docstring).
        their theta_k-> rho_k   it *is* ladmm.py's implicit extrapolation ratio.
    """
    beta: float = 1.0      # tau_ladmm = 1/(beta*gamma); also sets the stability
    # scale, since gamma/tau = beta*gamma^2
    mu: float = 0.7        # backtracking factor, (0,1)
    kappa: float = 0.99    # linesearch tolerance, (0,1)
    max_trials: int = 60   # safety cap per iteration; mu^60 ~ 1e-9 shrinkage
    growth: float = 1.0    # where in their step-2 interval
    # [gamma_{k-1}, gamma_{k-1} sqrt(1+rho_{k-1})] to start each linesearch:
    # 1.0 = always the largest (their first option), 0.0 = never increase gamma
    # (their second), in between = the "compromise" their Step 2 allows.
    #
    # A real cost knob here, not a detail. At growth=1 the trial always
    # over-reaches once gamma has settled, so nearly every iteration spends a
    # rejected trial -- measured ~2.0 trials/iteration on the SB problem -- and
    # every trial repeats the FP projection.


@dataclass
class LinesearchLadmmOpts:
    gamma0: float          # initial penalty (their tau_0 > 0)
    max_iter: int
    eps_abs: float
    eps_rel: float
    ls: MalitskyLinesearch = None  # set in __post_init__ if omitted
    norm_fn: Callable[[State], float] | None = None
    dual_norm_fn: Callable[[State], float] | None = None
    x_ref: State | None = None
    delta_ref: State | None = None
    delta_mask: State | None = None

    def __post_init__(self):
        if self.ls is None:
            self.ls = MalitskyLinesearch()


@dataclass
class LinesearchLadmmInfo:
    """Field names match ladmm.py's LadmmInfo where they mean the same thing,
    so the existing plotting/reporting helpers consume this unchanged."""
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    D: np.ndarray        # ||A x^k + B y^k - b||, the constraint violation
    P: np.ndarray        # tau*(x^{k-1}-x^k) + gamma*A^T(D^{k-1}-D^k), as ladmm.py
    scale_D: np.ndarray
    scale_P: np.ndarray
    err_x: np.ndarray
    err_delta: np.ndarray
    tau: np.ndarray      # 1/(beta*gamma_k)
    gamma: np.ndarray    # chosen by the linesearch
    rho: np.ndarray      # gamma_k/gamma_{k-1}, the implicit extrapolation ratio
    n_trials: np.ndarray  # linesearch trials at iteration k (1 = accepted first)
    iters: int
    converged: bool
    walltime: float


def ladmm_linesearch_solve(
    prox_f1: Callable[[State, float], State],
    solve_y: Callable[[State, State, float], State],
    A_fn: Callable[[State], State],
    At_fn: Callable[[State], State],
    B_fn: Callable[[State], State],
    b: State,
    x0: State,
    y0: State,   # accepted for interface parity with ladmm_solve; UNUSED -- see
    # the priming note in the body, which manufactures y^1 the way their step 1 does
    opts: LinesearchLadmmOpts,
) -> tuple[State, State, State, LinesearchLadmmInfo]:
    """Same callables as ladmm.py's ladmm_solve. The loop is ladmm.py's own
    (r, x, y, delta) cycle -- unrotated, unlagged -- with gamma_k chosen by the
    linesearch and tau_k tied to it by tau_k = 1/(beta*gamma_k).

    Cost note: each linesearch trial repeats the x-update, i.e. one prox_f1
    (the FP projection) plus one A_fn and one At_fn. The projection is the
    expensive operation in this solver, so `n_trials` in the info -- not the
    iteration count -- is the honest measure of work. This is a consequence of
    applying their algorithm to the dual saddle point as specified: their
    linesearch updates *their* y, which under this mapping is our x.
    """
    norm_fn = opts.norm_fn if opts.norm_fn is not None else (lambda v: v.norm())
    dual_norm_fn = opts.dual_norm_fn if opts.dual_norm_fn is not None else norm_fn
    ls = opts.ls
    if not (0.0 <= ls.growth <= 1.0):
        raise ValueError(f"growth must be in [0,1], got {ls.growth!r}")
    if not (0.0 < ls.mu < 1.0 and 0.0 < ls.kappa < 1.0 and ls.beta > 0.0):
        raise ValueError(f"need beta>0, mu in (0,1), kappa in (0,1); got {ls}")

    x = x0
    delta = type(b).zeros_like(b)
    Ax = A_fn(x)
    gamma, rho = opts.gamma0, 1.0     # their tau_0 and theta_0 = 1

    # --- Priming: Malitsky's Algorithm 1 runs its step 1 (the y/delta update)
    #     BEFORE the first linesearch, so that the first x-update already has a
    #     genuine (delta^k, delta^{k-1}) pair to extrapolate from. Without this
    #     the first bracket would be gamma_0 D^0 - delta^0 = v^1 rather than
    #     (1 + rho_1) v^1 -- short by the whole extrapolation factor -- and every
    #     later iterate would inherit the discrepancy. (Checked: without priming
    #     this solver departs from a direct transcription of their Algorithm 1
    #     by the 4th iteration.) y0 is therefore NOT used: the scheme
    #     manufactures its own y^1 here, exactly as their step 1 does.
    y = solve_y(delta, Ax, gamma)
    By = B_fn(y)
    D_prev = Ax + By - b
    Atd_prev = At_fn(delta)              # A^T delta^0
    delta = delta - D_prev * gamma       # delta^1
    Atd = At_fn(delta)                   # A^T delta^1 (Remark 1 cache)

    n = opts.max_iter
    H = {k: np.zeros(n) for k in ("dx", "dy", "dz", "D", "P", "scale_D",
                                  "scale_P", "tau", "gamma", "rho", "n_trials")}
    err_x_h = np.full(n, np.nan)
    err_d_h = np.full(n, np.nan)
    n_iters, converged = n, False

    t_start = time.perf_counter()
    for t in range(n):
        x_prev, y_prev, delta_prev, By_prev = x, y, delta, By
        Dk = Ax + By_prev - b                      # r^k, against the STORED y^k

        # --- Malitsky step 2: trial gamma_k in [gamma_{k-1}, gamma_{k-1} sqrt(1+rho_{k-1})],
        #     then backtrack. Their theta_{k-1} is our rho from the previous
        #     iteration, so the growth cap is sqrt(1+rho) -- the algorithm can
        #     grow the step by at most that much per iteration. ---
        gamma_prev = gamma
        # their step 2: any gamma_k in [gamma_{k-1}, gamma_{k-1} sqrt(1+rho_{k-1})]
        hi = np.sqrt(1.0 + rho)
        gamma = gamma_prev * (1.0 + ls.growth * (hi - 1.0))
        trials = 0
        while True:
            trials += 1
            rho = gamma / gamma_prev
            tau = 1.0 / (ls.beta * gamma)
            # ladmm.py's own bracket, A^T(gamma*D^k - delta^k). It equals the
            # Malitsky extrapolation (1+rho)v^k - rho*v^{k-1} with v = -delta
            # (module docstring), so by their Remark 1 its A^T comes from two
            # cached adjoints instead of a fresh At_fn on every trial:
            #   A^T[(1+rho)v^k - rho v^{k-1}] = -(1+rho) A^T d^k + rho A^T d^{k-1}.
            # Only rho varies inside the linesearch. At t=0 there is no
            # delta^{-1} to lean on, so the direct form is used there.
            if Atd_prev is None:
                g = At_fn(Dk * gamma - delta_prev)
            else:
                g = Atd * (-(1.0 + rho)) + Atd_prev * rho
            x_new = prox_f1(x_prev - g * (1.0 / tau), 1.0 / tau)
            Ax_new = A_fn(x_new)
            dxv = x_new - x_prev
            dxn = norm_fn(dxv)
            # their (8): sqrt(beta) tau_k ||K^* y^{k+1} - K^* y^k|| <= delta ||y^{k+1}-y^k||
            # here K^* = -A and their y is our x, so ||K^* dy|| = ||A dx||.
            lhs = np.sqrt(ls.beta) * gamma * norm_fn(Ax_new - Ax)
            if dxn == 0.0 or lhs <= ls.kappa * dxn or trials >= ls.max_trials:
                break
            gamma *= ls.mu
        H["n_trials"][t] = trials
        H["gamma"][t], H["tau"][t], H["rho"][t] = gamma, tau, rho

        x, Ax = x_new, Ax_new
        H["dx"][t] = dxn
        if opts.x_ref is not None:
            err_x_h[t] = norm_fn(x - opts.x_ref)

        # --- y- and delta-updates, with the SAME gamma_k the linesearch chose
        #     (Malitsky's step 1 of the next iteration) ---
        y = solve_y(delta_prev, Ax, gamma)
        By = B_fn(y)
        H["dy"][t] = norm_fn(y - y_prev)
        D = Ax + By - b
        H["D"][t] = norm_fn(D)
        delta = delta_prev - D * gamma
        H["dz"][t] = norm_fn(delta - delta_prev)
        if opts.delta_ref is not None:
            diff = delta - opts.delta_ref
            if opts.delta_mask is not None:
                diff = type(diff)(*(getattr(diff, f) * getattr(opts.delta_mask, f)
                                    for f in diff.__dataclass_fields__))
            err_d_h[t] = norm_fn(diff)

        # ladmm.py's own P, so the two are directly comparable
        P = (x_prev - x) * tau + At_fn(D_prev - D) * gamma
        H["P"][t] = norm_fn(P)
        D_prev = D
        Atd_prev, Atd = Atd, At_fn(delta)   # roll the cached adjoints forward

        sD = max(norm_fn(Ax), norm_fn(y))
        sP = dual_norm_fn(At_fn(delta))
        H["scale_D"][t], H["scale_P"][t] = sD, sP
        if (H["D"][t] <= opts.eps_abs + opts.eps_rel * sD
                and H["P"][t] <= opts.eps_abs + opts.eps_rel * sP):
            n_iters, converged = t + 1, True
            break

    info = LinesearchLadmmInfo(
        **{k: H[k][:n_iters] for k in H},
        err_x=err_x_h[:n_iters], err_delta=err_d_h[:n_iters],
        iters=n_iters, converged=converged,
        walltime=time.perf_counter() - t_start,
    )
    return x, y, delta, info
