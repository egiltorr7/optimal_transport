"""Wire grid_nu + operators_nu + FP projection_nu + KE prox_nu into a
linearized-ADMM solve, on a non-uniform time grid. Sibling of pipeline.py.

Same A/B/prox_f1 wiring as pipeline.py (ladmm.py itself is entirely
grid-agnostic -- nothing about it changes here). The things that actually
depend on the grid being non-uniform:

  - solve_y calls prox_ke_cc_nu (now just prox_ke_cc verbatim -- see
    prox_nu.py's docstring: once the y-update's penalty carries the same
    dt_vec[n]*dx weight the KE objective f_2 already does, that weight
    cancels out of the pointwise argmin exactly, so the constant-sigma
    Euclidean prox is already correct, not an approximation).
  - At_fn's rho piece uses interp_t_at_rho_weighted (operators_nu.py), the
    genuine N,M-weighted adjoint of the consensus map's time-average --
    NOT ops.interp_t_at_rho, which is only the *Euclidean* adjoint and
    matches the weighted one exclusively where dt_vec is locally constant
    (docs/nonuniform_grid_norms.tex Sec 4). At_fn's mx piece
    (ops.interp_x_at_m) is untouched: space is uniform, so its Euclidean
    and weighted adjoints already coincide (same note, Sec 5).
  - projection_nu.py's proj_fokker_planck_banded_nu (prox_f1) now uses R's
    genuine N-weighted adjoint internally (docs/nonuniform_grid_norms.tex
    Sec 9.5) instead of the plain Euclidean one -- verified two ways
    (scripts/validate_nu.py's adjoint-identity check against R, and a
    direct feasibility check: R(x_out)=0 to ~1e-11 on genuinely non-uniform
    grids) before being wired in here.
  - norm_fn/dual_norm_fn use a genuine non-uniform Riemann sum,
    sum_n dt_vec[n] * dx * (...), instead of the constant dt*dx scaling --
    this is the piece that keeps "the norm is an L^2 approximation" true
    once dt is no longer constant (on a uniform grid dt_vec[n]==dt for
    every n and this reduces exactly to pipeline.py's formula). This is
    also, now, literally N and M themselves (docs/
    nonuniform_grid_xy_variables.tex Sec 11.1) -- promoted from a
    convergence diagnostic to the metric At_fn's adjoint above is actually
    weighted by.

IMPORTANT: the three items above (solve_y, At_fn, prox_f1's internal
adjoint) are not independently swappable -- each was individually correct
in isolation but wiring only one or two of them in at a time made the
LADMM iteration either diverge or converge to a visibly wrong answer
(docs/nonuniform_grid_norms.tex Sec 8, "Empirical note: why R^* can't
wait"). All three are switched on together here, and were re-validated
end-to-end (scripts/sb_gaussian_nonuniform.py, clustered grid) as a set
before this file was left in this state.

x0/y0's initial guess is built from the actual t_centers/t_edges instead of
linspace(0,1,...), so it starts from the correct (non-uniform) interpolant
between rho0 and rho1 rather than one keyed to the wrong grid.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .ladmm import LadmmInfo, LadmmOpts, ladmm_solve
from .ladmm_linesearch import (
    LinesearchLadmmInfo,
    LinesearchLadmmOpts,
    MalitskyLinesearch,
    ladmm_linesearch_solve,
)
from .ladmm_reordered import (
    GlyAdaptiveStepSize,
    ReorderedLadmmInfo,
    ReorderedLadmmOpts,
    ladmm_reordered_solve,
)
from .operators_nu import interp_t_at_rho_weighted
from .pdhg import AdaptivePdhgInfo, AdaptivePdhgOpts, adaptive_pdhg_solve
from .alia import AliaInfo, AliaOpts, alia_solve
from .prox_nu import prox_ke_cc_nu
from .state import State


@dataclass
class LadmmConfig:
    gamma: float  # initial value if step_size is set (reordered solver only)
    tau: float    # initial value if step_size is set (reordered solver only)
    max_iter: int
    eps_abs: float
    eps_rel: float
    alpha: float = 1.0
    step_size: GlyAdaptiveStepSize | None = None  # None = fixed gamma/tau
    # (default). Requires reordered=True: the rule is derived for that cycle
    # only (docs/adaptive_pdhg_goldstein.tex Part V), so the default solver
    # rejects it rather than silently running something unproven.
    linesearch: MalitskyLinesearch | None = None  # None = fixed gamma/tau.
    # Set it to use Malitsky-Pock's linesearch (ladmm_linesearch.py), which runs
    # ladmm.py's OWN unrotated cycle -- so it is incompatible with reordered=True
    # and with step_size, and the solve rejects those combinations rather than
    # silently picking one. gamma is then the initial penalty and tau is ignored,
    # since the linesearch ties tau = 1/(beta*gamma).
    reordered: bool = False  # False = ladmm.py's (r, x, y, delta) cycle
    # (default, unchanged). True = ladmm_reordered.py's (y, r, delta, x)
    # cycle, which is Goldstein-Li-Yuan Algorithm 1 exactly -- see
    # docs/adaptive_pdhg_goldstein.tex. With a constant step the two solve
    # the same problem; once the step adapts they genuinely differ.


@dataclass
class SolveResult:
    rho_stag: np.ndarray  # (ntm, nx)  FP-feasible density, staggered grid
    mx_stag: np.ndarray   # (nt, nxm)  FP-feasible momentum, staggered grid
    rho_cc: np.ndarray    # (nt, nx)   KE-optimal density, cell-centre grid
    mx_cc: np.ndarray     # (nt, nx)   KE-optimal momentum, cell-centre grid
    info: LadmmInfo | ReorderedLadmmInfo | LinesearchLadmmInfo


def discretize_then_optimize_nu(
    problem,
    projection: Callable[[State, object, float, object], State],
    projection_state: object,
    vareps: float,
    ladmm_cfg: LadmmConfig,
    x0: State | None = None,
    y0: State | None = None,
    x_ref: State | None = None,
    delta_ref: State | None = None,
    delta_mask: State | None = None,
    x_refs: dict | None = None,
) -> SolveResult:

    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    ntm, nxm = nt - 1, nx - 1
    dt_vec, dx = problem.dt_vec, problem.dx

    if x0 is None or y0 is None:
        t_stag = problem.t_edges[1:-1][:, None]  # (ntm, 1) -- interior rho nodes
        x0 = State(
            rho=(1 - t_stag) * rho0 + t_stag * rho1,  # (ntm, nx)
            mx=np.zeros((nt, nxm)),
        )
        t_cc = problem.t_centers[:, None]  # (nt, 1)
        y0 = State(
            rho=(1 - t_cc) * rho0 + t_cc * rho1,  # (nt, nx)
            mx=np.zeros((nt, nx)),
        )

    b = State.zeros_like(y0)

    gamma = ladmm_cfg.gamma
    zeros_nt = np.zeros(nt)

    def A_fn(x: State) -> State:
        return State(
            rho=ops.interp_t_at_phi(x.rho, rho0, rho1),
            mx=ops.interp_x_at_phi(x.mx, zeros_nt, zeros_nt),
        )

    def At_fn(v: State) -> State:
        return State(
            rho=interp_t_at_rho_weighted(v.rho, dt_vec),
            mx=ops.interp_x_at_m(v.mx),
        )

    def B_fn(y: State) -> State:
        return y * -1.0

    def prox_f1(v: State, step: float) -> State:
        return projection(v, problem, vareps, projection_state)

    def solve_y(delta: State, z_hat: State, gamma: float) -> State:
        # gamma passed in, not the outer closed-over gamma (used only for
        # x0/y0/opts' initial value) -- the reordered solver's step_size may
        # adapt it mid-run, see ladmm.py's solve_y docstring note.
        return prox_ke_cc_nu(z_hat - delta * (1.0 / gamma), 1.0 / gamma)

    dt_dual = 0.5 * (dt_vec[:-1] + dt_vec[1:])  # (ntm,), center-to-center weight

    def _weighted_sq_sum(v: State) -> float:
        # Non-uniform Riemann sum: rho/mx are stored as densities (see
        # pipeline.py), so weighting row n by its local time-quadrature
        # weight and dx directly gives the L^2([0,1]xR) approximation --
        # reduces to pipeline.py's dt*dx*sum(...) exactly when dt_vec is
        # constant. mx always lives on the nt cell centers (dt_vec) whether
        # v is the staggered (x) or cell-centre (y) state; rho lives there
        # too for y, but on the ntm *interior interfaces* for x, whose
        # natural quadrature weight is the center-to-center dual width.
        w_rho = dt_vec if v.rho.shape[0] == nt else dt_dual
        return dx * (np.sum(w_rho[:, None] * v.rho**2) + np.sum(dt_vec[:, None] * v.mx**2))

    def norm_fn(v: State) -> float:
        return float(np.sqrt(_weighted_sq_sum(v)))

    def dual_norm_fn(v: State) -> float:
        # See pipeline.py's dual_norm_fn: delta's numeric value is
        # grid-convention-independent (homogeneous degree 0), so the same
        # weighted formula carries over unchanged.
        return float(np.sqrt(_weighted_sq_sum(v)))

    if ladmm_cfg.linesearch is not None:
        if ladmm_cfg.reordered:
            raise ValueError(
                "linesearch and reordered=True are different schemes: the "
                "Malitsky-Pock linesearch is derived for ladmm.py's own "
                "unrotated (r, x, y, delta) cycle (docs/"
                "malitsky_linesearch_ladmm.tex), not the reordered one"
            )
        if ladmm_cfg.step_size is not None:
            raise ValueError(
                "linesearch and step_size are two different adaptive rules "
                "(they move (gamma, tau) along orthogonal directions); pick one"
            )
        if ladmm_cfg.alpha != 1.0:
            raise ValueError(
                f"linesearch does not implement over-relaxation; got "
                f"alpha={ladmm_cfg.alpha!r}, expected 1.0")
        ls_opts = LinesearchLadmmOpts(
            gamma0=gamma, max_iter=ladmm_cfg.max_iter, eps_abs=ladmm_cfg.eps_abs,
            eps_rel=ladmm_cfg.eps_rel, ls=ladmm_cfg.linesearch,
            norm_fn=norm_fn, dual_norm_fn=dual_norm_fn,
            x_ref=x_ref, delta_ref=delta_ref, delta_mask=delta_mask,
        )
        x, y, _, info = ladmm_linesearch_solve(
            prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, ls_opts)
        return SolveResult(rho_stag=x.rho, mx_stag=x.mx, rho_cc=y.rho,
                           mx_cc=y.mx, info=info)

    if ladmm_cfg.reordered:
        # y0 is unused by this solver -- the reordered cycle makes its own y
        # from (x^k, delta^k). It is still built above, for b's shape.
        if ladmm_cfg.alpha != 1.0:
            raise ValueError(
                "reordered=True does not implement over-relaxation; "
                f"got alpha={ladmm_cfg.alpha!r}, expected 1.0"
            )
        r_opts = ReorderedLadmmOpts(
            gamma=gamma,
            tau=ladmm_cfg.tau,
            max_iter=ladmm_cfg.max_iter,
            eps_abs=ladmm_cfg.eps_abs,
            eps_rel=ladmm_cfg.eps_rel,
            step_size=ladmm_cfg.step_size,
            norm_fn=norm_fn,
            dual_norm_fn=dual_norm_fn,
            x_ref=x_ref,
            delta_ref=delta_ref,
            delta_mask=delta_mask,
        )
        x, y, _, info = ladmm_reordered_solve(
            prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, r_opts
        )
        return SolveResult(rho_stag=x.rho, mx_stag=x.mx, rho_cc=y.rho, mx_cc=y.mx, info=info)

    if ladmm_cfg.step_size is not None:
        raise ValueError(
            "step_size requires reordered=True: the Goldstein-Li-Yuan rule is "
            "derived for ladmm_reordered.py's cycle, and the two are different "
            "algorithms once the step size moves (docs/adaptive_pdhg_goldstein.tex "
            "Part VI)"
        )

    opts = LadmmOpts(
        gamma=gamma,
        tau=ladmm_cfg.tau,
        max_iter=ladmm_cfg.max_iter,
        eps_abs=ladmm_cfg.eps_abs,
        eps_rel=ladmm_cfg.eps_rel,
        alpha=ladmm_cfg.alpha,
        norm_fn=norm_fn,
        dual_norm_fn=dual_norm_fn,
        x_ref=x_ref,
        delta_ref=delta_ref,
        delta_mask=delta_mask,
        x_refs=x_refs,
        # rho-only companion to norm_fn, so x_refs are scored in the density
        # alone as well as in the full state. Same row weights as
        # _weighted_sq_sum, with the mx term dropped.
        rho_norm_fn=(lambda v: float(np.sqrt(
            dx * np.sum((dt_vec if v.rho.shape[0] == nt else dt_dual)[:, None] * v.rho**2)))),
    )

    x, y, _, info = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts)

    return SolveResult(rho_stag=x.rho, mx_stag=x.mx, rho_cc=y.rho, mx_cc=y.mx, info=info)


@dataclass
class PdhgConfig:
    """See pdhg.py's AdaptivePdhgOpts for what each field means and
    docs/adaptive_pdhg_goldstein.tex for where the defaults come from."""
    kappa0: float = 0.99
    sigma0: float = 0.99
    omega0: float = 0.5
    Delta: float = 1.5
    eta: float = 0.95
    s: float = 1.0
    max_iter: int = 40_000
    eps_abs: float = 1e-10
    eps_rel: float = 1e-8


@dataclass
class SolveResultPdhg:
    rho_stag: np.ndarray  # (ntm, nx)  FP-feasible density, staggered grid
    mx_stag: np.ndarray   # (nt, nxm)  FP-feasible momentum, staggered grid
    rho_cc: np.ndarray    # (nt, nx)   KE-optimal density, cell-centre grid
    mx_cc: np.ndarray     # (nt, nx)   KE-optimal momentum, cell-centre grid
    info: AdaptivePdhgInfo


def discretize_then_optimize_nu_pdhg(
    problem,
    projection: Callable[[State, object, float, object], State],
    projection_state: object,
    vareps: float,
    pdhg_cfg: PdhgConfig,
    x0: State | None = None,
    w0: State | None = None,
) -> SolveResultPdhg:
    """Same problem as discretize_then_optimize_nu, solved by
    pdhg.adaptive_pdhg_solve (Goldstein-Li-Yuan) instead of ladmm_solve --
    see docs/adaptive_pdhg_goldstein.tex for the full derivation. A_fn,
    At_fn, prox_ke_cc_nu, and norm_fn below are exactly the same objects
    (same weighting, same N/M metrics) as discretize_then_optimize_nu's --
    only the loop that calls them differs. prox_f1 here drops the unused
    `step` argument that ladmm.py's prox_f1(v, step) carries only for
    interface uniformity (f1's prox is an indicator, hence step-independent
    either way).

    w0 defaults to zero (matches ladmm_solve's own delta0=0 default) --
    unlike x0/y0, w is a dual-type variable with no natural warm-start
    analogous to the rho0/rho1 interpolant.
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    ntm, nxm = nt - 1, nx - 1
    dt_vec, dx = problem.dt_vec, problem.dx

    if x0 is None:
        t_stag = problem.t_edges[1:-1][:, None]  # (ntm, 1) -- interior rho nodes
        x0 = State(
            rho=(1 - t_stag) * rho0 + t_stag * rho1,  # (ntm, nx)
            mx=np.zeros((nt, nxm)),
        )
    if w0 is None:
        w0 = State(rho=np.zeros((nt, nx)), mx=np.zeros((nt, nx)))

    zeros_nt = np.zeros(nt)

    def A_fn(x: State) -> State:
        return State(
            rho=ops.interp_t_at_phi(x.rho, rho0, rho1),
            mx=ops.interp_x_at_phi(x.mx, zeros_nt, zeros_nt),
        )

    def At_fn(v: State) -> State:
        return State(
            rho=interp_t_at_rho_weighted(v.rho, dt_vec),
            mx=ops.interp_x_at_m(v.mx),
        )

    def prox_f1(v: State) -> State:
        return projection(v, problem, vareps, projection_state)

    dt_dual = 0.5 * (dt_vec[:-1] + dt_vec[1:])  # (ntm,), center-to-center weight

    def _weighted_sq_sum(v: State) -> float:
        # Identical to discretize_then_optimize_nu's -- see there for the
        # derivation of why this one formula is correct for both the
        # staggered (x, w-as-if-x-shaped -- never happens here) and
        # cell-centre (w, y) shapes.
        w_rho = dt_vec if v.rho.shape[0] == nt else dt_dual
        return dx * (np.sum(w_rho[:, None] * v.rho**2) + np.sum(dt_vec[:, None] * v.mx**2))

    def norm_fn(v: State) -> float:
        return float(np.sqrt(_weighted_sq_sum(v)))

    opts = AdaptivePdhgOpts(
        kappa0=pdhg_cfg.kappa0,
        sigma0=pdhg_cfg.sigma0,
        omega0=pdhg_cfg.omega0,
        Delta=pdhg_cfg.Delta,
        eta=pdhg_cfg.eta,
        s=pdhg_cfg.s,
        max_iter=pdhg_cfg.max_iter,
        eps_abs=pdhg_cfg.eps_abs,
        eps_rel=pdhg_cfg.eps_rel,
        norm_fn=norm_fn,
    )

    x, _, y, info = adaptive_pdhg_solve(prox_f1, prox_ke_cc_nu, A_fn, At_fn, x0, w0, opts)

    return SolveResultPdhg(rho_stag=x.rho, mx_stag=x.mx, rho_cc=y.rho, mx_cc=y.mx, info=info)


@dataclass
class AliaConfig:
    """See alia.py's AliaOpts. gamma0 is the *only* step-size input -- the whole
    point of ALiA is that it corrects it -- so there is no tau and no ratio to
    satisfy: the stability bookkeeping that tau > gamma*||A||^2 does for LADMM
    is done here by the subroutine's own a_k/b_k singular-value estimates."""
    gamma0: float
    max_iter: int
    eps_abs: float
    eps_rel: float
    sigma: float = 1.0
    alia_eps: float = 1e-3   # ALiA's epsilon, NOT the SB vareps and not
    # eps_abs/eps_rel -- see the THREE DIFFERENT EPSILONS note in alia.py.
    # 0 is permitted: it forfeits only the point-convergence guarantee of their
    # Theorem 2.1, and their own experiments report eps=0 working just as well.


@dataclass
class SolveResultAlia:
    rho_stag: np.ndarray
    mx_stag: np.ndarray
    rho_cc: np.ndarray
    mx_cc: np.ndarray
    info: AliaInfo


def discretize_then_optimize_nu_alia(
    problem,
    projection: Callable[[State, object, float, object], State],
    projection_state: object,
    vareps: float,
    alia_cfg: AliaConfig,
    x0: State | None = None,
    y0: State | None = None,
    x_ref: State | None = None,
    delta_ref: State | None = None,
    delta_mask: State | None = None,
) -> SolveResultAlia:
    """Same problem as discretize_then_optimize_nu, solved by ALiA (Jang, Sun,
    Yin & Ryu, arXiv:2602.15000) Algorithm 1 + Subroutine 1 instead of LADMM.

    A_fn, At_fn, the FP projection and prox_ke_cc_nu are the same objects, with
    the same N/M weighting, as the LADMM path's -- only the loop differs. The
    one addition is inner_fn: ALiA needs weighted inner *products* (for its
    lambda^A/lambda^B and its singular-value estimates), not just norms, and
    those must be the metric that makes At_fn an adjoint. See alia.py's NORMS
    note for why a Euclidean inner product here would silently mis-scale the
    stepsize.
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    ntm, nxm = nt - 1, nx - 1
    dt_vec, dx = problem.dt_vec, problem.dx

    if x0 is None or y0 is None:
        t_stag = problem.t_edges[1:-1][:, None]
        x0 = State(rho=(1 - t_stag) * rho0 + t_stag * rho1, mx=np.zeros((nt, nxm)))
        t_cc = problem.t_centers[:, None]
        y0 = State(rho=(1 - t_cc) * rho0 + t_cc * rho1, mx=np.zeros((nt, nx)))

    b = State.zeros_like(y0)
    zeros_nt = np.zeros(nt)

    def A_fn(x: State) -> State:
        return State(rho=ops.interp_t_at_phi(x.rho, rho0, rho1),
                     mx=ops.interp_x_at_phi(x.mx, zeros_nt, zeros_nt))

    def At_fn(v: State) -> State:
        return State(rho=interp_t_at_rho_weighted(v.rho, dt_vec),
                     mx=ops.interp_x_at_m(v.mx))

    def B_fn(y: State) -> State:
        return y * -1.0

    # B = -I is self-adjoint in M, so B^T = -I too (y and u share the
    # cell-centre grid and its weighting). This makes b_k == 1 exactly.
    Bt_fn = B_fn

    def prox_f1(v: State, step: float) -> State:
        return projection(v, problem, vareps, projection_state)

    def prox_g1(v: State, step: float) -> State:
        return prox_ke_cc_nu(v, step)

    dt_dual = 0.5 * (dt_vec[:-1] + dt_vec[1:])

    def inner_fn(p: State, q: State) -> float:
        # Same shape-sniffing as _weighted_sq_sum in discretize_then_optimize_nu
        # (rho on nt rows is cell-centre, on ntm rows is staggered), so one
        # callable serves both the x- and the y/u-space. inner_fn(v,v) is
        # exactly that function, so norms agree with the LADMM path's.
        w_rho = dt_vec if p.rho.shape[0] == nt else dt_dual
        return float(dx * (np.sum(w_rho[:, None] * p.rho * q.rho)
                           + np.sum(dt_vec[:, None] * p.mx * q.mx)))

    opts = AliaOpts(
        gamma0=alia_cfg.gamma0, sigma=alia_cfg.sigma, alia_eps=alia_cfg.alia_eps,
        max_iter=alia_cfg.max_iter, eps_abs=alia_cfg.eps_abs,
        eps_rel=alia_cfg.eps_rel, inner_fn=inner_fn,
        grad_f2=None, grad_g2=None,   # f2 = g2 = 0 for this problem
        x_ref=x_ref, delta_ref=delta_ref, delta_mask=delta_mask,
    )
    x, y, _, info = alia_solve(prox_f1, prox_g1, A_fn, At_fn, B_fn, Bt_fn,
                               b, x0, y0, opts)
    return SolveResultAlia(rho_stag=x.rho, mx_stag=x.mx,
                           rho_cc=y.rho, mx_cc=y.mx, info=info)
