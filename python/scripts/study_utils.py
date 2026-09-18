"""Shared machinery for grid_convergence.py / residual_history.py / the
multi-vareps study: runs the pipeline once per resolution and derives both
error metrics and residual histories from the same result, rather than
solving each resolution twice.

  L2 error:  sqrt(dt*dx * (sum((rho-rho_ana)^2) + sum((mx-mx_ana)^2)))
             -- density-scale weighted L2 norm, matching pipeline.py's
             norm_fn (rho, mx are stored as genuine densities, so this is a
             direct Riemann-sum weighting -- no dx-correction needed, unlike
             the old mass-per-cell convention).

  max error: max(|rho-rho_ana|) + max(|mx-mx_ana|)
             -- sum of two independent max-norms (not a pointwise-combined
             max -- on the staggered grid rho (ntm,nx) and mx (nt,nxm) have
             different shapes, so there's no shared (i,j) index set to
             combine pointwise). No dx-correction needed (density convention).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct

from optimal_transport.grid import setup_problem
from optimal_transport.operators import build_operators
from optimal_transport.pipeline import discretize_then_optimize
from optimal_transport.problems import analytical_sb_gaussian, prob_gaussian
from optimal_transport.projection_expsemi import precomp_expsemi_proj, proj_fokker_planck_expsemi
from optimal_transport.sinkhorn import sinkhorn_eval, sinkhorn_fit
from optimal_transport.time_grid import (
    clustered_time_grid,
    graded_ends_time_grid,
    refine_ends_time_grid,
    uniform_time_grid,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class _BooleanOptionalActionBackport(argparse.Action):
    """argparse.BooleanOptionalAction for Python 3.8.

    The real class landed in 3.9; this interpreter is 3.8, where reaching for
    argparse.BooleanOptionalAction raises AttributeError at parser-build time
    (i.e. every script using it dies on import of its own parse_args). Same
    behaviour as CPython's: registers both --flag and --no-flag, storing
    True/False respectively, with no argument consumed either way.
    """

    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        opts = []
        for option_string in option_strings:
            opts.append(option_string)
            if option_string.startswith("--"):
                opts.append("--no-" + option_string[2:])
        super().__init__(option_strings=opts, dest=dest, nargs=0, default=default,
                         required=required, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

    def format_usage(self):
        return " | ".join(self.option_strings)


# Scripts import this instead of touching argparse.BooleanOptionalAction
# directly, so they keep working on 3.8 and pick up the stdlib class as soon
# as they run on 3.9+.
BooleanOptionalAction = getattr(argparse, "BooleanOptionalAction", _BooleanOptionalActionBackport)

# analytical_sb_gaussian assumes free-space Brownian motion on all of R, only
# accurate while the density's spread stays small relative to the domain.
# Above this vareps, use the Sinkhorn-Hopf-Cole solver instead, which solves
# the exact discrete-marginal bridge on the actual (Neumann/reflecting)
# domain -- see sinkhorn.py.
SINKHORN_VAREPS_THRESHOLD = 5e-3
SINKHORN_MAX_ITER = 2000
SINKHORN_TOL = 1e-10
# Sinkhorn is fit ONCE per vareps on a spatial grid this fine, then coarsened
# down to each test resolution -- so the reference itself isn't a "moving
# target" with its own resolution-dependent error at each LADMM grid size.
# Must be an integer multiple of every resolution used (checked below).
SINKHORN_NX_FINE = 4096


TIME_GRIDS = ("uniform", "clustered", "graded_ends", "refine_ends")


def build_time_grid(name: str, nt: int, strength: float = 1.0, n_boundary: int = 5):
    """(TimeGrid, label, tag) for a named time grid.

    Shared by every non-uniform driver so the grid names, their parameters,
    and the tag that lands in result filenames stay defined in one place:

      uniform      uniform_time_grid(nt).
      clustered    clustered_time_grid(nt, strength) -- cosine clustering at
                   both ends. strength=0 reproduces uniform_time_grid
                   *exactly* (verified bit-for-bit), 1 is full clustering.
      graded_ends  graded_ends_time_grid(nt, n_boundary).
      refine_ends  refine_ends_time_grid(nt) -- cell 0 and cell nt-1 each
                   split in half, so the grid has nt+2 cells, not nt. Callers
                   that report resolution should read problem.nt back rather
                   than assume the nt they passed in.

    `strength` is ignored except by "clustered", `n_boundary` except by
    "graded_ends"; passing them for the other grids is harmless so callers
    can forward their parsed args unconditionally.
    """
    if name == "uniform":
        return uniform_time_grid(nt), "uniform", "uniform"
    if name == "clustered":
        return (clustered_time_grid(nt, strength=strength),
                f"clustered, strength={strength:g}", f"clustered_s{strength:g}")
    if name == "graded_ends":
        return (graded_ends_time_grid(nt, n_boundary=n_boundary),
                f"graded_ends, n_boundary={n_boundary}", f"graded_ends_nb{n_boundary}")
    if name == "refine_ends":
        return refine_ends_time_grid(nt), "refine_ends", "refine_ends"
    raise ValueError(f"unknown time grid {name!r}; expected one of {TIME_GRIDS}")

def coarsen_rho(rho_fine: np.ndarray, factor: int) -> np.ndarray:
    """Average blocks of `factor` consecutive fine cells into one coarse cell
    (rho is a density, not cell-integrated mass, so coarsening averages --
    the average density over the coarse cell equals the mean of the fine
    densities it contains, since all fine sub-cells have equal width).
    rho_fine shape (..., n_fine) -> (..., n_fine // factor).
    """
    n_fine = rho_fine.shape[-1]
    n_coarse = n_fine // factor
    return rho_fine[..., : n_coarse * factor].reshape(*rho_fine.shape[:-1], n_coarse, factor).mean(-1)


def sample_rho_pointwise(rho_fine: np.ndarray, nx: int, L: float = 1.0,
                         renormalize: bool = True) -> np.ndarray:
    """Sample a fine-grid density at the COARSE grid's own cell centres.

    Pointwise sampling, NOT block-averaging -- and the distinction is not
    cosmetic. grid_nu.setup_problem_nu builds the solver's marginals by
    evaluating the pdf pointwise at the coarse cell centres and renormalizing,
        rho0 = pdf(xx) / (pdf(xx).sum() * dx),
    so a block-averaged reference (coarsen_rho) disagrees with the solver's own
    rho0 by the coarsening error. Since the solver reproduces the marginals
    *exactly* at t=0 and t=1 (the boundary data is baked into A), that
    disagreement shows up as a spurious error floor at both endpoints, present
    only where a Sinkhorn reference is used (vareps > SINKHORN_VAREPS_THRESHOLD).
    Measured, sigma=0.05: 8.343e-03 at nx=64 and 2.089e-03 at nx=128, matching
    ||coarsen(rho0_fine) - rho0|| to four digits and falling as O(h^2).

    Both grids are uniform cell-centred on [0, L] with nx_fine a multiple of nx,
    so the coarse centres are either exactly a fine centre (odd factor) or
    exactly midway between two (even factor) -- the two branches below are exact
    linear interpolation, not an approximation.

    renormalize rescales to unit discrete mass on the coarse grid, matching
    setup_problem_nu's own convention for rho0/rho1; without it the sample
    carries the fine grid's normalization and is off by O(h^2) in mass.
    """
    n_fine = rho_fine.shape[-1]
    factor = n_fine // nx
    if factor * nx != n_fine:
        raise ValueError(f"nx={nx} must divide n_fine={n_fine}")
    j = np.arange(nx)
    if factor % 2 == 0:                     # coarse centre midway between two fine centres
        lo = j * factor + factor // 2 - 1
        out = 0.5 * (rho_fine[..., lo] + rho_fine[..., lo + 1])
    else:                                   # coarse centre coincides with a fine centre
        out = rho_fine[..., j * factor + (factor - 1) // 2]
    if renormalize:
        dx = L / nx
        out = out / (out.sum(axis=-1, keepdims=True) * dx)
    return out


def resample_rho_spectral(rho_fine: np.ndarray, nx: int, L: float = 1.0,
                          renormalize: bool = True) -> np.ndarray:
    """Evaluate the fine grid's own DCT-II series at the COARSE cell centres.

    The spatial analogue of what sinkhorn_eval already does in time: the fine
    cell-centred DCT-II basis is cos(k*pi*x) sampled at the fine centres, so the
    same series can be summed at ANY x. No reduction operator, no shared-point
    requirement, and it works for every nx.

    This exists because true subsampling is impossible here: a cell-centred grid
    puts its points at (2i-1)/(2nx), i.e. only ODD multiples of 1/(2nx), so the
    centres of nx and 2nx share NO point at all (odd/64 = odd'/128 would need
    2*odd = odd'). Coarse centres coincide with fine centres only for an ODD
    factor nx_fine/nx, and factor(2nx) = factor(nx)/2, so at most one nx in a
    doubling sequence can qualify -- no choice of nx_fine fixes a whole
    refinement sweep.

    Accuracy, sigma=0.05, nx_fine=4096: agrees with setup_problem's own
    pointwise marginal to ~1e-15 (machine precision), versus 6.1e-6 for
    sample_rho_pointwise's 2-point interpolation and 3.3e-2 (nx=32) to 5.2e-4
    (nx=256) for coarsen_rho's block average.
    """
    n_fine = rho_fine.shape[-1]
    x = (np.arange(1, nx + 1) - 0.5) / nx          # coarse centres on [0,1]
    F = dct(rho_fine, type=2, norm="ortho", axis=-1)
    k = np.arange(1, n_fine)
    basis = np.cos(np.pi * np.outer(x, k))         # (nx, n_fine-1)
    out = (F[..., :1] / np.sqrt(n_fine)
           + np.sqrt(2.0 / n_fine) * F[..., 1:] @ basis.T)
    if renormalize:
        out = out / (out.sum(axis=-1, keepdims=True) * (L / nx))
    return out


def _coarsen_mx(mx_fine: np.ndarray, factor: int) -> np.ndarray:
    """Pick the aligned fine staggered point (mx is a point-valued flux
    density ~ m_density(x), not cell-integrated, and needs no rescaling now
    that it's stored as a genuine density; staggered points at integer
    multiples of the coarse spacing align exactly with fine points, unlike
    cell centres -- verified directly, see conversation).
    mx_fine shape (..., nxm_fine) -> (..., nxm_fine // factor).
    """
    nxm_fine = mx_fine.shape[-1]
    nxm_coarse = nxm_fine // factor
    idx_fine = (np.arange(nxm_coarse) + 1) * factor - 1
    return mx_fine[..., idx_fine]


SINKHORN_GRIDS = ("fine", "same")


def sinkhorn_reference_fit(problem, vareps, grid="fine", make_fine_problem=None,
                           verbose=True):
    """Fit Sinkhorn for use as a reference, on one of two spatial grids.

    The two choices measure different things, and neither dominates -- see
    docs/sinkhorn_reference.tex:

      "fine"  Fit on SINKHORN_NX_FINE, then bring rho to the test grid with
              reference_solution's rho_reduce.  The reference is then close to
              the CONTINUUM solution, so the measured error is the solver's
              total discretization error -- the right choice for a convergence
              study (order, constants).  Costs a spatial reduction, and the
              reference's own evolution uses the FINE Laplacian, so it differs
              from the solver's discrete evolution at O(dx^2); that shows up as
              a plateau in the interior and as O(dx^2/dt) in the endpoint rows.

      "same"  Fit on the solver's OWN grid.  No reduction exists to get wrong,
              and sinkhorn_fit reads problem.rho0/rho1 directly, so the
              reference's marginals ARE the solver's marginals to within tol.
              The FP constraint and Sinkhorn share a spatial discretization
              (both the DCT-Neumann Laplacian of the same lambda_x) and
              Sinkhorn is exact in time, so the residual of this reference goes
              to ZERO at 2nd order under nt refinement: it isolates the TIME
              discretization error with nothing spatial left in it.  Measured at
              nx=128, vareps=0.1: interior residual 1.7e-4 -> 5.5e-8 for
              nt=128..8192, against a 3.6e-4 plateau for "fine".  The limitation
              is structural -- being spatially consistent by construction, it is
              BLIND to spatial error and cannot support a convergence claim.

    Either way the fit is exact in time (the semigroup is evaluated at whatever
    t is asked for), so nt never enters.  `make_fine_problem` is a callable
    nx -> problem, needed only for "fine", since the uniform and non-uniform
    drivers build their problems differently.
    """
    if grid not in SINKHORN_GRIDS:
        raise ValueError(f"grid must be one of {SINKHORN_GRIDS}, got {grid!r}")

    if grid == "same":
        if verbose:
            print(f"  Sinkhorn reference: fitting on the solver's own grid "
                  f"(nx={problem.nx}) -- no spatial reduction, marginals match exactly")
        fit_problem = problem
    else:
        if make_fine_problem is None:
            raise ValueError('grid="fine" requires make_fine_problem')
        if SINKHORN_NX_FINE % problem.nx:
            raise ValueError(
                f"SINKHORN_NX_FINE={SINKHORN_NX_FINE} must be a multiple of "
                f"nx={problem.nx} to reduce down to it"
            )
        if verbose:
            print(f"  Sinkhorn reference: fitting at nx={SINKHORN_NX_FINE}, "
                  f"reduced to nx={problem.nx}")
        fit_problem = make_fine_problem(SINKHORN_NX_FINE)

    fit = sinkhorn_fit(fit_problem, vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
    if verbose:
        print(f"    iters={fit.iters}  converged={fit.converged}  "
              f"marginal err: t=0 {fit.error:.2e}  t=1 {fit.error_right:.2e}  "
              f"wall={fit.walltime:.2f}s")
    return fit


def reference_solution(problem, vareps, ops, sink_fit=None, rho_reduce="spectral"):
    """Return (rho_stag, mx_stag, rho_cc, mx_cc) reference arrays, matching
    the shapes/grids of SolveResult, choosing analytical_sb_gaussian or
    Sinkhorn-Hopf-Cole (reduced from sink_fit, a fine-grid SinkhornFit)
    depending on vareps.

    rho_reduce selects how the fine-grid Sinkhorn rho reaches the coarse grid:

      "spectral"  (default) resample_rho_spectral -- sums the fine grid's DCT-II
                  series at the coarse cell centres.  Matches setup_problem's own
                  marginals to ~1e-15.
      "pointwise" sample_rho_pointwise -- 2-point interpolation; ~6e-6.
      "average"   coarsen_rho -- block average, the previous behaviour.  Disagrees
                  with the solver's rho0/rho1 by the O(h^2) coarsening error which,
                  because the marginals are baked into A exactly at t=0 and t=1,
                  appears as a spurious error floor at BOTH endpoints.

    mx needs none of this: staggered points sit at j*dx and NODE grids are nested,
    so _coarsen_mx subsamples exactly.  Time evaluation is exact via the semigroup
    in every case (sinkhorn_eval at arbitrary t) -- this is purely spatial.

    rho_reduce is IGNORED when sink_fit already lives on problem's own grid (see
    sinkhorn_reference_fit's grid="same"): the factor is then 1 and all three
    reducers are the identity -- block-averaging over blocks of one, picking each
    point, and evaluating the series at its own nodes respectively. The
    renormalization the latter two apply is a no-op there too, since the discrete
    mass sum(phi*psi)*dx is exactly conserved (the DCT-diagonalized Laplacian is
    symmetric, so d/dt <phi,psi> = 0) and the endpoints are pinned to tol.
    Verified: the three agree to <= 8.3e-16 relative, and mx is bit-identical.
    """
    _REDUCERS = {
        "spectral": lambda r, f: resample_rho_spectral(r, problem.nx, problem.L),
        "pointwise": lambda r, f: sample_rho_pointwise(r, problem.nx, problem.L),
        "average": coarsen_rho,
    }
    if rho_reduce not in _REDUCERS:
        raise ValueError(f"rho_reduce must be one of {sorted(_REDUCERS)}, got {rho_reduce!r}")
    _reduce = _REDUCERS[rho_reduce]
    nt, nx = problem.nt, problem.nx
    zeros_nt = np.zeros(nt)

    if vareps <= SINKHORN_VAREPS_THRESHOLD:
        rho_stag, mx_stag = analytical_sb_gaussian(problem, vareps)
        rho_cc = ops.interp_t_at_phi(rho_stag, problem.rho0, problem.rho1)
        mx_cc = ops.interp_x_at_phi(mx_stag, zeros_nt, zeros_nt)
        return rho_stag, mx_stag, rho_cc, mx_cc

    assert sink_fit is not None, "vareps > threshold requires a pre-fit SinkhornFit"
    nx_fine = sink_fit.phi_0.shape[0]
    factor = nx_fine // nx
    assert factor * nx == nx_fine, f"SINKHORN_NX_FINE={nx_fine} must be a multiple of nx={nx}"

    ntm = nt - 1
    dt = problem.dt

    # rho_stag: interior edge times t=k*dt, k=1..ntm (matches analytical_sb_gaussian's convention)
    t_edge = (np.arange(1, ntm + 1)) * dt
    rho_stag_fine, _ = sinkhorn_eval(sink_fit, t_edge)
    rho_stag = _reduce(rho_stag_fine, factor)

    # rho_cc/mx_stag: cell-centre times t=(k-0.5)*dt, k=1..nt -- evaluated
    # directly at these times (exact spectral evaluation, no interpolation
    # needed), giving both rho and mx on the SAME time grid in one call.
    t_cc = (np.arange(1, nt + 1) - 0.5) * dt
    rho_cc_fine, mx_cc_fine = sinkhorn_eval(sink_fit, t_cc)
    rho_cc = _reduce(rho_cc_fine, factor)
    mx_stag = _coarsen_mx(mx_cc_fine, factor)

    mx_cc = ops.interp_x_at_phi(mx_stag, zeros_nt, zeros_nt)

    return rho_stag, mx_stag, rho_cc, mx_cc


def l2_error(rho, mx, rho_ana, mx_ana, dt, dx):
    return np.sqrt(dt * dx * (np.sum((rho - rho_ana) ** 2) + np.sum((mx - mx_ana) ** 2)))


def max_error(rho, mx, rho_ana, mx_ana):
    return np.max(np.abs(rho - rho_ana)) + np.max(np.abs(mx - mx_ana))


def run_study(vareps, resolutions, ladmm_cfg, verbose=True,
              precomp=precomp_expsemi_proj, projection=proj_fokker_planck_expsemi):
    """Run discretize_then_optimize once per resolution; return a dict keyed
    by n holding N, error metrics (cell-centre and staggered), and the full
    LadmmInfo (residual histories) for that resolution.

    `precomp`/`projection` select the FP projection scheme as a
    (precomp(problem, vareps), proj(x, problem, vareps, precomputed)) pair --
    the same seam discretize_then_optimize already exposes.  Defaults to the
    ETD1 scheme (projection_expsemi.py) so existing callers are unaffected;
    pass projection_expsemi_theta's pair for ETD-theta, or projection.py's for
    the CN/banded scheme.

    If vareps > SINKHORN_VAREPS_THRESHOLD, Sinkhorn is fit ONCE on a fine
    spatial grid here (not once per resolution), then coarsened per
    resolution inside reference_solution -- see module docstring.
    """
    sink_fit = None
    if vareps > SINKHORN_VAREPS_THRESHOLD:
        assert all(SINKHORN_NX_FINE % n == 0 for n in resolutions), (
            f"SINKHORN_NX_FINE={SINKHORN_NX_FINE} must be a multiple of every resolution"
        )
        fine_problem = setup_problem(prob_gaussian(), nt=SINKHORN_NX_FINE, nx=SINKHORN_NX_FINE)
        sink_fit = sinkhorn_fit(fine_problem, vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
        if verbose:
            print(
                f"Sinkhorn reference fit (nx={SINKHORN_NX_FINE}): iters={sink_fit.iters}  "
                f"converged={sink_fit.converged}  error={sink_fit.error:.2e}  "
                f"wall={sink_fit.walltime:.2f}s"
            )

    study = {}
    for n in resolutions:
        problem = setup_problem(prob_gaussian(), nt=n, nx=n)
        problem.ops = build_operators(problem)
        ops = problem.ops
        ep = precomp(problem, vareps)

        if verbose:
            print(f"Solving nt=nx={n} vareps={vareps:g} ...")
        result = discretize_then_optimize(problem, projection, ep, vareps, ladmm_cfg)
        info = result.info
        if verbose:
            print(f"  iters={info.iters}  converged={info.converged}  wall={info.walltime:.2f}s")
            print(
                f"  final residuals:  dx={info.dx[-1]:.3e}  dy={info.dy[-1]:.3e}  "
                f"dz={info.dz[-1]:.3e}  D={info.D[-1]:.3e}  P={info.P[-1]:.3e}"
            )

        rho_ref_stag, mx_ref_stag, rho_ref_cc, mx_ref_cc = reference_solution(
            problem, vareps, ops, sink_fit
        )

        e_l2_cc = l2_error(result.rho_cc, result.mx_cc, rho_ref_cc, mx_ref_cc, problem.dt, problem.dx)
        e_max_cc = max_error(result.rho_cc, result.mx_cc, rho_ref_cc, mx_ref_cc)
        e_l2_stag = l2_error(
            result.rho_stag, result.mx_stag, rho_ref_stag, mx_ref_stag, problem.dt, problem.dx
        )
        e_max_stag = max_error(result.rho_stag, result.mx_stag, rho_ref_stag, mx_ref_stag)

        if verbose:
            print(
                f"  cell-centre:  L2={e_l2_cc:.6e}  max={e_max_cc:.6e}\n"
                f"  staggered:    L2={e_l2_stag:.6e}  max={e_max_stag:.6e}"
            )

        study[n] = dict(
            N=n * n,
            l2_cc=e_l2_cc,
            max_cc=e_max_cc,
            l2_stag=e_l2_stag,
            max_stag=e_max_stag,
            info=info,
        )
    return study


def print_orders(name, resolutions, errs):
    print(f"\n{name} empirical order (h-sense; ~2 for a 2nd-order scheme):")
    for i in range(len(resolutions) - 1):
        order = np.log2(errs[i] / errs[i + 1])
        print(f"  {resolutions[i]:4d} -> {resolutions[i + 1]:4d}:  order={order:.3f}")


def fit_tail_rate(arr, k_start=None, frac=0.5):
    """Fit log(arr[k]) ~ slope*log(k) + intercept over the tail window
    k in [k_start, len(arr)] (1-indexed iteration count) -- k_start defaults
    to frac (default: last half) of the way through the run.

    Distinguishes "still converging, just slowly" from "stuck at a
    numerical floor" on a residual/iterate-change history that *looks*
    flat on a semilogy plot: plain (non-accelerated) ADMM/LADMM is only
    guaranteed O(1/k) convergence for a general convex problem, no faster,
    absent extra structure -- so a genuinely still-converging run has
    slope ~ -1 here, not slope ~ 0. A slope flattening toward 0 means the
    curve is bending toward a real floor (hit the stopping tolerance,
    hit float precision, or genuinely stalled), not continuing to shrink.

    Returns (slope, r_squared). r_squared close to 1 means the tail
    genuinely looks like a clean power law over the fitted window; well
    below 1 means the window is noisy, too short, or the curve isn't
    straight in log-log (e.g. actually bending toward a floor).
    """
    n = len(arr)
    if k_start is None:
        k_start = max(1, int(round(n * (1.0 - frac))))
    k = np.arange(k_start, n + 1)
    y = np.asarray(arr[k_start - 1:n])
    valid = y > 0  # guard against exact zeros (e.g. a converged run's tail) before log
    logk, logy = np.log(k[valid]), np.log(y[valid])
    if logk.size < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(logk, logy, 1)
    fit = slope * logk + intercept
    ss_res = np.sum((logy - fit) ** 2)
    ss_tot = np.sum((logy - logy.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(r_squared)


def print_tail_rates(info, name="", k_start=None, frac=0.5):
    """Tail power-law fit (see fit_tail_rate) for every history LadmmInfo
    tracks -- run this on a "looks plateaued" residual plot before
    concluding it's actually stuck.
    """
    header = f"Tail power-law fit{f' ({name})' if name else ''}"
    print(f"\n{header} (log-log slope over the last {frac:.0%} of {info.iters} iters):")
    for field in ["dx", "dy", "dz", "D", "P"]:
        arr = getattr(info, field)
        if len(arr) < 4:
            continue
        slope, r2 = fit_tail_rate(arr, k_start=k_start, frac=frac)
        if slope != slope:  # nan
            continue
        flag = "~O(1/k), still converging" if slope < -0.7 else "flattening -- may be near a floor"
        print(f"  {field:>3s}: slope={slope:+.3f}  R^2={r2:.4f}  ({flag})")


def plot_grid_convergence(study, resolutions, vareps, out_path):
    ns = np.array(resolutions)
    l2_cc = np.array([study[n]["l2_cc"] for n in resolutions])
    max_cc = np.array([study[n]["max_cc"] for n in resolutions])
    l2_stag = np.array([study[n]["l2_stag"] for n in resolutions])
    max_stag = np.array([study[n]["max_stag"] for n in resolutions])

    print_orders("cell-centre L2", resolutions, l2_cc)
    print_orders("cell-centre max", resolutions, max_cc)
    print_orders("staggered L2", resolutions, l2_stag)
    print_orders("staggered max", resolutions, max_stag)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (cc_err, stag_err, title) in zip(
        axes, [(l2_cc, l2_stag, "L2 error"), (max_cc, max_stag, "max error")]
    ):
        ax.loglog(ns, cc_err, "o-", label="cell-centre (y, KE variable)")
        ax.loglog(ns, stag_err, "s-", label="staggered (x, FP variable)")
        # h = 1/n: O(n^-1) = O(h) [1st order], O(n^-2) = O(h^2) [2nd order]
        ax.loglog(ns, cc_err[-1] * (ns / ns[-1]) ** -1.0, "k--", lw=1, label="O(n^-1) = O(h)")
        ax.loglog(ns, cc_err[-1] * (ns / ns[-1]) ** -2.0, "k:", lw=1, label="O(n^-2) = O(h^2)")
        ax.set_xlabel("n = nx = nt")
        ax.set_ylabel("error")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both")

    fig.suptitle(f"Grid convergence, vareps={vareps:g}")
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nfigure saved to: {out_path}")


def plot_residual_history(study, resolutions, vareps, ladmm_cfg, out_path):
    residual_names = ["dx", "dy", "dz", "D", "P"]
    titles = {
        "dx": r"$dx = \|x^{k+1}-x^k\|$",
        "dy": r"$dy = \|y^{k+1}-y^k\|$",
        "dz": r"$dz = \|\delta^{k+1}-\delta^k\|$",
        "D": r"$D = \|Ax^k+By^k-b\|$",
        "P": r"$P$ (primal residual)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))

    for ax, name in zip(axes, residual_names):
        for n, color in zip(resolutions, colors):
            arr = getattr(study[n]["info"], name)
            it = np.arange(1, len(arr) + 1)
            ax.semilogy(it, arr, color=color, label=f"n={n}")

        # O(1/k), O(1/k^2) reference decay curves, anchored to the largest
        # resolution's curve at its final iteration.
        ref_arr = getattr(study[resolutions[-1]]["info"], name)
        k_last = len(ref_arr)
        ref_val = ref_arr[-1]
        k_ref = np.arange(1, k_last + 1)
        ax.semilogy(k_ref, ref_val * (k_last / k_ref), "k--", lw=1, label="O(1/k)")
        ax.semilogy(k_ref, ref_val * (k_last / k_ref) ** 2, "k:", lw=1, label="O(1/k^2)")

        ax.set_xlabel("iteration")
        ax.set_ylabel(name)
        ax.set_title(titles[name])
        ax.legend(fontsize=8)
        ax.grid(True, which="both")

    axes[-1].axis("off")  # 6th subplot slot unused (5 residuals)

    fig.suptitle(
        f"LADMM residual histories vs resolution  vareps={vareps:g}  "
        f"gamma={ladmm_cfg.gamma:g}  tau={ladmm_cfg.tau:g}"
    )
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"figure saved to: {out_path}")
