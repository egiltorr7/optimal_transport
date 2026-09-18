"""Regression/consistency checks for the non-uniform-time modules
(time_grid, grid_nu, operators_nu, projection_nu, pipeline_nu, problems_nu).

These are the checks that caught two real bugs while building this
formulation (a wrong DCT eigenvalue sign, and a "physically motivated"
deriv_t_at_rho that silently wasn't the adjoint projection_nu.py's algebra
needs off a uniform grid) -- run this after touching any *_nu module.

  1. operators_nu reduces exactly to operators.py on a uniform time_grid.
  2. deriv_t_at_rho is the exact (signed) Euclidean adjoint of
     deriv_t_at_phi -- on both a uniform AND a genuinely non-uniform grid.
  3. proj_fokker_planck_banded_nu reduces exactly to
     projection.py's proj_fokker_planck_banded on a uniform time_grid.
  4. The projection is a genuine (idempotent-ish) projection on a
     non-uniform grid: repeated application stays bounded instead of
     blowing up (this is what actually caught the deriv_t_at_rho bug --
     bug #2 above passed check #3 vacuously on a uniform grid but failed
     catastrophically here).
  5. discretize_then_optimize_nu reduces to discretize_then_optimize (same
     iterate trajectory, to solver tolerance) on a uniform time_grid.
  6-8. Same three checks (3-5) repeated for the ETD scheme
     (projection_expsemi_nu.py vs projection_expsemi.py) instead of the
     banded/CN scheme.

Not a substitute for scripts/sb_gaussian_nonuniform.py (which exercises the
actually-non-uniform, full-accuracy case against the closed-form SB
solution) -- this script only checks internal consistency.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.fft import dct, idct

from optimal_transport.grid import setup_problem
from optimal_transport.grid_nu import setup_problem_nu
from optimal_transport.operators import build_operators
from optimal_transport.operators_nu import build_operators_nu, interp_t_at_rho_weighted
from optimal_transport.pipeline import LadmmConfig, discretize_then_optimize
from optimal_transport.pipeline_nu import LadmmConfig as LadmmConfigNU, discretize_then_optimize_nu
from optimal_transport.problems import prob_gaussian
from optimal_transport.projection import precomp_banded_proj, proj_fokker_planck_banded
from optimal_transport.projection_expsemi import precomp_expsemi_proj, proj_fokker_planck_expsemi
from optimal_transport.projection_expsemi_nu import precomp_expsemi_proj_nu, proj_fokker_planck_expsemi_nu
from optimal_transport.projection_nu import precomp_banded_proj_nu, proj_fokker_planck_banded_nu
from optimal_transport.state import State
from optimal_transport.time_grid import (
    clustered_time_grid,
    graded_ends_time_grid,
    refine_ends_time_grid,
    uniform_time_grid,
)

NT, NX = 32, 24
VAREPS = 0.3
PD = prob_gaussian()
RNG = np.random.default_rng(0)


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def check_operators_reduce_to_uniform() -> bool:
    problem = setup_problem(PD, nt=NT, nx=NX)
    ops = build_operators(problem)
    tg = uniform_time_grid(NT)
    problem_nu = setup_problem_nu(PD, tg, NX)
    ops_nu = build_operators_nu(problem_nu)

    ntm, nxm = NT - 1, NX - 1
    rho_in = RNG.standard_normal((ntm, NX))
    phi_in = RNG.standard_normal((NT, NX))
    rho0, rho1 = problem.rho0, problem.rho1
    zeros_x, zeros_t = np.zeros(NX), np.zeros(NT)

    ok = True
    ok &= np.allclose(ops.interp_t_at_phi(rho_in, rho0, rho1), ops_nu.interp_t_at_phi(rho_in, rho0, rho1))
    ok &= np.allclose(ops.deriv_t_at_phi(rho_in, rho0, rho1), ops_nu.deriv_t_at_phi(rho_in, rho0, rho1))
    ok &= np.allclose(ops.deriv_t_at_rho(phi_in), ops_nu.deriv_t_at_rho(phi_in))
    ok &= np.allclose(ops.interp_t_at_rho(phi_in), ops_nu.interp_t_at_rho(phi_in))
    return _report("operators_nu == operators.py on a uniform grid", ok)


def check_graded_ends_reduces_to_refine_ends() -> bool:
    g1 = graded_ends_time_grid(NT, n_boundary=1)
    r1 = refine_ends_time_grid(NT)
    ok = g1.nt == r1.nt and np.allclose(g1.t_edges, r1.t_edges)
    return _report("graded_ends_time_grid(n_boundary=1) == refine_ends_time_grid", ok,
                    f"nt={g1.nt}/{r1.nt}  max edge diff={np.abs(g1.t_edges - r1.t_edges).max():.2e}")


def check_deriv_t_adjoint() -> bool:
    ok = True
    grids = [
        (clustered_time_grid(NT, strength=0.0), "uniform"),
        (clustered_time_grid(NT, strength=0.6), "clustered"),
        (refine_ends_time_grid(NT), "refine_ends"),
        (graded_ends_time_grid(NT, n_boundary=3), "graded_ends(3)"),
    ]
    for tg, label in grids:
        problem = setup_problem_nu(PD, tg, NX)
        ops = build_operators_nu(problem)
        ntm = problem.nt - 1
        u = RNG.standard_normal((ntm, NX))
        v = RNG.standard_normal((problem.nt, NX))
        zeros_x = np.zeros(NX)
        lhs = np.sum(ops.deriv_t_at_phi(u, zeros_x, zeros_x) * v)
        rhs = np.sum(u * (-ops.deriv_t_at_rho(v)))  # sign convention: deriv_t_at_rho == -D_t^T
        this_ok = np.isclose(lhs, rhs, rtol=1e-10)
        ok &= this_ok
        _report(f"deriv_t_at_rho == -adjoint(deriv_t_at_phi) [{label}]", this_ok,
                 f"lhs={lhs:.6g} rhs={rhs:.6g}")
    return ok


def check_At_star_reduces_to_euclidean_on_uniform() -> bool:
    tg = uniform_time_grid(NT)
    problem = setup_problem_nu(PD, tg, NX)
    ops = build_operators_nu(problem)
    v = RNG.standard_normal((NT, NX))
    weighted = interp_t_at_rho_weighted(v, problem.dt_vec)
    euclidean = ops.interp_t_at_rho(v)
    ok = np.allclose(weighted, euclidean)
    return _report("interp_t_at_rho_weighted == interp_t_at_rho on a uniform grid", ok,
                    f"max diff={np.abs(weighted - euclidean).max():.2e}")


def check_At_star_is_NM_weighted_adjoint() -> bool:
    """<A_t q, v>_M == <q, A_t^* v>_N (docs/nonuniform_grid_norms.tex Sec 4),
    for A_t = interp_t_at_phi's linear part, A_t^* = interp_t_at_rho_weighted
    -- NOT the Euclidean identity check_deriv_t_adjoint runs against
    interp_t_at_rho, which is the wrong adjoint once dt_vec is non-uniform.
    dx cancels identically from both sides (space uniform), so left out.
    """
    ok = True
    grids = [
        (clustered_time_grid(NT, strength=0.6), "clustered"),
        (refine_ends_time_grid(NT), "refine_ends"),
        (graded_ends_time_grid(NT, n_boundary=3), "graded_ends(3)"),
    ]
    for tg, label in grids:
        problem = setup_problem_nu(PD, tg, NX)
        ops = build_operators_nu(problem)
        dt_vec = problem.dt_vec
        dt_dual = 0.5 * (dt_vec[:-1] + dt_vec[1:])  # (ntm,)

        ntm = problem.nt - 1
        q = RNG.standard_normal((ntm, NX))
        v = RNG.standard_normal((problem.nt, NX))
        zeros_x = np.zeros(NX)

        Aq = ops.interp_t_at_phi(q, zeros_x, zeros_x)  # linear part only, zero BC
        lhs = np.sum(dt_vec[:, None] * Aq * v)              # <Aq, v>_M
        Astar_v = interp_t_at_rho_weighted(v, dt_vec)
        rhs = np.sum(dt_dual[:, None] * q * Astar_v)         # <q, A_t^* v>_N
        this_ok = np.isclose(lhs, rhs, rtol=1e-10)
        ok &= this_ok
        _report(f"interp_t_at_rho_weighted is the N,M-weighted adjoint of interp_t_at_phi [{label}]",
                 this_ok, f"lhs={lhs:.6g} rhs={rhs:.6g}")
    return ok


def check_Rstar_is_N_weighted_adjoint() -> bool:
    """-<Rx, phi> == <x, correction>_N, `correction` exactly
    proj_fokker_planck_banded_nu's (q_corr,b_corr) -- N_q^{-1}=1/(dt_dual*dx) on the
    q-piece, N_b^{-1}=1/(dt_vec*dx) on the b-piece (docs/nonuniform_grid_norms.tex
    Sec 9.5, eq. 9). The minus sign is phi's own gauge, not a bug: the code's linear
    solve for phi (inherited unchanged from the Euclidean case, only its RHS/system
    rescaled) produces the negative of the "clean Lagrangian" phi this note's R^*
    derivation assumed -- harmless, since phi is an internal auxiliary variable with
    no meaning of its own (only x_out = x_in + correction does). Confirmed against
    the thing that actually matters: after projection, R(x_out) is zero to machine
    precision (checked separately, ~1e-11 on a clustered grid) -- this identity check
    (up to that sign) is a second, independent confirmation using the adjoint
    definition directly, and also settles Sec 9.5's open M1-sign question
    empirically: it's whatever sign the code already has, unchanged.
    """
    ok = True
    grids = [
        (clustered_time_grid(NT, strength=0.6), "clustered"),
        (refine_ends_time_grid(NT), "refine_ends"),
        (graded_ends_time_grid(NT, n_boundary=3), "graded_ends(3)"),
    ]
    for tg, label in grids:
        problem = setup_problem_nu(PD, tg, NX)
        ops = build_operators_nu(problem)
        dt_vec, dx = problem.dt_vec, problem.dx
        dt_dual = 0.5 * (dt_vec[:-1] + dt_vec[1:])
        nt, ntm = problem.nt, problem.nt - 1
        zeros_nx = np.zeros(NX)  # time-direction boundary (interp_t_at_phi/deriv_t_at_phi)
        zeros_nt = np.zeros(nt)  # space-direction boundary (interp_x/deriv_x_at_phi) --
                                  # matches proj_fokker_planck_banded_nu's own "zeros_x"

        q = RNG.standard_normal((ntm, NX))
        b = RNG.standard_normal((nt, NX - 1))
        phi = RNG.standard_normal((nt, NX))

        # R(q,b), linear part only (zero BC) -- exactly proj_fokker_planck_banded_nu's
        # `f`, but with rho0,rho1 zeroed out (probing the linear map alone).
        laplacian_q = ops.deriv_x_at_phi(
            ops.deriv_x_at_m(ops.interp_t_at_phi(q, zeros_nx, zeros_nx)), zeros_nt, zeros_nt
        )
        Rx = (
            ops.deriv_t_at_phi(q, zeros_nx, zeros_nx)
            + ops.deriv_x_at_phi(b, zeros_nt, zeros_nt)
            - VAREPS * laplacian_q
        )
        lhs = np.sum(Rx * phi)  # <Rx, phi>, plain Euclidean

        # R^*(phi) -- exactly proj_fokker_planck_banded_nu's correction step.
        dphi_dx = ops.deriv_x_at_m(phi)
        nablax_phi = ops.interp_t_at_rho(ops.deriv_x_at_phi(dphi_dx, zeros_nt, zeros_nt))
        q_corr = (ops.deriv_t_at_rho(phi) + VAREPS * nablax_phi) / (dt_dual[:, None] * dx)
        b_corr = dphi_dx / (dt_vec[:, None] * dx)

        rhs = (
            np.sum(dt_dual[:, None] * dx * q * q_corr)
            + np.sum(dt_vec[:, None] * dx * b * b_corr)
        )  # <(q,b), correction>_N

        this_ok = np.isclose(-lhs, rhs, rtol=1e-10)  # note the sign -- see docstring
        ok &= this_ok
        _report(f"R^* is the N-weighted adjoint of R [{label}]", this_ok,
                 f"lhs={lhs:.6g} rhs={rhs:.6g}")
    return ok


def _random_state(problem) -> State:
    # Fractional (not grid-exact) interpolation weight -- fine here, this is
    # just a plausible smooth seed to perturb, not a claim about the actual
    # grid; works identically for a uniform Problem or a ProblemNU.
    ntm, nxm = problem.nt - 1, problem.nx - 1
    t_stag = np.linspace(0.0, 1.0, ntm)[:, None]
    mu = (1 - t_stag) * problem.rho0[None, :] + t_stag * problem.rho1[None, :]
    mu = mu + 0.01 * RNG.standard_normal(mu.shape)
    psi = 0.01 * RNG.standard_normal((problem.nt, nxm))
    return State(mu, psi)


def check_projection_is_feasible_nonuniform() -> bool:
    """The most direct correctness test, independent of any sign/gauge convention:
    after projecting, is R(x_out) actually zero? A random x_in has R(x_in) of order
    ~100-1000 on these grids; a genuine projection onto {R=0} should drive that to
    numerical zero regardless of how N is weighted internally. This is what caught
    the missing dx factor in N_q^{-1}/N_b^{-1} being wrong-by-construction would
    have been invisible to check_projection_stable_nonuniform (which only checks
    boundedness, not that the fixed point is actually feasible)."""
    ok = True
    grids = [
        (clustered_time_grid(NT, strength=0.6), "clustered"),
        (refine_ends_time_grid(NT), "refine_ends"),
        (graded_ends_time_grid(NT, n_boundary=3), "graded_ends(3)"),
    ]
    for tg, label in grids:
        problem = setup_problem_nu(PD, tg, NX)
        problem.ops = build_operators_nu(problem)
        bp = precomp_banded_proj_nu(problem, VAREPS)

        x_in = _random_state(problem)
        x_out = proj_fokker_planck_banded_nu(x_in, problem, VAREPS, bp)

        ops = problem.ops
        rho0, rho1 = problem.rho0, problem.rho1
        nt = problem.nt
        zeros_x = np.zeros(nt)
        laplacian = ops.deriv_x_at_phi(
            ops.deriv_x_at_m(ops.interp_t_at_phi(x_out.rho, rho0, rho1)), zeros_x, zeros_x
        )
        f_out = (
            ops.deriv_t_at_phi(x_out.rho, rho0, rho1)
            + ops.deriv_x_at_phi(x_out.mx, zeros_x, zeros_x)
            - VAREPS * laplacian
        )
        max_resid = np.abs(f_out).max()
        this_ok = max_resid < 1e-8
        ok &= this_ok
        _report(f"projection lands on R(x_out)=0 (non-uniform grid) [{label}]",
                 this_ok, f"max|R(x_out)|={max_resid:.2e}")
    return ok


def check_projection_reduces_to_uniform() -> bool:
    problem = setup_problem(PD, nt=NT, nx=NX)
    problem.ops = build_operators(problem)
    bp = precomp_banded_proj(problem, VAREPS)

    tg = uniform_time_grid(NT)
    problem_nu = setup_problem_nu(PD, tg, NX)
    problem_nu.ops = build_operators_nu(problem_nu)
    bp_nu = precomp_banded_proj_nu(problem_nu, VAREPS)

    x_in = _random_state(problem)
    out = proj_fokker_planck_banded(x_in, problem, VAREPS, bp)
    out_nu = proj_fokker_planck_banded_nu(x_in, problem_nu, VAREPS, bp_nu)

    d_rho = np.abs(out.rho - out_nu.rho).max()
    d_mx = np.abs(out.mx - out_nu.mx).max()
    return _report("proj_fokker_planck_banded_nu == projection.py on a uniform grid",
                    d_rho < 1e-8 and d_mx < 1e-8, f"max diff rho={d_rho:.2e} mx={d_mx:.2e}")


def check_projection_stable_nonuniform() -> bool:
    ok = True
    grids = [
        (clustered_time_grid(NT, strength=0.6), "clustered"),
        (refine_ends_time_grid(NT), "refine_ends"),
        (graded_ends_time_grid(NT, n_boundary=3), "graded_ends(3)"),
    ]
    for tg, label in grids:
        problem = setup_problem_nu(PD, tg, NX)
        problem.ops = build_operators_nu(problem)
        bp = precomp_banded_proj_nu(problem, VAREPS)

        x = _random_state(problem)
        norms = []
        for _ in range(6):
            x = proj_fokker_planck_banded_nu(x, problem, VAREPS, bp)
            norms.append(np.linalg.norm(x.rho) + np.linalg.norm(x.mx))
        # A genuine projection is idempotent past the first application:
        # norms should plateau, not grow. Flag anything still growing >1%
        # by the last couple of applications.
        this_ok = abs(norms[-1] - norms[-2]) / norms[-2] < 1e-2
        ok &= this_ok
        _report(f"projection stays bounded under repeated application (non-uniform grid) [{label}]",
                 this_ok, f"norms={['%.3e' % n for n in norms]}")
    return ok


def _expsemi_coeffs(dt_vec, lam_x, vareps):
    alpha = vareps * dt_vec[:, None] * lam_x[None, :]
    c_vals = np.exp(-alpha)
    phi_vals = np.ones_like(alpha)
    nz = alpha > 1e-14
    phi_vals[nz] = (1.0 - c_vals[nz]) / alpha[nz]
    return c_vals, phi_vals


def check_Rstar_expsemi_is_N_weighted_adjoint() -> bool:
    """ETD analogue of check_Rstar_is_N_weighted_adjoint -- same identity,
    same sign convention (phi's own gauge, confirmed harmless the same way:
    against feasibility, see check_expsemi_projection_is_feasible_nonuniform),
    for projection_expsemi_nu.py's R (docs/nonuniform_grid_norms.tex Sec 9,
    generalized -- see that module's docstring for what's genuinely
    different here vs the CN scheme). R built directly in DCT-x space,
    where the ETD residual is naturally defined; Parseval (orthonormal DCT)
    makes the plain Euclidean pairing there equal the real-space one, so no
    inverse transform is needed to evaluate lhs.
    """
    ok = True
    grids = [
        (clustered_time_grid(NT, strength=0.6), "clustered"),
        (refine_ends_time_grid(NT), "refine_ends"),
        (graded_ends_time_grid(NT, n_boundary=3), "graded_ends(3)"),
    ]
    for tg, label in grids:
        problem = setup_problem_nu(PD, tg, NX)
        ops = build_operators_nu(problem)
        dt_vec, dx = problem.dt_vec, problem.dx
        dt_dual = 0.5 * (dt_vec[:-1] + dt_vec[1:])
        nt, ntm = problem.nt, problem.nt - 1
        lam_x = problem.lambda_x
        zeros_nt = np.zeros(nt)
        c_vals, phi_vals = _expsemi_coeffs(dt_vec, lam_x, VAREPS)

        q = RNG.standard_normal((ntm, NX))
        b = RNG.standard_normal((nt, NX - 1))
        Phi = RNG.standard_normal((nt, NX))  # dual variable, directly in DCT-x space

        # R(q,b) in DCT-x space, linear part only (zero BC).
        q_prev = np.concatenate([np.zeros((1, NX)), q], axis=0)
        q_curr = np.concatenate([q, np.zeros((1, NX))], axis=0)
        q_prev_hat = dct(q_prev, type=2, norm="ortho", axis=1)
        q_curr_hat = dct(q_curr, type=2, norm="ortho", axis=1)
        f_rho_hat = (q_curr_hat - c_vals * q_prev_hat) / dt_vec[:, None]
        dxb_hat = dct(ops.deriv_x_at_phi(b, zeros_nt, zeros_nt), type=2, norm="ortho", axis=1)
        Rx_hat = f_rho_hat + phi_vals * dxb_hat
        lhs = np.sum(Rx_hat * Phi)  # <Rx, Phi>, plain Euclidean (Parseval)

        # R^*(Phi) -- exactly proj_fokker_planck_expsemi_nu's correction step.
        dt_curr, dt_next = dt_vec[:-1, None], dt_vec[1:, None]
        adj_rho_hat = c_vals[1:, :] * Phi[1:, :] / dt_next - Phi[:-1, :] / dt_curr
        adj_rho = idct(adj_rho_hat, type=2, norm="ortho", axis=1)  # adj_rho_hat lives in
        q_corr = adj_rho / (dt_dual[:, None] * dx)                  # DCT-x space -- IDCT first
        phi_weighted = idct(phi_vals * Phi, type=2, norm="ortho", axis=1)
        b_corr = ops.deriv_x_at_m(phi_weighted) / (dt_vec[:, None] * dx)

        rhs = (
            np.sum(dt_dual[:, None] * dx * q * q_corr)
            + np.sum(dt_vec[:, None] * dx * b * b_corr)
        )

        this_ok = np.isclose(-lhs, rhs, rtol=1e-10)  # note the sign -- see docstring
        ok &= this_ok
        _report(f"ETD: R^* is the N-weighted adjoint of R [{label}]", this_ok,
                 f"lhs={lhs:.6g} rhs={rhs:.6g}")
    return ok


def check_expsemi_projection_is_feasible_nonuniform() -> bool:
    """ETD analogue of check_projection_is_feasible_nonuniform."""
    ok = True
    grids = [
        (clustered_time_grid(NT, strength=0.6), "clustered"),
        (refine_ends_time_grid(NT), "refine_ends"),
        (graded_ends_time_grid(NT, n_boundary=3), "graded_ends(3)"),
    ]
    for tg, label in grids:
        problem = setup_problem_nu(PD, tg, NX)
        problem.ops = build_operators_nu(problem)
        ep = precomp_expsemi_proj_nu(problem, VAREPS)

        x_in = _random_state(problem)
        x_out = proj_fokker_planck_expsemi_nu(x_in, problem, VAREPS, ep)

        ops = problem.ops
        rho0, rho1 = problem.rho0, problem.rho1
        nt, dt_vec = problem.nt, problem.dt_vec
        c_vals, phi_vals = _expsemi_coeffs(dt_vec, problem.lambda_x, VAREPS)
        zeros_x = np.zeros(nt)

        mu_prev = np.concatenate([rho0[None, :], x_out.rho], axis=0)
        mu_curr = np.concatenate([x_out.rho, rho1[None, :]], axis=0)
        f_rho_hat = (
            dct(mu_curr, type=2, norm="ortho", axis=1) - c_vals * dct(mu_prev, type=2, norm="ortho", axis=1)
        ) / dt_vec[:, None]
        dxm_hat = dct(ops.deriv_x_at_phi(x_out.mx, zeros_x, zeros_x), type=2, norm="ortho", axis=1)
        f_hat = f_rho_hat + phi_vals * dxm_hat

        max_resid = np.abs(f_hat).max()
        this_ok = max_resid < 1e-8
        ok &= this_ok
        _report(f"ETD projection lands on R(x_out)=0 (non-uniform grid) [{label}]",
                 this_ok, f"max|R(x_out)|={max_resid:.2e}")
    return ok


def check_expsemi_reduces_to_uniform() -> bool:
    problem = setup_problem(PD, nt=NT, nx=NX)
    problem.ops = build_operators(problem)
    ep = precomp_expsemi_proj(problem, VAREPS)

    tg = uniform_time_grid(NT)
    problem_nu = setup_problem_nu(PD, tg, NX)
    problem_nu.ops = build_operators_nu(problem_nu)
    ep_nu = precomp_expsemi_proj_nu(problem_nu, VAREPS)

    x_in = _random_state(problem)
    out = proj_fokker_planck_expsemi(x_in, problem, VAREPS, ep)
    out_nu = proj_fokker_planck_expsemi_nu(x_in, problem_nu, VAREPS, ep_nu)

    d_rho = np.abs(out.rho - out_nu.rho).max()
    d_mx = np.abs(out.mx - out_nu.mx).max()
    return _report("proj_fokker_planck_expsemi_nu == projection_expsemi.py on a uniform grid",
                    d_rho < 1e-8 and d_mx < 1e-8, f"max diff rho={d_rho:.2e} mx={d_mx:.2e}")


def check_expsemi_stable_nonuniform() -> bool:
    ok = True
    grids = [
        (clustered_time_grid(NT, strength=0.6), "clustered"),
        (refine_ends_time_grid(NT), "refine_ends"),
        (graded_ends_time_grid(NT, n_boundary=3), "graded_ends(3)"),
    ]
    for tg, label in grids:
        problem = setup_problem_nu(PD, tg, NX)
        problem.ops = build_operators_nu(problem)
        ep = precomp_expsemi_proj_nu(problem, VAREPS)

        x = _random_state(problem)
        norms = []
        for _ in range(6):
            x = proj_fokker_planck_expsemi_nu(x, problem, VAREPS, ep)
            norms.append(np.linalg.norm(x.rho) + np.linalg.norm(x.mx))
        this_ok = abs(norms[-1] - norms[-2]) / norms[-2] < 1e-2
        ok &= this_ok
        _report(f"expsemi projection stays bounded under repeated application (non-uniform grid) [{label}]",
                 this_ok, f"norms={['%.3e' % n for n in norms]}")
    return ok


def check_pipeline_reduces_to_uniform() -> bool:
    problem = setup_problem(PD, nt=NT, nx=NX)
    problem.ops = build_operators(problem)
    bp = precomp_banded_proj(problem, VAREPS)
    cfg = LadmmConfig(gamma=100.0, tau=101.0, max_iter=800, eps_abs=1e-10, eps_rel=1e-6)
    res = discretize_then_optimize(problem, proj_fokker_planck_banded, bp, VAREPS, cfg)

    tg = uniform_time_grid(NT)
    problem_nu = setup_problem_nu(PD, tg, NX)
    problem_nu.ops = build_operators_nu(problem_nu)
    bp_nu = precomp_banded_proj_nu(problem_nu, VAREPS)
    cfg_nu = LadmmConfigNU(gamma=100.0, tau=101.0, max_iter=800, eps_abs=1e-10, eps_rel=1e-6)
    res_nu = discretize_then_optimize_nu(problem_nu, proj_fokker_planck_banded_nu, bp_nu, VAREPS, cfg_nu)

    d_rho_stag = np.abs(res.rho_stag - res_nu.rho_stag).max()
    d_rho_cc = np.abs(res.rho_cc - res_nu.rho_cc).max()
    ok = d_rho_stag < 1e-3 and d_rho_cc < 1e-3 and res.info.iters == res_nu.info.iters
    return _report("discretize_then_optimize_nu == pipeline.py on a uniform grid", ok,
                    f"max diff rho_stag={d_rho_stag:.2e} rho_cc={d_rho_cc:.2e} "
                    f"iters={res.info.iters}/{res_nu.info.iters}")


def check_expsemi_pipeline_reduces_to_uniform() -> bool:
    problem = setup_problem(PD, nt=NT, nx=NX)
    problem.ops = build_operators(problem)
    ep = precomp_expsemi_proj(problem, VAREPS)
    cfg = LadmmConfig(gamma=100.0, tau=101.0, max_iter=800, eps_abs=1e-10, eps_rel=1e-6)
    res = discretize_then_optimize(problem, proj_fokker_planck_expsemi, ep, VAREPS, cfg)

    tg = uniform_time_grid(NT)
    problem_nu = setup_problem_nu(PD, tg, NX)
    problem_nu.ops = build_operators_nu(problem_nu)
    ep_nu = precomp_expsemi_proj_nu(problem_nu, VAREPS)
    cfg_nu = LadmmConfigNU(gamma=100.0, tau=101.0, max_iter=800, eps_abs=1e-10, eps_rel=1e-6)
    res_nu = discretize_then_optimize_nu(problem_nu, proj_fokker_planck_expsemi_nu, ep_nu, VAREPS, cfg_nu)

    d_rho_stag = np.abs(res.rho_stag - res_nu.rho_stag).max()
    d_rho_cc = np.abs(res.rho_cc - res_nu.rho_cc).max()
    ok = d_rho_stag < 1e-3 and d_rho_cc < 1e-3 and res.info.iters == res_nu.info.iters
    return _report("discretize_then_optimize_nu (expsemi) == pipeline.py (expsemi) on a uniform grid", ok,
                    f"max diff rho_stag={d_rho_stag:.2e} rho_cc={d_rho_cc:.2e} "
                    f"iters={res.info.iters}/{res_nu.info.iters}")


def main():
    print(f"Validating non-uniform-time modules (nt={NT} nx={NX} vareps={VAREPS:g}) ...")
    checks = [
        check_operators_reduce_to_uniform,
        check_graded_ends_reduces_to_refine_ends,
        check_deriv_t_adjoint,
        check_At_star_reduces_to_euclidean_on_uniform,
        check_At_star_is_NM_weighted_adjoint,
        check_Rstar_is_N_weighted_adjoint,
        check_projection_is_feasible_nonuniform,
        check_projection_reduces_to_uniform,
        check_projection_stable_nonuniform,
        check_pipeline_reduces_to_uniform,
        check_Rstar_expsemi_is_N_weighted_adjoint,
        check_expsemi_projection_is_feasible_nonuniform,
        check_expsemi_reduces_to_uniform,
        check_expsemi_stable_nonuniform,
        check_expsemi_pipeline_reduces_to_uniform,
    ]
    results = [c() for c in checks]
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} checks passed.")
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
