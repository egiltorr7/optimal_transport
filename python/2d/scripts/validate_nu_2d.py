"""Regression/consistency checks for the non-uniform-time 2D modules
(time_grid, grid_nu_2d, operators_nu_2d, projection_nu_2d, pipeline_nu_2d,
problems_nu_2d).

2D analogue of the 1D scripts/validate_nu.py, and the same idea: every check
is either "reduces exactly to something already trusted" or "satisfies an
identity the algebra depends on". Run after touching any *_nu_2d module.

  1. operators_nu_2d reduces exactly to operators_2d on a uniform time grid.
  2. deriv_t_at_rho is the exact (signed) Euclidean adjoint of
     deriv_t_at_phi -- on a uniform AND a genuinely non-uniform grid. This
     is the identity projection_nu_2d.py's assembled T silently depends on.
  3. R* (as projection_nu_2d applies it) is R's genuine N-weighted adjoint,
     on a clustered grid.
  4. proj_fokker_planck_banded_nu_2d reduces exactly to projection_2d's
     proj_fokker_planck_banded_2d on a uniform time grid.
  5. The projection is genuinely feasible on a non-uniform grid: the FP
     residual of its output is ~0 (this is the check that would catch a
     wrong N weight, which check 4 is blind to -- on a uniform grid every
     weight is the same constant and cancels).
  6. The projection is idempotent on a non-uniform grid: projecting twice
     changes nothing.
  7. discretize_then_optimize_nu_2d reduces to discretize_then_optimize_2d
     (same iterate trajectory) on a uniform time grid, from a shared x0/y0
     -- see that check for why the two modules' own defaults differ.

Not a substitute for sb_gaussian_nonuniform_2d.py, which exercises the
actually-non-uniform case against the closed-form SB solution -- this script
only checks internal consistency.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import jax_config

jax_config.configure(device="auto", x64=True)

import jax.numpy as jnp
import numpy as np

from setup.grid_2d import setup_problem_2d
from setup.grid_nu_2d import setup_problem_nu_2d
from setup.operators_2d import build_operators_2d
from setup.operators_nu_2d import build_operators_nu_2d
from setup.pipeline_2d import LadmmConfig, discretize_then_optimize_2d
from setup.pipeline_nu_2d import LadmmConfig as LadmmConfigNU
from setup.pipeline_nu_2d import discretize_then_optimize_nu_2d
from setup.problems_2d import prob_gaussian_2d
from setup.projection_2d import precomp_banded_proj_2d, proj_fokker_planck_banded_2d
from setup.projection_nu_2d import precomp_banded_proj_nu_2d, proj_fokker_planck_banded_nu_2d
from setup.state_2d import State2D
from setup.time_grid import clustered_time_grid, uniform_time_grid

NT, NX, NY = 12, 8, 8
VAREPS = 0.05
TOL = 1e-10


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def _rel(a, b) -> float:
    """Relative difference in the Frobenius norm, safe at a == b == 0."""
    den = max(float(jnp.linalg.norm(jnp.ravel(b))), 1e-300)
    return float(jnp.linalg.norm(jnp.ravel(a) - jnp.ravel(b))) / den


def _uniform_problem():
    p = setup_problem_2d(prob_gaussian_2d(), nt=NT, nx=NX, ny=NY)
    p.ops = build_operators_2d(p)
    return p


def _nu_problem(time_grid):
    p = setup_problem_nu_2d(prob_gaussian_2d(), time_grid, nx=NX, ny=NY)
    p.ops = build_operators_nu_2d(p)
    return p


def _rand(rng, *shape):
    return jnp.asarray(rng.standard_normal(shape))


def _rand_state(rng, nt, nx, ny) -> State2D:
    return State2D(
        rho=_rand(rng, nt - 1, nx, ny),
        mx=_rand(rng, nt, nx - 1, ny),
        my=_rand(rng, nt, nx, ny - 1),
    )


# ---------------------------------------------------------------- 1
def check_operators_reduce_to_uniform() -> bool:
    rng = np.random.default_rng(0)
    pu, pn = _uniform_problem(), _nu_problem(uniform_time_grid(NT))
    ou, on = pu.ops, pn.ops

    rho_s = _rand(rng, NT - 1, NX, NY)   # staggered rho (interior time nodes)
    cc = _rand(rng, NT, NX, NY)          # cell-centre field (phi's grid)
    mx_s = _rand(rng, NT, NX - 1, NY)
    my_s = _rand(rng, NT, NX, NY - 1)
    bc_t0, bc_t1 = _rand(rng, NX, NY), _rand(rng, NX, NY)
    bc_x0, bc_x1 = _rand(rng, NT, NY), _rand(rng, NT, NY)
    bc_y0, bc_y1 = _rand(rng, NT, NX), _rand(rng, NT, NX)

    cases = {
        "interp_t_at_phi": (lambda o: o.interp_t_at_phi(rho_s, bc_t0, bc_t1)),
        "interp_t_at_rho": (lambda o: o.interp_t_at_rho(cc)),
        "interp_x_at_phi": (lambda o: o.interp_x_at_phi(mx_s, bc_x0, bc_x1)),
        "interp_x_at_m": (lambda o: o.interp_x_at_m(cc)),
        "interp_y_at_phi": (lambda o: o.interp_y_at_phi(my_s, bc_y0, bc_y1)),
        "interp_y_at_m": (lambda o: o.interp_y_at_m(cc)),
        "deriv_t_at_phi": (lambda o: o.deriv_t_at_phi(rho_s, bc_t0, bc_t1)),
        "deriv_t_at_rho": (lambda o: o.deriv_t_at_rho(cc)),
        "deriv_x_at_phi": (lambda o: o.deriv_x_at_phi(mx_s, bc_x0, bc_x1)),
        "deriv_x_at_m": (lambda o: o.deriv_x_at_m(cc)),
        "deriv_y_at_phi": (lambda o: o.deriv_y_at_phi(my_s, bc_y0, bc_y1)),
        "deriv_y_at_m": (lambda o: o.deriv_y_at_m(cc)),
        "interp_t_at_rho_adj": (lambda o: o.interp_t_at_rho_adj(cc)),
        "interp_x_at_m_adj": (lambda o: o.interp_x_at_m_adj(cc)),
        "interp_y_at_m_adj": (lambda o: o.interp_y_at_m_adj(cc)),
    }
    worst_name, worst = "", 0.0
    for name, fn in cases.items():
        d = _rel(fn(on), fn(ou))
        if d > worst:
            worst_name, worst = name, d
    return _report("operators_nu_2d reduces to operators_2d (uniform grid)",
                   worst < TOL, f"worst: {worst_name} rel={worst:.2e}")


# ---------------------------------------------------------------- 2
def check_deriv_t_adjoint() -> bool:
    """<deriv_t_at_phi(q, 0, 0), v> == -<q, deriv_t_at_rho(v)>.

    The sign is the point: deriv_t_at_rho is MINUS the Euclidean transpose
    (see operators_nu_2d.py's bidiagonal derivation), and projection_nu_2d's
    assembled T carries a matching outer negation, so the two cancel.
    """
    rng = np.random.default_rng(1)
    ok = True
    details = []
    for label, tg in (("uniform", uniform_time_grid(NT)),
                      ("clustered", clustered_time_grid(NT, strength=1.0))):
        ops = _nu_problem(tg).ops
        q = _rand(rng, NT - 1, NX, NY)
        v = _rand(rng, NT, NX, NY)
        zt = jnp.zeros((NX, NY))
        lhs = float(jnp.sum(ops.deriv_t_at_phi(q, zt, zt) * v))
        rhs = -float(jnp.sum(q * ops.deriv_t_at_rho(v)))
        d = abs(lhs - rhs) / max(abs(rhs), 1e-300)
        ok = ok and d < TOL
        details.append(f"{label} rel={d:.2e}")
    return _report("deriv_t_at_rho is the signed Euclidean adjoint of deriv_t_at_phi",
                   ok, ", ".join(details))


# ---------------------------------------------------------------- 3
def check_Rstar_is_N_weighted_adjoint() -> bool:
    """<R_lin(x), phi>_euclidean == -<x, R*(phi)>_N on a clustered grid.

    R* here is literally the correction projection_nu_2d applies (its last
    three lines), and the N weights are the ones it divides by. The minus
    is the same outer negation as check 2. If either weight were wrong --
    N_q vs N_b swapped, or a missing dy -- this identity breaks while the
    uniform-grid reduction (check 4) still passes.
    """
    rng = np.random.default_rng(2)
    p = _nu_problem(clustered_time_grid(NT, strength=1.0))
    ops = p.ops
    dt_vec, dt_dual, ca = p.dt_vec, p.dt_dual, p.cell_area
    zx, zy, zt = jnp.zeros((NT, NY)), jnp.zeros((NT, NX)), jnp.zeros((NX, NY))

    x = _rand_state(rng, NT, NX, NY)
    phi = _rand(rng, NT, NX, NY)

    # R at zero BC (the linear part).
    lap = (ops.deriv_x_at_phi(ops.deriv_x_at_m(ops.interp_t_at_phi(x.rho, zt, zt)), zx, zx)
           + ops.deriv_y_at_phi(ops.deriv_y_at_m(ops.interp_t_at_phi(x.rho, zt, zt)), zy, zy))
    Rx = (ops.deriv_t_at_phi(x.rho, zt, zt)
          + ops.deriv_x_at_phi(x.mx, zx, zx)
          + ops.deriv_y_at_phi(x.my, zy, zy)
          - VAREPS * lap)

    # R*, exactly as projection_nu_2d applies it.
    dphi_dx, dphi_dy = ops.deriv_x_at_m(phi), ops.deriv_y_at_m(phi)
    nabla = ops.interp_t_at_rho(ops.deriv_x_at_phi(dphi_dx, zx, zx)
                                + ops.deriv_y_at_phi(dphi_dy, zy, zy))
    w_q, w_b = (dt_dual * ca)[:, None, None], (dt_vec * ca)[:, None, None]
    Rs_rho = (ops.deriv_t_at_rho(phi) + VAREPS * nabla) / w_q
    Rs_mx, Rs_my = dphi_dx / w_b, dphi_dy / w_b

    lhs = float(jnp.sum(Rx * phi))
    # <.,.>_N: the same weights that define N, so they cancel R*'s division.
    rhs = -float(
        jnp.sum(w_q * x.rho * Rs_rho)
        + jnp.sum(w_b * x.mx * Rs_mx)
        + jnp.sum(w_b * x.my * Rs_my)
    )
    d = abs(lhs - rhs) / max(abs(rhs), 1e-300)
    return _report("R* is R's N-weighted adjoint (clustered grid)", d < TOL, f"rel={d:.2e}")


# ---------------------------------------------------------------- 4
def check_projection_reduces_to_uniform() -> bool:
    rng = np.random.default_rng(3)
    pu, pn = _uniform_problem(), _nu_problem(uniform_time_grid(NT))
    x = _rand_state(rng, NT, NX, NY)

    out_u = proj_fokker_planck_banded_2d(x, pu, VAREPS, precomp_banded_proj_2d(pu, VAREPS))
    out_n = proj_fokker_planck_banded_nu_2d(x, pn, VAREPS, precomp_banded_proj_nu_2d(pn, VAREPS))

    worst = max(_rel(out_n.rho, out_u.rho), _rel(out_n.mx, out_u.mx), _rel(out_n.my, out_u.my))
    return _report("projection_nu_2d reduces to projection_2d (uniform grid)",
                   worst < TOL, f"rel={worst:.2e}")


def _fp_residual(x: State2D, p, vareps: float) -> jnp.ndarray:
    """The FP residual with the real (inhomogeneous) rho0/rho1 BCs."""
    ops = p.ops
    zx, zy = jnp.zeros((p.nt, p.ny)), jnp.zeros((p.nt, p.nx))
    mu_phi = ops.interp_t_at_phi(x.rho, p.rho0, p.rho1)
    lap = (ops.deriv_x_at_phi(ops.deriv_x_at_m(mu_phi), zx, zx)
           + ops.deriv_y_at_phi(ops.deriv_y_at_m(mu_phi), zy, zy))
    return (ops.deriv_t_at_phi(x.rho, p.rho0, p.rho1)
            + ops.deriv_x_at_phi(x.mx, zx, zx)
            + ops.deriv_y_at_phi(x.my, zy, zy)
            - vareps * lap)


# ---------------------------------------------------------------- 5
def check_projection_is_feasible_nonuniform() -> bool:
    rng = np.random.default_rng(4)
    p = _nu_problem(clustered_time_grid(NT, strength=1.0))
    bp = precomp_banded_proj_nu_2d(p, VAREPS)
    x = _rand_state(rng, NT, NX, NY)

    before = float(jnp.linalg.norm(_fp_residual(x, p, VAREPS)))
    after = float(jnp.linalg.norm(_fp_residual(
        proj_fokker_planck_banded_nu_2d(x, p, VAREPS, bp), p, VAREPS)))
    d = after / max(before, 1e-300)
    return _report("projection output is FP-feasible (clustered grid)",
                   d < 1e-9, f"||R(Px)||/||R(x)||={d:.2e}")


# ---------------------------------------------------------------- 6
def check_projection_is_idempotent_nonuniform() -> bool:
    rng = np.random.default_rng(5)
    p = _nu_problem(clustered_time_grid(NT, strength=1.0))
    bp = precomp_banded_proj_nu_2d(p, VAREPS)
    x = _rand_state(rng, NT, NX, NY)

    p1 = proj_fokker_planck_banded_nu_2d(x, p, VAREPS, bp)
    p2 = proj_fokker_planck_banded_nu_2d(p1, p, VAREPS, bp)
    worst = max(_rel(p2.rho, p1.rho), _rel(p2.mx, p1.mx), _rel(p2.my, p1.my))
    return _report("projection is idempotent (clustered grid)", worst < 1e-9, f"rel={worst:.2e}")


# ---------------------------------------------------------------- 7
def check_pipeline_reduces_to_uniform() -> bool:
    """Same iterate trajectory, given the same starting point.

    x0/y0 are built here and passed to both solvers explicitly, rather than
    letting each build its own default. The two defaults genuinely differ,
    and not because of anything non-uniform: pipeline_2d.py places x0's rho
    on jnp.linspace(0, 1, ntm) (inherited from the 1D pipeline.py), whereas
    the staggered rho actually lives on the ntm *interior* time edges,
    t_edges[1:-1] = [1/nt, ..., (nt-1)/nt] -- which is what pipeline_nu_2d.py
    uses, following pipeline_nu.py's deliberate correction of exactly this.
    Both converge to the same solution, so it only matters for a fixed
    iteration budget; pinning x0/y0 makes this check measure the solver path
    (A*, norms, projection, prox) instead of that initial-guess difference.
    """
    pu, pn = _uniform_problem(), _nu_problem(uniform_time_grid(NT))
    kw = dict(gamma=1.0, tau=1.01, max_iter=30, eps_abs=1e-15, eps_rel=1e-15)

    t_stag = pn.t_edges[1:-1][:, None, None]
    t_cc = pn.t_centers[:, None, None]
    x0 = State2D(
        rho=(1 - t_stag) * pn.rho0[None] + t_stag * pn.rho1[None],
        mx=jnp.zeros((NT, NX - 1, NY)),
        my=jnp.zeros((NT, NX, NY - 1)),
    )
    y0 = State2D(
        rho=(1 - t_cc) * pn.rho0[None] + t_cc * pn.rho1[None],
        mx=jnp.zeros((NT, NX, NY)),
        my=jnp.zeros((NT, NX, NY)),
    )

    ru = discretize_then_optimize_2d(
        pu, proj_fokker_planck_banded_2d, precomp_banded_proj_2d(pu, VAREPS),
        VAREPS, LadmmConfig(**kw), x0=x0, y0=y0)
    rn = discretize_then_optimize_nu_2d(
        pn, proj_fokker_planck_banded_nu_2d, precomp_banded_proj_nu_2d(pn, VAREPS),
        VAREPS, LadmmConfigNU(**kw), x0=x0, y0=y0)

    worst = max(_rel(rn.rho_stag, ru.rho_stag), _rel(rn.mx_stag, ru.mx_stag),
                _rel(rn.my_stag, ru.my_stag), _rel(rn.rho_cc, ru.rho_cc),
                _rel(rn.mx_cc, ru.mx_cc), _rel(rn.my_cc, ru.my_cc))
    return _report("pipeline_nu_2d reduces to pipeline_2d (uniform grid, 30 iters)",
                   worst < 1e-9, f"rel={worst:.2e}")


def main():
    print(f"Validating non-uniform-time 2D modules (nt={NT} nx={NX} ny={NY} vareps={VAREPS:g}) ...")
    checks = [
        check_operators_reduce_to_uniform,
        check_deriv_t_adjoint,
        check_Rstar_is_N_weighted_adjoint,
        check_projection_reduces_to_uniform,
        check_projection_is_feasible_nonuniform,
        check_projection_is_idempotent_nonuniform,
        check_pipeline_reduces_to_uniform,
    ]
    results = [c() for c in checks]
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} checks passed.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
