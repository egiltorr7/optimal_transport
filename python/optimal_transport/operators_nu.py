"""Staggered-grid interpolation/derivative operators for a non-uniform time
grid. Sibling of operators.py.

Only the time-derivative closures change: deriv_t_at_phi/deriv_t_at_rho
divide by the *local* dt_vec[i] instead of the scalar dt. Everything else
carries over verbatim:

  - interp_t_at_phi/interp_t_at_rho are plain averages -- dt never appears
    in them, uniform or not (see time_grid.py: t_centers are exact interval
    midpoints by construction, so a plain average of the two bounding
    "rho" nodes is the exact linear interpolant at the center regardless of
    interval width).
  - deriv_x_at_*/interp_x_at_* only ever touch dx, which stays uniform.
  - interp_t_at_rho_adj/interp_x_at_m_adj: interp_t_at_phi/interp_x_at_phi
    at zero BC, same as operators.py -- still exactly self-adjoint (in the
    same unweighted, Euclidean, array-index inner product operators.py
    uses) since interp_t_at_phi never depended on dt in the first place.

deriv_t_at_rho IS defined as the exact Euclidean adjoint of deriv_t_at_phi
(at zero BC) here -- projection.py's proj_fokker_planck_banded reuses
deriv_t_at_rho as part of applying A^T (see its docstring: "these are
exactly the phi operators evaluated at zero BC ... its own transpose"), and
projection_nu.py's per-mode operator is built by literally reusing that same
formula, so it has to actually BE the adjoint for the assembled system to be
a genuine A A^T (positive semi-definite, hence a well-posed least-squares
projection) rather than an arbitrary non-symmetric system. A naive
"physically motivated" central difference over the distance between
neighboring cell *centers* looks appealing but is NOT this adjoint once
dt_vec is non-uniform (verified numerically -- and empirically: using it
here makes the projection non-contractive, blowing up under LADMM
iteration). Concretely, deriv_t_at_phi is the bidiagonal map
    (D_t)[i,i] = 1/dt_vec[i], (D_t)[i+1,i] = -1/dt_vec[i+1]
so its Euclidean transpose is
    deriv_t_at_rho(v)[j] = v[j+1]/dt_vec[j+1] - v[j]/dt_vec[j]
-- each term normalized by *its own* row's step, not a shared distance.
This coincides with operators.py's (in_arr[1:]-in_arr[:-1])/dt exactly when
dt_vec is constant (both terms then share the same denominator), which is
why the distinction is invisible on a uniform grid.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .operators import Operators, _node_t, _node_x


def build_operators_nu(problem) -> Operators:
    dx = problem.dx
    nt, nx = problem.nt, problem.nx
    dt_vec = problem.dt_vec[:, None]  # (nt, 1), broadcasts against (nt, nx)/(nt, nxm)

    zeros_nx = np.zeros(nx)
    zeros_nt = np.zeros(nt)

    def interp_t_at_phi(in_arr, bc0, bc1):
        node = _node_t(in_arr, bc0, bc1)
        return 0.5 * (node[:-1] + node[1:])

    def interp_t_at_rho(in_arr):
        return 0.5 * (in_arr[:-1] + in_arr[1:])

    def interp_x_at_phi(in_arr, bc0, bc1):
        node = _node_x(in_arr, bc0, bc1)
        return 0.5 * (node[:, :-1] + node[:, 1:])

    def interp_x_at_m(in_arr):
        return 0.5 * (in_arr[:, :-1] + in_arr[:, 1:])

    def deriv_t_at_phi(in_arr, bc0, bc1):
        node = _node_t(in_arr, bc0, bc1)
        return (node[1:] - node[:-1]) / dt_vec

    def deriv_t_at_rho(in_arr):
        return in_arr[1:] / dt_vec[1:] - in_arr[:-1] / dt_vec[:-1]

    def deriv_x_at_phi(in_arr, bc0, bc1):
        node = _node_x(in_arr, bc0, bc1)
        return (node[:, 1:] - node[:, :-1]) / dx

    def deriv_x_at_m(in_arr):
        return (in_arr[:, 1:] - in_arr[:, :-1]) / dx

    def interp_t_at_rho_adj(in_arr):
        return interp_t_at_phi(in_arr, zeros_nx, zeros_nx)

    def interp_x_at_m_adj(in_arr):
        return interp_x_at_phi(in_arr, zeros_nt, zeros_nt)

    return Operators(
        interp_t_at_phi=interp_t_at_phi,
        interp_t_at_rho=interp_t_at_rho,
        interp_x_at_phi=interp_x_at_phi,
        interp_x_at_m=interp_x_at_m,
        deriv_t_at_phi=deriv_t_at_phi,
        deriv_t_at_rho=deriv_t_at_rho,
        deriv_x_at_phi=deriv_x_at_phi,
        deriv_x_at_m=deriv_x_at_m,
        interp_t_at_rho_adj=interp_t_at_rho_adj,
        interp_x_at_m_adj=interp_x_at_m_adj,
    )


def interp_t_at_rho_weighted(v: np.ndarray, dt_vec: np.ndarray) -> np.ndarray:
    """A_t^*: the N,M-weighted adjoint of interp_t_at_phi's time-average
    (docs/nonuniform_grid_norms.tex Sec 4), for N,M the quadrature-consistent
    norms (dual-cell width on q's interior edges, dt_vec elsewhere) --
    NOT interp_t_at_rho, which is only the *Euclidean* adjoint.

    A width-weighted average of v's two flanking cells (each weighted by its
    OWN dt_vec, not the fixed 1/2,1/2 interp_t_at_rho always uses):
        (A_t^* v)[j] = (dt_vec[j]*v[j] + dt_vec[j+1]*v[j+1]) / (dt_vec[j] + dt_vec[j+1])
    Reduces to interp_t_at_rho's plain average exactly when dt_vec is locally
    constant -- true at every row on a uniform grid, essentially no row under
    Chebyshev clustering (see nonuniform_grid_refine_ends.tex's "interp_t_at_rho:
    an accuracy caveat" -- this is precisely the fix that caveat's error term
    was pointing at).

    v: (nt, nx) or (nt,), living at time-centers (phi's grid). Returns
    (nt-1, nx) or (nt-1,), living at q's interior time-edges.

    This is specifically A's (the consensus map's) weighted adjoint, used by
    pipeline_nu.py's At_fn for the linearized-ADMM gradient step. It is
    deliberately NOT wired into Operators/interp_t_at_rho, and NOT used by
    projection_nu.py: R's (the FP-residual's) own weighted adjoint is a
    different, not-yet-derived object (nonuniform_grid_xy_variables.tex
    Sec 11.3's still-open item) -- swapping this in there without that
    separate derivation would be wrong, not just incomplete.
    """
    dt_left, dt_right = dt_vec[:-1], dt_vec[1:]  # (ntm,) each
    w = (dt_left + dt_right)
    shape = (slice(None),) + (None,) * (v.ndim - 1)
    return (dt_left[shape] * v[:-1] + dt_right[shape] * v[1:]) / w[shape]
