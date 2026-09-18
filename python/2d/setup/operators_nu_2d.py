"""Staggered-grid operators for 2D on a non-uniform time grid.

Cross of operators_2d.py and operators_nu.py. Only the two time-derivative
closures change relative to operators_2d.py; everything else is that module
verbatim, for operators_nu.py's reasons carried into the extra y axis:

  - interp_t_at_phi/interp_t_at_rho are plain averages -- dt never appears
    in them (t_centers are exact interval midpoints, so the plain average of
    the two bounding rho nodes is the exact linear interpolant at the centre
    whatever the interval width).
  - deriv_x_at_*/interp_x_at_*/deriv_y_at_*/interp_y_at_* only touch dx/dy,
    which stay uniform: the non-uniformity is in time only.
  - the three *_adj closures are the *_at_phi maps at zero BC, unchanged,
    since none of them ever depended on dt.

deriv_t_at_rho is again the exact *Euclidean* adjoint of deriv_t_at_phi at
zero BC, not a physically-motivated central difference over the distance
between neighbouring cell centres -- see operators_nu.py's docstring for why
that distinction is not cosmetic (projection_nu_2d.py assembles its per-mode
operator by literally reusing these closures, so it has to genuinely BE the
adjoint for the assembled system to be a symmetric PSD A A^T; the "obvious"
alternative made the 1D projection non-contractive and blew up under LADMM).
Concretely, deriv_t_at_phi is the bidiagonal map

    (D_t)[i,i] = 1/dt_vec[i],   (D_t)[i+1,i] = -1/dt_vec[i+1]

so its transpose is

    deriv_t_at_rho(v)[j] = v[j+1]/dt_vec[j+1] - v[j]/dt_vec[j]

-- each term normalized by *its own* row's step, which coincides with
operators_2d.py's (in_arr[1:]-in_arr[:-1])/dt exactly when dt_vec is
constant. That is why the distinction is invisible on a uniform grid.
"""
from __future__ import annotations

import jax.numpy as jnp

from .operators_2d import Operators2D, _node_t, _node_x, _node_y


def build_operators_nu_2d(problem) -> Operators2D:
    dx, dy = problem.dx, problem.dy
    nt, nx, ny = problem.nt, problem.nx, problem.ny
    dt_vec = problem.dt_vec[:, None, None]  # (nt,1,1), broadcasts against (nt,nx,ny) etc.

    zeros_x = jnp.zeros((nt, ny))  # x-wall BC plane
    zeros_y = jnp.zeros((nt, nx))  # y-wall BC plane
    zeros_t = jnp.zeros((nx, ny))  # t-wall BC plane

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

    def interp_y_at_phi(in_arr, bc0, bc1):
        node = _node_y(in_arr, bc0, bc1)
        return 0.5 * (node[:, :, :-1] + node[:, :, 1:])

    def interp_y_at_m(in_arr):
        return 0.5 * (in_arr[:, :, :-1] + in_arr[:, :, 1:])

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

    def deriv_y_at_phi(in_arr, bc0, bc1):
        node = _node_y(in_arr, bc0, bc1)
        return (node[:, :, 1:] - node[:, :, :-1]) / dy

    def deriv_y_at_m(in_arr):
        return (in_arr[:, :, 1:] - in_arr[:, :, :-1]) / dy

    def interp_t_at_rho_adj(in_arr):
        return interp_t_at_phi(in_arr, zeros_t, zeros_t)

    def interp_x_at_m_adj(in_arr):
        return interp_x_at_phi(in_arr, zeros_x, zeros_x)

    def interp_y_at_m_adj(in_arr):
        return interp_y_at_phi(in_arr, zeros_y, zeros_y)

    return Operators2D(
        interp_t_at_phi=interp_t_at_phi,
        interp_t_at_rho=interp_t_at_rho,
        interp_x_at_phi=interp_x_at_phi,
        interp_x_at_m=interp_x_at_m,
        interp_y_at_phi=interp_y_at_phi,
        interp_y_at_m=interp_y_at_m,
        deriv_t_at_phi=deriv_t_at_phi,
        deriv_t_at_rho=deriv_t_at_rho,
        deriv_x_at_phi=deriv_x_at_phi,
        deriv_x_at_m=deriv_x_at_m,
        deriv_y_at_phi=deriv_y_at_phi,
        deriv_y_at_m=deriv_y_at_m,
        interp_t_at_rho_adj=interp_t_at_rho_adj,
        interp_x_at_m_adj=interp_x_at_m_adj,
        interp_y_at_m_adj=interp_y_at_m_adj,
    )


def interp_t_at_rho_weighted(v: jnp.ndarray, dt_vec: jnp.ndarray) -> jnp.ndarray:
    """A_t^*: the N,M-weighted adjoint of interp_t_at_phi's time-average.

    Identical to operators_nu.py's 1D function -- the weighting is purely in
    time, and the trailing-axis broadcasting below is already written to be
    rank-agnostic, so the same formula serves (nt, nx) and (nt, nx, ny):

        (A_t^* v)[j] = (dt_vec[j]*v[j] + dt_vec[j+1]*v[j+1]) / (dt_vec[j] + dt_vec[j+1])

    A width-weighted average of v's two flanking cells (each weighted by its
    OWN dt_vec), NOT interp_t_at_rho's fixed 1/2,1/2 Euclidean adjoint; the
    two coincide exactly where dt_vec is locally constant (every row of a
    uniform grid, essentially no row under clustering). See
    docs/nonuniform_grid_norms.tex Sec 4.

    v lives at time-centres (phi's grid); returns the (nt-1, ...) array on
    q's interior time-edges.

    This is specifically A's (the consensus map's) weighted adjoint, used by
    pipeline_nu_2d.py's At_fn. Deliberately NOT wired into Operators2D and
    NOT used by projection_nu_2d.py, for operators_nu.py's reason: R's (the
    FP residual's) weighted adjoint is a different object, applied there
    separately.
    """
    dt_left, dt_right = dt_vec[:-1], dt_vec[1:]  # (ntm,) each
    w = dt_left + dt_right
    shape = (slice(None),) + (None,) * (v.ndim - 1)
    return (dt_left[shape] * v[:-1] + dt_right[shape] * v[1:]) / w[shape]
