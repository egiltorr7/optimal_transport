"""Staggered-grid interpolation/derivative operators for 2D.

Port of matlab/shared/2d/discretization/disc_staggered_1st.m -- 2D
analogue of operators.py, extending the (t, x) staggered scheme with a
second spatial axis y.

Grid dimensions:
  rho lives at  (ntm, nx, ny)  -- time-interior nodes, space cell-centers
  mx  lives at  (nt,  nxm, ny) -- time cell-centers,   x-interior nodes
  my  lives at  (nt,  nx, nym) -- time cell-centers,   y-interior nodes
  phi lives at  (nt,  nx, ny)  -- auxiliary potential, cell-centers in all axes

Same approach as operators.py: every operator is a two-point average or
difference along one axis, implemented as array-slice arithmetic on a
"node" array formed by stacking the boundary plane(s) onto the interior
input -- no matrix is ever built or multiplied, and (unlike the MATLAB
version) no `squeeze`/broadcasting-shape juggling is needed since jnp
broadcasting of a (1, n, ny) chunk against an (n, ny) BC plane on
concatenate just works.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp


def _node_t(in_arr: jnp.ndarray, bc0: jnp.ndarray, bc1: jnp.ndarray) -> jnp.ndarray:
    """(ntm, nx, ny) interior + boundary planes -> (ntm+2, nx, ny)."""
    return jnp.concatenate([bc0[None, :, :], in_arr, bc1[None, :, :]], axis=0)


def _node_x(in_arr: jnp.ndarray, bc0: jnp.ndarray, bc1: jnp.ndarray) -> jnp.ndarray:
    """(nt, nxm, ny) interior + boundary planes -> (nt, nxm+2, ny)."""
    return jnp.concatenate([bc0[:, None, :], in_arr, bc1[:, None, :]], axis=1)


def _node_y(in_arr: jnp.ndarray, bc0: jnp.ndarray, bc1: jnp.ndarray) -> jnp.ndarray:
    """(nt, nx, nym) interior + boundary planes -> (nt, nx, nym+2)."""
    return jnp.concatenate([bc0[:, :, None], in_arr, bc1[:, :, None]], axis=2)


@dataclass
class Operators2D:
    interp_t_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    interp_t_at_rho: Callable[[jnp.ndarray], jnp.ndarray]
    interp_x_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    interp_x_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    interp_y_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    interp_y_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    deriv_t_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    deriv_t_at_rho: Callable[[jnp.ndarray], jnp.ndarray]
    deriv_x_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    deriv_x_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    deriv_y_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    deriv_y_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    interp_t_at_rho_adj: Callable[[jnp.ndarray], jnp.ndarray]
    interp_x_at_m_adj: Callable[[jnp.ndarray], jnp.ndarray]
    interp_y_at_m_adj: Callable[[jnp.ndarray], jnp.ndarray]


def build_operators_2d(problem) -> Operators2D:
    dt, dx, dy = problem.dt, problem.dx, problem.dy
    nt, nx, ny = problem.nt, problem.nx, problem.ny

    zeros_x = jnp.zeros((nt, ny))  # x-wall BC plane: in_arr with the x axis dropped
    zeros_y = jnp.zeros((nt, nx))  # y-wall BC plane: in_arr with the y axis dropped
    zeros_t = jnp.zeros((nx, ny))  # t-wall BC plane: in_arr with the t axis dropped

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
        return (node[1:] - node[:-1]) / dt

    def deriv_t_at_rho(in_arr):
        return (in_arr[1:] - in_arr[:-1]) / dt

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

    # Adjoints of the purely linear parts (no BC terms) -- same reasoning
    # as operators.py's 1D version: a symmetric two-point average/difference
    # is, up to the zero-ghost boundary, its own transpose. Verified
    # numerically against random-array adjoint identities, see scratchpad.
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
