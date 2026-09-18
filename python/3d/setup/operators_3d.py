"""Staggered-grid interpolation/derivative operators for 3D.

3D analogue of operators_2d.py, extending the (t, x, y) staggered scheme
with a third spatial axis z.

Grid dimensions:
  rho lives at  (ntm, nx, ny, nz)  -- time-interior nodes, space cell-centers
  mx  lives at  (nt,  nxm, ny, nz) -- time cell-centers,   x-interior nodes
  my  lives at  (nt,  nx, nym, nz) -- time cell-centers,   y-interior nodes
  mz  lives at  (nt,  nx, ny, nzm) -- time cell-centers,   z-interior nodes
  phi lives at  (nt,  nx, ny, nz)  -- auxiliary potential, cell-centers in all axes

Same approach as operators_2d.py: every operator is a two-point average or
difference along one axis, implemented as array-slice arithmetic on a
"node" array formed by stacking the boundary hyperplane(s) onto the
interior input -- no matrix is ever built or multiplied.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp


def _node_t(in_arr: jnp.ndarray, bc0: jnp.ndarray, bc1: jnp.ndarray) -> jnp.ndarray:
    """(ntm, nx, ny, nz) interior + boundary planes -> (ntm+2, nx, ny, nz)."""
    return jnp.concatenate([bc0[None, :, :, :], in_arr, bc1[None, :, :, :]], axis=0)


def _node_x(in_arr: jnp.ndarray, bc0: jnp.ndarray, bc1: jnp.ndarray) -> jnp.ndarray:
    """(nt, nxm, ny, nz) interior + boundary planes -> (nt, nxm+2, ny, nz)."""
    return jnp.concatenate([bc0[:, None, :, :], in_arr, bc1[:, None, :, :]], axis=1)


def _node_y(in_arr: jnp.ndarray, bc0: jnp.ndarray, bc1: jnp.ndarray) -> jnp.ndarray:
    """(nt, nx, nym, nz) interior + boundary planes -> (nt, nx, nym+2, nz)."""
    return jnp.concatenate([bc0[:, :, None, :], in_arr, bc1[:, :, None, :]], axis=2)


def _node_z(in_arr: jnp.ndarray, bc0: jnp.ndarray, bc1: jnp.ndarray) -> jnp.ndarray:
    """(nt, nx, ny, nzm) interior + boundary planes -> (nt, nx, ny, nzm+2)."""
    return jnp.concatenate([bc0[:, :, :, None], in_arr, bc1[:, :, :, None]], axis=3)


@dataclass
class Operators3D:
    interp_t_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    interp_t_at_rho: Callable[[jnp.ndarray], jnp.ndarray]
    interp_x_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    interp_x_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    interp_y_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    interp_y_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    interp_z_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    interp_z_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    deriv_t_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    deriv_t_at_rho: Callable[[jnp.ndarray], jnp.ndarray]
    deriv_x_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    deriv_x_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    deriv_y_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    deriv_y_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    deriv_z_at_phi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    deriv_z_at_m: Callable[[jnp.ndarray], jnp.ndarray]
    interp_t_at_rho_adj: Callable[[jnp.ndarray], jnp.ndarray]
    interp_x_at_m_adj: Callable[[jnp.ndarray], jnp.ndarray]
    interp_y_at_m_adj: Callable[[jnp.ndarray], jnp.ndarray]
    interp_z_at_m_adj: Callable[[jnp.ndarray], jnp.ndarray]


def build_operators_3d(problem) -> Operators3D:
    dt, dx, dy, dz = problem.dt, problem.dx, problem.dy, problem.dz
    nt, nx, ny, nz = problem.nt, problem.nx, problem.ny, problem.nz

    # BC hyperplanes: the operand array with its own axis dropped.
    zeros_t = jnp.zeros((nx, ny, nz))  # t-wall
    zeros_x = jnp.zeros((nt, ny, nz))  # x-wall
    zeros_y = jnp.zeros((nt, nx, nz))  # y-wall
    zeros_z = jnp.zeros((nt, nx, ny))  # z-wall

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

    def interp_z_at_phi(in_arr, bc0, bc1):
        node = _node_z(in_arr, bc0, bc1)
        return 0.5 * (node[:, :, :, :-1] + node[:, :, :, 1:])

    def interp_z_at_m(in_arr):
        return 0.5 * (in_arr[:, :, :, :-1] + in_arr[:, :, :, 1:])

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

    def deriv_z_at_phi(in_arr, bc0, bc1):
        node = _node_z(in_arr, bc0, bc1)
        return (node[:, :, :, 1:] - node[:, :, :, :-1]) / dz

    def deriv_z_at_m(in_arr):
        return (in_arr[:, :, :, 1:] - in_arr[:, :, :, :-1]) / dz

    # Adjoints of the purely linear parts (no BC terms) -- same reasoning as
    # operators_2d.py: a symmetric two-point average/difference is, up to the
    # zero-ghost boundary, its own transpose. Checked numerically against
    # random-array adjoint identities by scripts/validate_3d.py.
    def interp_t_at_rho_adj(in_arr):
        return interp_t_at_phi(in_arr, zeros_t, zeros_t)

    def interp_x_at_m_adj(in_arr):
        return interp_x_at_phi(in_arr, zeros_x, zeros_x)

    def interp_y_at_m_adj(in_arr):
        return interp_y_at_phi(in_arr, zeros_y, zeros_y)

    def interp_z_at_m_adj(in_arr):
        return interp_z_at_phi(in_arr, zeros_z, zeros_z)

    return Operators3D(
        interp_t_at_phi=interp_t_at_phi,
        interp_t_at_rho=interp_t_at_rho,
        interp_x_at_phi=interp_x_at_phi,
        interp_x_at_m=interp_x_at_m,
        interp_y_at_phi=interp_y_at_phi,
        interp_y_at_m=interp_y_at_m,
        interp_z_at_phi=interp_z_at_phi,
        interp_z_at_m=interp_z_at_m,
        deriv_t_at_phi=deriv_t_at_phi,
        deriv_t_at_rho=deriv_t_at_rho,
        deriv_x_at_phi=deriv_x_at_phi,
        deriv_x_at_m=deriv_x_at_m,
        deriv_y_at_phi=deriv_y_at_phi,
        deriv_y_at_m=deriv_y_at_m,
        deriv_z_at_phi=deriv_z_at_phi,
        deriv_z_at_m=deriv_z_at_m,
        interp_t_at_rho_adj=interp_t_at_rho_adj,
        interp_x_at_m_adj=interp_x_at_m_adj,
        interp_y_at_m_adj=interp_y_at_m_adj,
        interp_z_at_m_adj=interp_z_at_m_adj,
    )
