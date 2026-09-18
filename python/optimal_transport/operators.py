"""Staggered-grid interpolation/derivative operators for 1D.

Port of matlab/shared/1d/discretization/disc_staggered_1st.m.

Grid dimensions:
  rho lives at  (ntm, nx)  -- time-interior nodes, space cell-centers
  mx  lives at  (nt,  nxm) -- time cell-centers,   space-interior nodes
  phi lives at  (nt,  nx)  -- auxiliary potential, cell-centers in both axes

The MATLAB version builds these as dense Toeplitz matrices. Every one of
them is really just a two-point average or difference, so here they are
implemented directly as array-slice arithmetic on a "node" array formed by
stacking the boundary value(s) onto the interior input -- no matrix is ever
built or multiplied.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _node_t(in_arr: np.ndarray, bc0: np.ndarray, bc1: np.ndarray) -> np.ndarray:
    """(ntm, nx) interior + boundary rows -> (ntm+2, nx) full node array."""
    return np.concatenate([bc0[None, :], in_arr, bc1[None, :]], axis=0)


def _node_x(in_arr: np.ndarray, bc0: np.ndarray, bc1: np.ndarray) -> np.ndarray:
    """(nt, nxm) interior + boundary cols -> (nt, nxm+2) full node array."""
    return np.concatenate([bc0[:, None], in_arr, bc1[:, None]], axis=1)


@dataclass
class Operators:
    interp_t_at_phi: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    interp_t_at_rho: Callable[[np.ndarray], np.ndarray]
    interp_x_at_phi: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    interp_x_at_m: Callable[[np.ndarray], np.ndarray]
    deriv_t_at_phi: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    deriv_t_at_rho: Callable[[np.ndarray], np.ndarray]
    deriv_x_at_phi: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    deriv_x_at_m: Callable[[np.ndarray], np.ndarray]
    interp_t_at_rho_adj: Callable[[np.ndarray], np.ndarray]
    interp_x_at_m_adj: Callable[[np.ndarray], np.ndarray]


def build_operators(problem) -> Operators:
    dt = problem.dt
    dx = problem.dx
    nt, nx = problem.nt, problem.nx

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
        return (node[1:] - node[:-1]) / dt

    def deriv_t_at_rho(in_arr):
        return (in_arr[1:] - in_arr[:-1]) / dt

    def deriv_x_at_phi(in_arr, bc0, bc1):
        node = _node_x(in_arr, bc0, bc1)
        return (node[:, 1:] - node[:, :-1]) / dx

    def deriv_x_at_m(in_arr):
        return (in_arr[:, 1:] - in_arr[:, :-1]) / dx

    # Adjoints of the purely linear parts (no BC terms). A symmetric
    # two-point average/difference is, up to the zero-ghost boundary, its
    # own transpose -- so these are exactly the "phi" operators evaluated
    # at zero BC. Verified numerically against a dense-matrix reference,
    # see scratchpad validation.
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
