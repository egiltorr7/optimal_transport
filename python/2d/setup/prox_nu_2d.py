"""Kinetic-energy prox on the 2D collocated grid, non-uniform time step.
Cross of prox_2d.py and prox_nu.py.

prox_nu.py's conclusion carries over verbatim, and for a reason that does
not care how many spatial dimensions there are: once the y-update's penalty
is weighted by the same M = diag(dt_vec[n] * dx * dy) that the KE objective
f_2 already carries (docs/nonuniform_grid_xy_variables.tex Sec 11.5, eq. 24
-- both are Step 2's own quadrature), that weight cancels out of the
*pointwise* argmin exactly. No dt_vec anywhere, not even implicitly via a
rescaled step -- so this is prox_ke_cc_2d, unchanged, called pointwise.

Kept as a thin wrapper rather than deleted so pipeline_nu_2d.py's call site
(and anything else naming the non-uniform entry point explicitly) doesn't
have to change, exactly as in the 1D prox_nu.py.
"""
from __future__ import annotations

from .prox_2d import prox_ke_cc_2d
from .state_2d import State2D


def prox_ke_cc_nu_2d(x_in: State2D, sigma: float) -> State2D:
    return prox_ke_cc_2d(x_in, sigma)
