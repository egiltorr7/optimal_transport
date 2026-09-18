"""Kinetic-energy proximal operator on the cell-centered grid, non-uniform
time step. Sibling of prox.py.

Once the y-update's penalty is honestly weighted by the same M =
diag(dt_vec[n]*dx) that the KE objective f_2 already carries (docs/
nonuniform_grid_xy_variables.tex Sec 11.5, eq. 24 -- both discretize-first,
Step 2's own quadrature), that weight cancels out of the pointwise argmin
*exactly*: no dt_vec anywhere, not even implicitly via a rescaled step. The
per-row sigma_eff rescaling this module used to apply was compensating for a
mismatch (weighted f_2, unweighted penalty) that doesn't exist once M is the
actual metric -- so this is now just prox_ke_cc, unchanged, called
pointwise. Kept as a thin wrapper (rather than deleted) so pipeline_nu.py's
call site, and anything else that names this non-uniform-grid entry point
explicitly, doesn't need to change.

This piece was verified in isolation before (reverted once when wired in
alone -- see pipeline_nu.py's git history / docs/nonuniform_grid_norms.tex
Sec 8) and is only safe to enable together with At_fn's A_t^* adjoint and
projection_nu.py's N-weighted R^* -- all three now switched on together,
tested end-to-end against scripts/sb_gaussian_nonuniform.py.
"""
from __future__ import annotations

from .prox import prox_ke_cc
from .state import State


def prox_ke_cc_nu(x_in: State, sigma: float) -> State:
    return prox_ke_cc(x_in, sigma)
