"""Non-uniform time grids, clustered near t=0 and t=1.

New, additive module: nothing in grid.py/operators.py/projection.py/
pipeline.py depends on this or is changed by it. `setup_problem` (grid.py)
keeps assuming a uniform dt=1/nt throughout; the non-uniform formulation
lives entirely in the *_nu siblings (grid_nu.py, operators_nu.py,
projection_nu.py, pipeline_nu.py), which consume a TimeGrid instead of a
bare nt.

Grid convention (matches operators.py's staggered layout): nt subintervals
on [0,1] with edges t_edges[0..nt] (t_edges[0]=0, t_edges[-1]=1). rho lives
at the nt-1 *interior* edges (t_edges[1:-1]); mx/phi live at the nt interval
midpoints t_centers. Both are exactly what grid.py's uniform nt/dt produce
when t_edges = linspace(0,1,nt+1) -- so uniform_time_grid below is a drop-in
equivalent of grid.py's implicit grid, just made explicit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TimeGrid:
    t_edges: np.ndarray    # (nt+1,) monotone, t_edges[0]=0, t_edges[-1]=1
    dt_vec: np.ndarray      # (nt,)   interval widths, dt_vec = diff(t_edges)
    t_centers: np.ndarray   # (nt,)   interval midpoints (mx/phi live here)
    nt: int

    @property
    def dt_ref(self) -> float:
        """Reference uniform step 1/nt -- mean(dt_vec) by construction, since
        sum(dt_vec)==1. Used to keep gamma/tau/sigma at their usual scale
        when a *_nu module needs a per-row relative weight (see prox_nu.py):
        dt_vec/dt_ref averages to 1 regardless of nt or clustering strength.
        """
        return 1.0 / self.nt


def _time_grid_from_edges(t_edges: np.ndarray) -> TimeGrid:
    dt_vec = np.diff(t_edges)
    if np.any(dt_vec <= 0):
        raise ValueError("t_edges must be strictly increasing")
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    return TimeGrid(t_edges=t_edges, dt_vec=dt_vec, t_centers=t_centers, nt=dt_vec.shape[0])


def uniform_time_grid(nt: int) -> TimeGrid:
    """Uniform grid -- the *_nu modules run this through the exact same code
    path as clustered_time_grid, so it's the regression check against
    grid.py/setup_problem's implicit uniform grid (see validate_nu.py)."""
    return _time_grid_from_edges(np.linspace(0.0, 1.0, nt + 1))


def refine_ends_time_grid(nt: int) -> TimeGrid:
    """Uniform nt-cell grid with cell 0 and cell nt-1 each split into two
    equal halves. Returns a TimeGrid with nt+2 cells total: the first and
    last cells have width h/2 (h = 1/nt), the nt-2 interior cells keep
    width h unchanged.

    Simplest possible non-uniform grid -- a single local refinement at each
    endpoint, as opposed to clustered_time_grid's smooth global reshaping.
    See docs/nonuniform_grid_refine_ends.tex for the worked nt=6 example and
    a hand-check of the operator stencils at the two width junctions (base
    cell 1/2 and base cell nt-2/nt-1). graded_ends_time_grid below is the
    natural next step -- the same idea, applied to more than one cell per
    end with a dyadically increasing refinement level.
    """
    if nt < 2:
        raise ValueError("nt must be >= 2 to refine both ends without overlap")
    h = 1.0 / nt
    base_edges = np.linspace(0.0, 1.0, nt + 1)
    t_edges = np.concatenate([
        [base_edges[0], base_edges[0] + 0.5 * h],
        base_edges[1:-1],
        [base_edges[-1] - 0.5 * h, base_edges[-1]],
    ])
    return _time_grid_from_edges(t_edges)


def graded_ends_time_grid(nt: int, n_boundary: int = 5) -> TimeGrid:
    """Dyadic graded mesh at both ends: the first n_boundary and last
    n_boundary cells of the base nt-cell grid are each recursively halved,
    one more halving for every cell closer to the boundary than the last.

    Concretely, base cell i counted from the boundary (i=1 = the cell
    touching t=0 or t=1, ..., i=n_boundary = the cell furthest from the
    boundary but still in the graded region) is split into
    2**(n_boundary-i+1) equal sub-cells. The "+1" is what keeps the
    recursion self-consistent all the way through: cell i=n_boundary (the
    innermost one) is treated as one level finer than the untouched interior
    cell it borders (which is, implicitly, "level 0"), exactly the same
    "half the width of the one further from the boundary" rule applied to
    every other cell in the chain -- so there's no special-casing at the
    graded/interior seam, just one more application of the same rule with
    the interior playing the role of an (n_boundary+1)-th, unrefined member
    of the chain. This also makes n_boundary=1 reduce exactly to
    refine_ends_time_grid (both split the one boundary cell into 2 halves)
    -- the natural regression check, see validate_nu.py.

    Cell count: each end's n_boundary base cells become 2**(n_boundary+1)-2
    cells (geometric sum), so total cells = nt + 2*(2**(n_boundary+1) - 2 -
    n_boundary). Grows fast in n_boundary (doubling each step) -- keep it
    modest (5 is already 62 cells per end replacing 5) rather than pushing
    it into double digits.
    """
    if n_boundary < 1:
        raise ValueError("n_boundary must be >= 1")
    if 2 * n_boundary > nt:
        raise ValueError(f"2*n_boundary ({2 * n_boundary}) must be <= nt ({nt}): the two graded "
                          f"regions would overlap")
    base_edges = np.linspace(0.0, 1.0, nt + 1)

    left = [base_edges[0]]
    for i in range(1, n_boundary + 1):
        level = n_boundary - i + 1  # i=1 (touches t=0) -> highest level, most-split
        sub = np.linspace(base_edges[i - 1], base_edges[i], 2**level + 1)
        left.extend(sub[1:])  # sub[0] already appended by the previous cell (or base_edges[0])
    left = np.array(left)

    right = 1.0 - left[::-1]  # mirror image about t=0.5
    interior = base_edges[n_boundary: nt - n_boundary + 1]
    t_edges = np.concatenate([left[:-1], interior, right[1:]])
    return _time_grid_from_edges(t_edges)


def clustered_time_grid(nt: int, strength: float = 1.0) -> TimeGrid:
    """Grid clustered near both t=0 and t=1 via a cosine (Chebyshev-like) map.

    xi = linspace(0,1,nt+1) is the uniform reference; c = 0.5*(1-cos(pi*xi))
    is the same map underlying Chebyshev-Gauss-Lobatto nodes, which clusters
    points quadratically near both endpoints and stays near-uniform mid-
    domain. `strength` blends between the identity (0 -> uniform grid,
    exactly reproducing uniform_time_grid) and the full cosine map (1 ->
    maximal clustering, c's own endpoint spacing ~ (pi/nt)^2 vs uniform's
    1/nt). Restricted to [0,1]: c'(xi) dips to 0 at both endpoints, so any
    blend weight above 1 would overshoot into a negative derivative there
    (points folding back on themselves, grid stops being monotone).
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    xi = np.linspace(0.0, 1.0, nt + 1)
    c = 0.5 * (1.0 - np.cos(np.pi * xi))
    t_edges = (1.0 - strength) * xi + strength * c
    t_edges[0], t_edges[-1] = 0.0, 1.0  # pin endpoints exactly (avoid float drift)
    return _time_grid_from_edges(t_edges)
