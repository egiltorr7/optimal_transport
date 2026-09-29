"""Density evolution as an x-t waterfall, one figure per (vareps, resolution).

Each selected time is drawn as a curve in (x, t, rho) at its true t, so the
spacing along the time axis shows WHERE the snapshots were taken, not just
their order.

CHOOSING THE TIMES is the difficulty. Uniformly spaced times work at small
vareps, where the bimodal pair simply translates and merges, but not at large
vareps: diffusion flattens the structure within
    t_diff ~ sigma^2 / vareps
(sigma=0.05 -> t_diff = 2.5 at vareps=1e-3, but 0.0025 at vareps=1), after
which every snapshot is the same near-uniform blob. So:

  t_diff >= 1/2   structure survives the interval -> uniform fractions.
  t_diff <  1/2   it collapses early -> sample LOGARITHMICALLY from the first
                  grid time out to 1/2 and mirror about 1/2, putting the
                  snapshots inside the window where something is happening at
                  both ends. On the clustered time grid this costs nothing:
                  dt is already fine there. In the waterfall this shows up as
                  curves bunched at the two ends with a gap across the middle,
                  which is itself the point -- the middle is featureless.

Requested times are snapped to the nearest actual grid row.

Usage:
    python scripts/post_density_profiles.py --case bimodal
    python scripts/post_density_profiles.py --case gaussian --n-times 15
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

SUITE = Path(__file__).resolve().parent.parent / "results" / "suite"
OUT = Path(__file__).resolve().parent.parent / "results" / "post"
EPS_LIST = [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SIGMA = 0.05
RESOLUTIONS = [32, 64, 128, 256]
# Only the finest grid is drawn by default: these figures are for showing what
# the solution looks like, and the coarse grids only add under-resolved copies
# (at n=32 a sigma=0.05 hump sits on 1.6 cells). --n 64 etc. draws the others.
DEFAULT_RESOLUTIONS = [256]
# Snapshot count per branch. Kept separate so the two sampling rules can
# diverge if one ever needs a different density; currently both are 19.
# --n-times overrides both.
N_TIMES_UNIFORM = 19
N_TIMES_CLUSTERED = 19
# Chosen from contact sheets of view angles.
#
# AZIM sets how horizontal each axis sits: for a camera at azimuth th the
# screen-horizontal extent is ~|sin th| for x and ~|cos th| for t, so th=-45 is
# the crossover. At -60 (x 0.87 / t 0.50) x dominates and time recedes steeply;
# at -40 (x 0.64 / t 0.77) time dominates and x recedes far enough to start
# squashing the profiles. -50 (x 0.77 / t 0.64) sits between: time is tilted up
# toward a textbook y-axis while x keeps enough extent to draw the humps.
#
# BOX_ASPECT matters more than either angle: stretching the TIME axis against a
# short density axis is what pulls successive snapshots apart. Raising elev
# instead just stacks them on top of each other.
ELEV, AZIM = 20, -50
BOX_ASPECT = (1.4, 1.9, 0.6)     # (x, t, rho)


def tag(case, vareps, n):
    grid = "clustered" if vareps >= 1e-2 else "uniform"
    return f"{case}_eps{vareps:g}_n{n}_{grid}_ETD"


def pick_times(t_grid, vareps, n_times=None):
    """n_times=None -> the per-branch default (see N_TIMES_* above)."""
    t_diff = np.inf if vareps <= 0 else SIGMA**2 / vareps
    if t_diff >= 0.5:
        count = n_times or N_TIMES_UNIFORM
        want = np.linspace(t_grid[0], t_grid[-1], count)
    else:
        # Cosine (Chebyshev) clustering, NOT geometric. Geometric spacing spans
        # decades, but the plot's time axis is LINEAR, so those snapshots land
        # on top of each other near t=0 and t=1 while leaving a bare middle.
        # The cosine map biases toward both ends by a modest, bounded factor
        # (end gaps ~ (pi/n)^2/2 against a uniform 1/n, so ~10x at n=19 rather
        # than the ~1000x of a geomspace) and stays visually even. It also
        # includes both endpoints exactly, so the two marginals always appear.
        count = n_times or N_TIMES_CLUSTERED
        u = np.linspace(0.0, 1.0, count)
        want = t_grid[0] + (t_grid[-1] - t_grid[0]) * 0.5 * (1 - np.cos(np.pi * u))
    idx = np.unique([int(np.argmin(np.abs(t_grid - w))) for w in want])
    return idx, t_diff


def waterfall(case, vareps, n, n_times, show_ref, elev=ELEV, azim=AZIM):
    f = SUITE / f"{tag(case, vareps, n)}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    rho, xx = d["rho_stag"], d["xx"]
    t_grid = d["t_edges"][1:-1]
    idx, t_diff = pick_times(t_grid, vareps, n_times)

    refs = [r for r in d["ref_names"]]
    ref_name = ("sinkhorn_same" if "sinkhorn_same" in refs
                else ("analytic" if "analytic" in refs else None))
    ref_rho = d[f"ref_{ref_name}_rho_stag"] if (ref_name and show_ref) else None

    fig = plt.figure(figsize=(9.5, 5.2))
    ax = fig.add_subplot(projection="3d")
    # colour encodes time, so the ordering is readable without the legend
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(idx)))
    for k, c in zip(idx, colors):
        tk = np.full_like(xx, t_grid[k])
        ax.plot(xx, tk, rho[k], color=c, lw=1.7)
        if ref_rho is not None:
            ax.plot(xx, tk, ref_rho[k], color="0.35", lw=0.8, ls="--")

    ax.set_xlabel("Position $x$", labelpad=8)
    ax.set_ylabel("Time $t$", labelpad=8)
    ax.set_zlabel(r"Density $\rho$", labelpad=6)
    ax.set_ylim(0, 1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(BOX_ASPECT)
    # Drop the shaded panes entirely and keep a faint grid: at this angle the
    # default grey boxes compete with the curves.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("0.85")
        axis._axinfo["grid"]["color"] = "0.9"
    ax.grid(True)
    # tight_layout leaves a large empty band around a 3d axes (it reserves room
    # for the UNROTATED bounding box, while a flat view fills only a horizontal
    # band of it), so place the axes explicitly and title via the figure.
    ax.set_position([0.01, -0.11, 0.98, 1.16])
    # Snapshot count and which sampling rule fired are reported on stdout by
    # main(), so they stay out of the figure.
    fig.suptitle(f"Density Evolution: $\\varepsilon$ = {vareps:g}, n = {n}",
                 fontsize=12.5, y=0.95)
    p = OUT / f"waterfall_{case}_eps{vareps:g}_n{n}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p, idx, t_grid, t_diff


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="bimodal")
    ap.add_argument("--n-times", type=int, default=None,
                    help=f"override the per-branch defaults "
                         f"(uniform {N_TIMES_UNIFORM}, clustered {N_TIMES_CLUSTERED})")
    ap.add_argument("--elev", type=float, default=ELEV)
    ap.add_argument("--azim", type=float, default=AZIM)
    ap.add_argument("--n", type=int, action="append",
                    help=f"resolutions to draw (default: {DEFAULT_RESOLUTIONS}; "
                         f"available: {RESOLUTIONS})")
    ap.add_argument("--show-ref", action="store_true",
                    help="overlay the reference as thin dashed curves")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    resolutions = a.n or DEFAULT_RESOLUTIONS
    made = 0
    for vareps in EPS_LIST:
        for n in resolutions:
            r = waterfall(a.case, vareps, n, a.n_times, a.show_ref,
                          elev=a.elev, azim=a.azim)
            if r is None:
                continue
            p, idx, t_grid, t_diff = r
            kind = "uniform" if t_diff >= 0.5 else "clustered"
            print(f"  {p.name:<42} [{kind:>9}] {len(idx):2d} snapshots, "
                  f"t in [{t_grid[idx[0]]:.4f}, {t_grid[idx[-1]]:.4f}]")
            made += 1
    print(f"\n{made} figures in {OUT}")


if __name__ == "__main__":
    main()
