"""Grid convergence study for the Gaussian SB solve (single vareps).

For a range of grid resolutions (nx=nt=n everywhere), runs the full pipeline
(staggered discretization + expsemi FP projection + cubic KE prox + linearized
ADMM), compares against the closed-form analytical SB solution on BOTH the
cell-centre grid (y, the KE-optimal variable) and the staggered grid (x, the
FP-feasible variable). See study_utils.py for the error-metric definitions.

x-axis is n = nx = nt, not N = nx*nt. Since h := dx = dt = 1/n, a scheme
that's O(h) is O(n^-1) and O(h^2) is O(n^-2) -- reference lines of these
slopes are plotted alongside the error curves and labelled accordingly.
Printed empirical orders are likewise in the conventional h-sense (i.e. ~2
for a 2nd-order scheme), computed directly from n.

For a run across multiple vareps values, see multi_eps_study.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimal_transport.pipeline import LadmmConfig
from study_utils import RESULTS_DIR, plot_grid_convergence, run_study


def main():
    vareps = 0.1
    resolutions = [16, 32, 64, 128]
    gamma = 1
    # Grid-independent relative stopping criterion (see ladmm.py): D and P
    # both need to drop below eps_abs + eps_rel*scale, where scale is a
    # natural-magnitude quantity built from the same grid-independent norm_fn
    # -- so a fixed eps_rel represents comparable relative precision at every
    # resolution, unlike a fixed absolute tol.
    ladmm_cfg = LadmmConfig(gamma=gamma, tau=gamma*(101.0/100.0), max_iter=40_000, eps_abs=1e-10, eps_rel=1e-15)

    study = run_study(vareps, resolutions, ladmm_cfg)
    out_path = RESULTS_DIR / f"grid_convergence_eps{vareps:g}.png"
    plot_grid_convergence(study, resolutions, vareps, out_path)


if __name__ == "__main__":
    main()
