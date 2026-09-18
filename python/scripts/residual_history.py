"""Plot LADMM residual histories vs iteration, one subplot per residual,
overlaying multiple grid resolutions for comparison (single vareps).

Diagnostic for understanding ADMM convergence behavior across resolutions:
runs one solve per resolution to a fixed iteration budget, then for each of
the five residuals (dx, dy, dz, D, P) plots all resolutions together on a
semilog scale -- kept as separate subplots (rather than all five residuals
overlaid on one axes) since dx/dy/dz/D/P live at different natural scales
(see the dz=gamma*D, P~=tau*dx identities discussed in pipeline.py/ladmm.py)
and would otherwise crowd a single plot.

For a run across multiple vareps values, see multi_eps_study.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimal_transport.pipeline import LadmmConfig
from study_utils import RESULTS_DIR, plot_residual_history, run_study


def main():
    resolutions = [16, 32, 64, 128]
    vareps = 0.0
    # eps_abs=-1, eps_rel=0 -> eps_D=eps_P=-1, never satisfied (D,P>=0), so
    # this always runs the full max_iter budget -- we want the whole
    # trajectory for this diagnostic plot, not an early stop.
    ladmm_cfg = LadmmConfig(gamma=100.0, tau=101.0, max_iter=20_000, eps_abs=-1, eps_rel=0)

    study = run_study(vareps, resolutions, ladmm_cfg)
    out_path = RESULTS_DIR / f"residual_history_vs_resolution_eps{vareps:g}.png"
    plot_residual_history(study, resolutions, vareps, ladmm_cfg, out_path)


if __name__ == "__main__":
    main()
