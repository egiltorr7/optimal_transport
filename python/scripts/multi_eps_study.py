"""Grid convergence + residual history, for vareps in {0, 1e-4, 1e-2}.

Runs each resolution once per vareps (using the grid-independent D/P
stopping criterion from ladmm.py) and produces both a grid-convergence plot
and a residual-history plot per vareps, sharing the same underlying solves
between the two rather than re-running the pipeline twice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimal_transport.pipeline import LadmmConfig
from study_utils import RESULTS_DIR, plot_grid_convergence, plot_residual_history, run_study


def main():
    resolutions = [16, 32, 64, 128, 256, 512]
    vareps_list = [0, 1e-3, 1e-2, 1e-1,1, 10]
    # Grid-independent relative stopping criterion (see ladmm.py): a fixed
    # eps_rel represents comparable relative precision at every resolution,
    # unlike a fixed absolute tol.
    ladmm_cfg = LadmmConfig(gamma=100.0, tau=101.0, max_iter=20_000, eps_abs=1e-10, eps_rel=1e-15)

    for vareps in vareps_list:
        print(f"\n{'=' * 60}\nvareps = {vareps:g}\n{'=' * 60}")
        study = run_study(vareps, resolutions, ladmm_cfg)

        plot_grid_convergence(
            study, resolutions, vareps, RESULTS_DIR / f"grid_convergence_eps{vareps:g}.png"
        )
        plot_residual_history(
            study, resolutions, vareps, ladmm_cfg,
            RESULTS_DIR / f"residual_history_vs_resolution_eps{vareps:g}.png",
        )


if __name__ == "__main__":
    main()
