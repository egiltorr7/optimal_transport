"""Sweep gamma (tau = TAU_RATIO*gamma held fixed) across vareps and resolution,
to find good LADMM step sizes under the new density-convention normalization
(rho0/rho1 now satisfy sum(rho)*dx == 1, not sum(rho) == 1 -- see grid.py).
That change alters the KE objective's magnitude, so the old gamma=100 default
is not guaranteed to still be well-tuned.

For each vareps, produces one figure: 7 rows (dx, dy, dz, D, P residuals,
plus err_x = ||x^k - x_ref|| and err_delta = ||delta^k - delta_ref|| against
the analytical/Sinkhorn ground truth) x 4 columns (resolutions), each
subplot overlaying the residual history for every gamma in the sweep, so a
good step size can be read off directly. err_x/err_delta are the only two
of the seven that are an actual distance to the true solution (the other
five are purely internal algorithmic quantities) -- see ladmm.py's
LadmmOpts.x_ref/delta_ref/delta_mask and LadmmInfo.err_x/err_delta.

delta_ref = -grad f2(y_ref), valid only where f2 is differentiable (rho>0);
wherever the reference rho is at/near machine-zero, f2's true KKT condition
is -delta* in the SUBdifferential (a set, not a point) rather than a unique
gradient value, and dividing by the near-zero float rho independently loses
all precision anyway -- delta_mask (built from a rho>1e-12 floor, matching
prox_ke_cc's own convention) excludes those points from the err_delta norm
rather than comparing against an arbitrary/unreliable single value.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimal_transport.grid import setup_problem
from optimal_transport.operators import build_operators
from optimal_transport.pipeline import LadmmConfig, discretize_then_optimize
from optimal_transport.problems import prob_gaussian
from optimal_transport.projection_expsemi import precomp_expsemi_proj, proj_fokker_planck_expsemi
from optimal_transport.state import State
from study_utils import (
    RESULTS_DIR,
    SINKHORN_MAX_ITER,
    SINKHORN_NX_FINE,
    SINKHORN_TOL,
    SINKHORN_VAREPS_THRESHOLD,
    reference_solution,
)
from optimal_transport.sinkhorn import sinkhorn_fit

TAU_RATIO = 101.0 / 100.0  # same tau/gamma ratio as the gamma=100, tau=101 default
RHO_FLOOR = 1e-12  # matches prox_ke_cc's own near-zero-density floor


def _build_delta_ref(rho_ref_cc, mx_ref_cc):
    """delta_ref = -grad f2(y_ref), f2(rho,m) = m^2/(2*rho); delta_mask zeros
    out points where rho_ref_cc <= RHO_FLOOR (f2 not differentiable there --
    see module docstring)."""
    mask2d = (rho_ref_cc > RHO_FLOOR).astype(float)
    rho_safe = np.where(rho_ref_cc > RHO_FLOOR, rho_ref_cc, 1.0)
    delta_ref = State(
        rho=mx_ref_cc**2 / (2 * rho_safe**2),
        mx=-mx_ref_cc / rho_safe,
    )
    delta_mask = State(mask2d, mask2d)
    return delta_ref, delta_mask


def main():
    gammas = [0.01, 0.1, 1, 10, 100]
    resolutions = [16, 32, 64, 128]
    vareps_list = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    max_iter = 40_000
    eps_abs, eps_rel = 1e-10, 1e-10

    colors = plt.cm.plasma(np.linspace(0, 0.85, len(gammas)))
    summary = {}  # (vareps, n) -> list of (gamma, iters, converged, final_err_x, final_err_delta)

    residual_names = ["dx", "dy", "dz", "D", "P", "err_x", "err_delta"]
    titles = {
        "dx": r"$dx = \|x^{k+1}-x^k\|$",
        "dy": r"$dy = \|y^{k+1}-y^k\|$",
        "dz": r"$dz = \|\delta^{k+1}-\delta^k\|$",
        "D": "D (dual residual)",
        "P": "P (primal residual)",
        "err_x": r"$\|x^k - x_{ref}\|$ (true error)",
        "err_delta": r"$\|\delta^k - \delta_{ref}\|$ (true dual error)",
    }

    for vareps in vareps_list:
        # Reference solution for err_x: analytical (small vareps) or a single
        # fine-grid Sinkhorn fit, reused (coarsened) across every resolution
        # in this vareps block -- mirrors study_utils.run_study.
        sink_fit = None
        if vareps > SINKHORN_VAREPS_THRESHOLD:
            fine_problem = setup_problem(prob_gaussian(), nt=SINKHORN_NX_FINE, nx=SINKHORN_NX_FINE)
            sink_fit = sinkhorn_fit(fine_problem, vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
            print(
                f"vareps={vareps:g}: Sinkhorn reference fit (nx={SINKHORN_NX_FINE})  "
                f"iters={sink_fit.iters}  converged={sink_fit.converged}  error={sink_fit.error:.2e}"
            )

        fig, axes = plt.subplots(
            len(residual_names), len(resolutions), figsize=(4.2 * len(resolutions), 3.2 * len(residual_names)),
            sharex="col",
        )

        for col, n in enumerate(resolutions):
            problem = setup_problem(prob_gaussian(), nt=n, nx=n)
            problem.ops = build_operators(problem)
            ep = precomp_expsemi_proj(problem, vareps)

            rho_ref_stag, mx_ref_stag, rho_ref_cc, mx_ref_cc = reference_solution(
                problem, vareps, problem.ops, sink_fit
            )
            x_ref = State(rho_ref_stag, mx_ref_stag)
            delta_ref, delta_mask = _build_delta_ref(rho_ref_cc, mx_ref_cc)

            print(f"vareps={vareps:g}  n={n}")
            summary[(vareps, n)] = []
            for gamma, color in zip(gammas, colors):
                tau = gamma * TAU_RATIO
                cfg = LadmmConfig(gamma=gamma, tau=tau, max_iter=max_iter, eps_abs=eps_abs, eps_rel=eps_rel)
                result = discretize_then_optimize(
                    problem, proj_fokker_planck_expsemi, ep, vareps, cfg,
                    x_ref=x_ref, delta_ref=delta_ref, delta_mask=delta_mask,
                )
                info = result.info
                it = np.arange(1, info.iters + 1)
                for row, name in enumerate(residual_names):
                    axes[row, col].semilogy(it, getattr(info, name), color=color, lw=1.2, label=f"$\\gamma$={gamma:g}")

                final_err_x = info.err_x[-1]
                final_err_delta = info.err_delta[-1]
                summary[(vareps, n)].append((gamma, info.iters, info.converged, final_err_x, final_err_delta))
                print(
                    f"  gamma={gamma:6g}  iters={info.iters:6d}  converged={info.converged}  "
                    f"final err_x={final_err_x:.3e}  final err_delta={final_err_delta:.3e}"
                )

            axes[0, col].set_title(f"n={n}")
            for row in range(len(residual_names)):
                axes[row, col].grid(True, which="both")
            axes[-1, col].set_xlabel("iteration")

        for row, name in enumerate(residual_names):
            axes[row, 0].set_ylabel(titles[name])
        axes[0, -1].legend(fontsize=8, loc="upper right")

        fig.suptitle(f"Gamma sweep (tau = {TAU_RATIO:.4g}*gamma), vareps={vareps:g}")
        fig.tight_layout()

        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / f"gamma_sweep_eps{vareps:g}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  figure saved to: {out_path}\n")

    print("\n" + "=" * 70)
    print("Best gamma per (vareps, n):")
    print("=" * 70)
    for (vareps, n), rows in summary.items():
        converged_rows = [r for r in rows if r[2]]
        if converged_rows:
            best_speed = min(converged_rows, key=lambda r: r[1])
            print(
                f"  vareps={vareps:8g}  n={n:4d}  fewest-iters gamma={best_speed[0]:6g}  "
                f"iters={best_speed[1]}  final err_x={best_speed[3]:.3e}  final err_delta={best_speed[4]:.3e}"
            )
        else:
            print(f"  vareps={vareps:8g}  n={n:4d}  NONE converged within {max_iter} iters")
        best_err = min(rows, key=lambda r: r[3])
        print(
            f"  vareps={vareps:8g}  n={n:4d}  lowest-err_x gamma={best_err[0]:6g}  "
            f"iters={best_err[1]}  converged={best_err[2]}  final err_x={best_err[3]:.3e}  "
            f"final err_delta={best_err[4]:.3e}"
        )


if __name__ == "__main__":
    main()
