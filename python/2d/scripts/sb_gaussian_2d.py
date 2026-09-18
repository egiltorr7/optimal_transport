"""2D analogue of sb_gaussian.py: solve the Schrodinger bridge between two
isotropic 2D Gaussians and compare against the closed-form reference.

Runs on JAX; call jax_config.configure() first to pick CPU/GPU and
precision (see that module's docstring for why x64=True matters for this
comparison).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import jax_config

jax_config.configure(device="auto", x64=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from setup.grid_2d import setup_problem_2d
from setup.operators_2d import build_operators_2d
from setup.pipeline_2d import LadmmConfig, discretize_then_optimize_2d
from setup.problems_2d import analytical_sb_gaussian_2d, prob_gaussian_2d
from setup.projection_2d import precomp_banded_proj_2d, proj_fokker_planck_banded_2d


def main():
    nt, nx, ny = 16, 16, 16
    vareps = 0.05
    gamma = 1.0

    problem = setup_problem_2d(prob_gaussian_2d(sigma=0.12), nt=nt, nx=nx, ny=ny)
    problem.ops = build_operators_2d(problem)

    bp = precomp_banded_proj_2d(problem, vareps)
    ladmm_cfg = LadmmConfig(gamma=gamma, tau=gamma * (101.0 / 100.0), max_iter=40_000, eps_abs=1e-10, eps_rel=1e-8)

    print(
        f"Running SB Gaussian 2D: nt={nt} nx={nx} ny={ny} gamma={ladmm_cfg.gamma:g} "
        f"tau={ladmm_cfg.tau:g} eps={vareps:g} ..."
    )
    result = discretize_then_optimize_2d(problem, proj_fokker_planck_banded_2d, bp, vareps, ladmm_cfg)
    info = result.info
    print(
        f"iters={info.iters} converged={info.converged} walltime={info.walltime:.2f}s "
        f"({info.walltime / info.iters * 1000:.3f} ms/iter)"
    )

    rho_ana, mx_ana, my_ana = analytical_sb_gaussian_2d(problem, vareps)
    rho_err = float(np.linalg.norm(result.rho_stag - rho_ana) / np.linalg.norm(rho_ana))
    mx_err = float(np.linalg.norm(result.mx_stag - mx_ana) / np.linalg.norm(mx_ana))
    my_err = float(np.linalg.norm(result.my_stag - my_ana) / np.linalg.norm(my_ana))
    print(f"relative error vs analytical SB:  rho={rho_err:.3e}  mx={mx_err:.3e}  my={my_err:.3e}")

    # Sanity: total mass at each time slice should stay ~1.
    mass_t = np.asarray(result.rho_stag).sum(axis=(1, 2)) * problem.dx * problem.dy
    print(f"mass drift: min={mass_t.min():.6f} max={mass_t.max():.6f}")

    # Plot mid-time density slice, numerical vs analytical, side by side.
    mid = (nt - 1) // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = float(max(result.rho_stag[mid].max(), rho_ana[mid].max()))
    for ax, data, title in [
        (axes[0], np.asarray(result.rho_stag[mid]), "numerical"),
        (axes[1], np.asarray(rho_ana[mid]), "analytical"),
        (axes[2], np.asarray(result.rho_stag[mid] - rho_ana[mid]), "difference"),
    ]:
        im = ax.imshow(data.T, origin="lower", extent=[0, 1, 0, 1], vmax=vmax if title != "difference" else None)
        ax.set_title(f"rho(t={mid}/{nt - 1}) -- {title}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"sb_gaussian_2d_nt{nt}_nx{nx}_ny{ny}_eps{vareps:g}.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
