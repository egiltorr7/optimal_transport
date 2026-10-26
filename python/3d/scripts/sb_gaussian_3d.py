"""3D analogue of sb_gaussian_2d.py: solve the Schrodinger bridge between two
isotropic 3D Gaussians and compare against the closed-form reference.

Runs on JAX; call jax_config.configure() first to pick CPU/GPU and precision
(see that module's docstring for why x64=True matters for this comparison).

Defaults: ETD scheme, DCT-in-t mode solver, matmul DCT, compiled loop with
50 iterations per chunk, float64. --precision mixedc is the float32-storage
mode that matches float64 (setup/precision.py). Memory: about 55 arrays of
nt*nx*ny*nz entries in the mixed modes, 80 in float64 (docs/gpu_mixed_precision_2d.tex,
Sections 5 and 8): 64^4 is ~7-11 GB, 96^4 ~37-54 GB, 112^4 fits one GH200 only in mixedc.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import jax_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nt", type=int, default=16)
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.02, help="vareps (diffusion)")
    ap.add_argument("--sigma", type=float, default=0.08, help="marginal Gaussian width")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=40_000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu", "tpu"])
    ap.add_argument("--x64", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--scheme", default="etd", choices=["cn", "etd"])
    ap.add_argument("--modes", default="dct", choices=["thomas", "dct"], help="per-mode time solver")
    ap.add_argument("--dct", default="matmul", choices=["fft2n", "fftn", "matmul"], help="DCT kernel")
    ap.add_argument("--jit", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--check-every", type=int, default=50, help="iterations per compiled chunk")
    ap.add_argument("--diag", default="full", choices=["full", "stopping"])
    ap.add_argument("--precision", default="fp64", choices=["fp64", "fp32", "mixed", "mixedm", "mixedc"],
                    help="see setup/precision.py; mixedc is the one that matches fp64")
    args = ap.parse_args()

    # x64 stays on for every mode but fp32 (setup, precompute and references
    # are built in float64; the mixed modes need float64 to exist).
    jax_config.configure(device=args.device, x64=args.x64)
    import jax
    if args.precision != "fp64":
        jax.config.update("jax_default_matmul_precision", "highest")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from setup.grid_3d import setup_problem_3d
    from setup.operators_3d import build_operators_3d
    from setup import projection_3d as P3
    from setup.pipeline_3d import LadmmConfig, discretize_then_optimize_3d
    from setup.precision import configure_precision
    from setup.problems_3d import analytical_sb_gaussian_3d, prob_gaussian_3d
    from setup.projection_3d import precomp_banded_proj_3d, proj_fokker_planck_banded_3d
    from setup.projection_expsemi_3d import precomp_expsemi_proj_3d, proj_fokker_planck_expsemi_3d

    nt, nx, ny, nz = args.nt, args.nx, args.ny, args.nz
    vareps = args.eps

    problem = setup_problem_3d(prob_gaussian_3d(sigma=args.sigma), nt=nt, nx=nx, ny=ny, nz=nz)
    problem.ops = build_operators_3d(problem)

    P3.set_dct_method(args.dct)
    if args.scheme == "cn":
        bp, proj = precomp_banded_proj_3d(problem, vareps, method=args.modes), proj_fokker_planck_banded_3d
    else:
        bp, proj = precomp_expsemi_proj_3d(problem, vareps, method=args.modes), proj_fokker_planck_expsemi_3d
    # the float64 problem stays in scope for the reference; `problem_solve`
    # is what the pipeline sees (float32 in the reduced-precision modes)
    problem_solve, bp, proj, prec_kw = configure_precision(args.precision, problem, bp, proj, build_operators_3d)
    if args.precision == "fp32":
        jax.config.update("jax_enable_x64", False)
    ladmm_cfg = LadmmConfig(
        gamma=args.gamma, tau=args.gamma * (101.0 / 100.0),
        max_iter=args.max_iter, eps_abs=1e-10, eps_rel=1e-8,
        jit=args.jit, check_every=args.check_every, diag_level=args.diag, **prec_kw,
    )

    n_cells = nt * nx * ny * nz
    print(
        f"Running SB Gaussian 3D: nt={nt} nx={nx} ny={ny} nz={nz} "
        f"({n_cells:,} cells, ~{n_cells * 8 / 2**20:.1f} MB per float64 array) "
        f"gamma={ladmm_cfg.gamma:g} tau={ladmm_cfg.tau:g} eps={vareps:g} sigma={args.sigma:g} "
        f"scheme={args.scheme} modes={args.modes} dct={args.dct} jit={args.jit} precision={args.precision} ..."
    )
    result = discretize_then_optimize_3d(problem_solve, proj, bp, vareps, ladmm_cfg)
    jax.block_until_ready(result.rho_stag)
    jax.config.update("jax_enable_x64", True)
    info = result.info
    steady = (info.walltime - info.walltime_first_chunk) / max(info.iters - args.check_every, 1) if args.jit \
        else info.walltime / info.iters
    print(
        f"iters={info.iters} converged={info.converged} walltime={info.walltime:.2f}s "
        f"({steady * 1000:.3f} ms/iter steady state, compile {info.walltime_first_chunk:.1f}s, "
        f"dtype x={result.rho_stag.dtype} y={result.rho_cc.dtype})"
    )

    rho_ana, mx_ana, my_ana, mz_ana = analytical_sb_gaussian_3d(problem, vareps)

    def rel(got, want):
        return float(np.linalg.norm(np.asarray(got) - np.asarray(want)) / np.linalg.norm(np.asarray(want)))

    print(
        f"relative error vs analytical SB:  rho={rel(result.rho_stag, rho_ana):.3e}  "
        f"mx={rel(result.mx_stag, mx_ana):.3e}  my={rel(result.my_stag, my_ana):.3e}  "
        f"mz={rel(result.mz_stag, mz_ana):.3e}"
    )

    # Sanity: total mass at each time slice should stay ~1.
    cell_vol = problem.dx * problem.dy * problem.dz
    mass_t = np.asarray(result.rho_stag).sum(axis=(1, 2, 3)) * cell_vol
    print(f"mass drift: min={mass_t.min():.6f} max={mass_t.max():.6f}")

    # Plot: z-integrated (marginal) density at mid-time, numerical vs
    # analytical vs difference. Integrating out z rather than slicing keeps
    # the comparison from hinging on one arbitrary plane.
    mid = (nt - 1) // 2
    num = np.asarray(result.rho_stag[mid]).sum(axis=2) * problem.dz
    ana = np.asarray(rho_ana[mid]).sum(axis=2) * problem.dz

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = float(max(num.max(), ana.max()))
    for ax, data, title in [
        (axes[0], num, "numerical"),
        (axes[1], ana, "analytical"),
        (axes[2], num - ana, "difference"),
    ]:
        im = ax.imshow(
            data.T, origin="lower", extent=[0, 1, 0, 1],
            vmax=vmax if title != "difference" else None,
        )
        ax.set_title(f"$\\int \\rho\\,dz$ (t={mid}/{nt - 1}) -- {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"sb_gaussian_3d_nt{nt}_nx{nx}_ny{ny}_nz{nz}_eps{vareps:g}.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
