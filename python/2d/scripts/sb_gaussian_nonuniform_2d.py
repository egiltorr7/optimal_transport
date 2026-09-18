"""2D Schrodinger bridge on a non-uniform time grid.

Cross of sb_gaussian_2d.py (uniform 2D) and sb_gaussian_nonuniform.py (1D
non-uniform): swapped over to the *_nu_2d modules end to end (time_grid,
grid_nu_2d, operators_nu_2d, projection_nu_2d, pipeline_nu_2d), and compared
against the closed-form SB solution sampled at the *actual* (non-uniform)
grid points -- problems_nu_2d.analytical_sb_gaussian_nu_2d -- rather than at
uniform k*dt fractions.

The problem: two isotropic 2D Gaussians whose centres both sit on the line
x = y (mu0 = (m0, m0) -> mu1 = (m1, m1)), so the transport runs along that
diagonal and the solution is symmetric under x <-> y. That symmetry is the
point of the diagonal-profile figure below: the bridge is effectively a 1D
problem embedded in 2D at 45 degrees to both grid axes, so nothing about it
is aligned with the x or y stencils, and the combined DCT eigenvalue
lambda_xy = lambda_x + lambda_y has to be right in both axes at once for the
profile to track the closed form.

Grid choices (--grid), all from time_grid.py:
  uniform      the plain grid -- run through the identical code path, so
               uniform and non-uniform runs are directly comparable.
  clustered    cosine clustering at both ends, --strength in [0,1]
               (--strength 0 reproduces `uniform` bit-for-bit).
  graded_ends  dyadic grading over --n-boundary cells at each end.
  refine_ends  cell 0 and cell nt-1 each split in half (nt+2 cells total).

Only the CN/banded projection is wired up -- projection_expsemi_nu.py's ETD
scheme has no 2D counterpart yet, so unlike the 1D script there is no
--scheme to choose between.

Outputs (into 2d/results/): a density-snapshot figure, a diagonal-profile
figure with the analytical reference overlaid, and an .npz of the solution.

NOTE on vareps: the closed-form SB reference is only a meaningful target
below ~5e-3 (study_utils.SINKHORN_VAREPS_THRESHOLD in the 1D tree); above
it, even the trusted 1D solver disagrees with the closed form and its own
scripts switch to a Sinkhorn fit instead. There is no Sinkhorn reference in
the 2D tree, so keep --vareps small (the default is 1e-4) or read the error
numbers as diagnostics rather than as accuracy.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================================
# GPU / precision handles. These MUST be decided before JAX is configured, and
# JAX must be configured before any array exists -- so they are read here, by a
# throwaway parser, ahead of the heavy imports below. Edit the defaults, or
# override with --device / --no-x64 / --no-plot.
# ============================================================================
DEVICE = "gpu"   # "auto" | "cpu" | "gpu" | "tpu". "auto" takes a GPU when a
                  # CUDA-enabled jaxlib is installed and one is visible, else
                  # CPU. Naming a device explicitly makes a missing backend a
                  # hard error instead of a silent fall back to CPU -- on a GPU
                  # box, prefer --device gpu so a broken install is obvious.
X64 = True        # float64. The main GPU MEMORY knob: x64=False halves the
                  # bytes per array, but caps agreement with the float64
                  # analytical reference at ~1e-6..1e-7 regardless of
                  # resolution, so keep it True while validating and turn it
                  # off only once you are memory-bound and know the floor.
PLOT = "auto"     # "auto" | "on" | "off". "auto" plots on CPU and skips on
                  # GPU/TPU: on a GPU run the figures are the slow, useless
                  # part -- the .npz is written either way, post-process later.
GPU_ID = None     # WHICH physical GPU, on a multi-GPU box (None = leave
                  # CUDA_VISIBLE_DEVICES alone). --device gpu only picks the
                  # PLATFORM; without this JAX sees every card and preallocates
                  # memory on all of them, so a run that computes on one still
                  # blocks the rest. Set it and JAX sees exactly that card,
                  # renumbered 0. Check `nvidia-smi` for a free one first.
PREALLOCATE = None  # None = JAX default (grab ~75% of the card up front).
                    # False = grow on demand: slower, but shares the card.
MEM_FRACTION = None  # e.g. 0.4 to cap preallocation and sit alongside another
                     # job. None = JAX default.
# ============================================================================


def _pre_parse():
    """Read only the flags that must act before JAX is imported."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", choices=("auto", "cpu", "gpu", "tpu"), default=DEVICE)
    p.add_argument("--x64", dest="x64", action="store_true", default=X64)
    p.add_argument("--no-x64", dest="x64", action="store_false")
    p.add_argument("--plot", dest="plot", action="store_const", const="on", default=PLOT)
    p.add_argument("--no-plot", dest="plot", action="store_const", const="off")
    p.add_argument("--gpu-id", type=int, default=GPU_ID)
    p.add_argument("--no-preallocate", dest="preallocate", action="store_const",
                   const=False, default=PREALLOCATE)
    p.add_argument("--mem-fraction", type=float, default=MEM_FRACTION)
    return p.parse_known_args()[0]


_PRE = _pre_parse()

from setup import jax_config

jax_config.configure(device=_PRE.device, x64=_PRE.x64, gpu_id=_PRE.gpu_id,
                     preallocate=_PRE.preallocate, mem_fraction=_PRE.mem_fraction)

import jax

BACKEND = jax.default_backend()
DO_PLOT = {"on": True, "off": False}.get(_PRE.plot, BACKEND not in ("gpu", "tpu"))

# matplotlib is imported only when something will actually be drawn -- a GPU box
# need not have it installed, and importing it is not free.
if DO_PLOT:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

import numpy as np

from setup.grid_nu_2d import setup_problem_nu_2d
from setup.operators_nu_2d import build_operators_nu_2d
from setup.pipeline_nu_2d import LadmmConfig, discretize_then_optimize_nu_2d
from setup.problems_2d import prob_gaussian_2d
from setup.problems_nu_2d import analytical_sb_gaussian_nu_2d
from setup.projection_nu_2d import precomp_banded_proj_nu_2d, proj_fokker_planck_banded_nu_2d
from setup.time_grid import (
    clustered_time_grid,
    graded_ends_time_grid,
    refine_ends_time_grid,
    uniform_time_grid,
)

# ============================================================================
# Defaults -- each is overridable from the command line (e.g. `--nt 64`).
# ============================================================================
GRID = "clustered"   # "uniform" | "clustered" | "graded_ends" | "refine_ends"
NT = 32
NX = 64              # see the SIGMA note: 32 leaves only 1.6 cells per sigma
NY = 64
VAREPS = 1e-4        # well below the ~5e-3 threshold, so the closed-form SB
                     # reference is a meaningful target (see module docstring)
MU0 = 1 / 3          # both centres live on the line x = y: mu0 = (MU0, MU0)
MU1 = 2 / 3          #                                      mu1 = (MU1, MU1)
SIGMA = 0.05
# SIGMA/MU0/MU1 match the 1D tree's prob_gaussian() defaults deliberately, and
# NOT prob_gaussian_2d()'s own (sigma=0.1, 0.35 -> 0.65), because the closed-
# form reference is the FREE-SPACE solution renormalized over the box while
# the solver has no-flux walls: the two disagree at the order of the mass the
# box truncates, so that mass has to sit below the error being measured. At
# sigma=0.05, mu=1/3, the centre is 6.7 sigma from the wall -- rho_wall/peak
# ~ 2e-10, truncated mass ~ 5e-11, i.e. genuinely boundary-free. At the 2D
# defaults it is only 3.5 sigma -- truncated mass ~ 9e-4, which at eps=1e-4
# (no diffusion to blur it) is the same order as the density error itself,
# so the boundary contaminates the number rather than perturbing it.
#
# The cost is spatial resolution: a narrower sigma needs a finer grid to
# resolve. NX=NY=64 gives sigma/dx = 3.2 cells; at NX=32 it is 1.6, where
# spatial error dominates instead. The 1D runs get 6.4 cells by using
# nx=128, which is cheap in 1D and is not here.
STRENGTH = 1.0       # clustered only: 0=uniform, 1=full cosine clustering
N_BOUNDARY = 2       # graded_ends only: cells graded at each end
GAMMA = 1.0
TAU = GAMMA * 1.01
MAX_ITER = 40_000
EPS_ABS = 1e-10
EPS_REL = 1e-8
N_SNAPSHOTS = 5      # time slices drawn in both figures

TIME_GRIDS = ("uniform", "clustered", "graded_ends", "refine_ends")


def build_time_grid(name: str, nt: int, strength: float, n_boundary: int):
    if name == "uniform":
        return uniform_time_grid(nt)
    if name == "clustered":
        return clustered_time_grid(nt, strength=strength)
    if name == "graded_ends":
        return graded_ends_time_grid(nt, n_boundary=n_boundary)
    if name == "refine_ends":
        return refine_ends_time_grid(nt)
    raise ValueError(f"unknown grid {name!r}, expected one of {TIME_GRIDS}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=("auto", "cpu", "gpu", "tpu"), default=DEVICE,
                   help="JAX backend (read before JAX is imported; see the handles block)")
    p.add_argument("--x64", dest="x64", action="store_true", default=X64,
                   help="float64 (default). --no-x64 halves memory, costs accuracy")
    p.add_argument("--no-x64", dest="x64", action="store_false")
    p.add_argument("--plot", dest="plot", action="store_const", const="on", default=PLOT,
                   help="force figures on; default skips them on GPU/TPU")
    p.add_argument("--no-plot", dest="plot", action="store_const", const="off",
                   help="skip figures and write only the .npz")
    p.add_argument("--gpu-id", type=int, default=GPU_ID,
                   help="which physical GPU on a multi-GPU box (sets "
                        "CUDA_VISIBLE_DEVICES, so the other cards are neither "
                        "used nor preallocated)")
    p.add_argument("--no-preallocate", dest="preallocate", action="store_const",
                   const=False, default=PREALLOCATE,
                   help="grow GPU memory on demand instead of grabbing ~75%% up front")
    p.add_argument("--mem-fraction", type=float, default=MEM_FRACTION,
                   help="cap GPU preallocation at this fraction, e.g. 0.4")
    p.add_argument("--grid", choices=list(TIME_GRIDS), default=GRID)
    p.add_argument("--nt", type=int, default=NT,
                   help="base time resolution (refine_ends: total cells = nt+2)")
    p.add_argument("--nx", type=int, default=NX)
    p.add_argument("--ny", type=int, default=NY)
    p.add_argument("--vareps", type=float, default=VAREPS)
    p.add_argument("--mu0", type=float, default=MU0, help="start centre, at (mu0, mu0) on x=y")
    p.add_argument("--mu1", type=float, default=MU1, help="end centre, at (mu1, mu1) on x=y")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--strength", type=float, default=STRENGTH,
                   help="clustered grid only: 0=uniform, 1=full cosine clustering")
    p.add_argument("--n-boundary", type=int, default=N_BOUNDARY,
                   help="graded_ends grid only: cells graded at each end")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--tau", type=float, default=None, help="default: gamma * 101/100")
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--eps-abs", type=float, default=EPS_ABS)
    p.add_argument("--eps-rel", type=float, default=EPS_REL)
    p.add_argument("--n-snapshots", type=int, default=N_SNAPSHOTS)
    p.add_argument("--tag", default="", help="extra suffix on the output filenames")
    return p.parse_args()


def _snapshot_indices(ntm: int, n: int) -> np.ndarray:
    """n time indices spread over the ntm interior rho slices, ends included."""
    return np.unique(np.linspace(0, ntm - 1, min(n, ntm)).round().astype(int))


def plot_density_snapshots(rho_num, rho_ana, t_rho, idx, problem, path):
    """Rows = time slices; columns = numerical / analytical / difference."""
    n = len(idx)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.3 * n), squeeze=False)
    for r, k in enumerate(idx):
        vmax = float(max(rho_num[k].max(), rho_ana[k].max()))
        panels = [
            (rho_num[k], "numerical", dict(vmin=0.0, vmax=vmax)),
            (rho_ana[k], "analytical", dict(vmin=0.0, vmax=vmax)),
            (rho_num[k] - rho_ana[k], "difference", dict(cmap="RdBu_r")),
        ]
        for ax, (data, title, kw) in zip(axes[r], panels):
            if title == "difference":
                lim = float(np.abs(data).max()) or 1.0
                kw = dict(kw, vmin=-lim, vmax=lim)
            # .T + origin="lower": array axis 0 is x, axis 1 is y, so the
            # transpose puts x on the horizontal axis and y on the vertical.
            im = ax.imshow(data.T, origin="lower", extent=[0, 1, 0, 1], **kw)
            ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":", alpha=0.6)  # the x=y line
            ax.set_title(f"t = {t_rho[k]:.4f} -- {title}", fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("2D SB density on a non-uniform time grid: rho(t, x, y)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diagonal_profiles(rho_num, rho_ana, t_rho, idx, problem, path):
    """rho along the line x = y, numerical vs the closed-form reference.

    The transport runs along this line, so the diagonal slice carries the
    whole structure of the bridge -- and, unlike an axis-aligned cut, it is
    aligned with neither grid direction, so an error in either axis' DCT
    eigenvalue shows up here rather than cancelling.

    Plotted against x (equivalently y) rather than arc length sqrt(2)*x, so
    the marginal centres mu0/mu1 sit at their nominal coordinates.
    """
    xx = np.asarray(problem.xx)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(idx)))

    for c, k in zip(colors, idx):
        d_num = np.diagonal(rho_num[k])
        d_ana = np.diagonal(rho_ana[k])
        axes[0].plot(xx, d_num, color=c, lw=1.8, label=f"t = {t_rho[k]:.4f}")
        axes[0].plot(xx, d_ana, color=c, lw=1.2, ls="--")
        axes[1].plot(xx, d_num - d_ana, color=c, lw=1.4, label=f"t = {t_rho[k]:.4f}")

    axes[0].set_title("rho along x = y  (solid: numerical, dashed: analytical)")
    axes[0].set_xlabel("x  (along the line x = y)")
    axes[0].set_ylabel("rho(t, x, x)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("numerical - analytical, along x = y")
    axes[1].set_xlabel("x  (along the line x = y)")
    axes[1].set_ylabel("difference")
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    tau = args.tau if args.tau is not None else args.gamma * 101.0 / 100.0

    time_grid = build_time_grid(args.grid, args.nt, args.strength, args.n_boundary)
    prob_def = prob_gaussian_2d(mu0=(args.mu0, args.mu0), mu1=(args.mu1, args.mu1),
                                sigma=args.sigma)
    problem = setup_problem_nu_2d(prob_def, time_grid, nx=args.nx, ny=args.ny)
    problem.ops = build_operators_nu_2d(problem)

    dt_vec = np.asarray(problem.dt_vec)
    print(f"grid={args.grid} nt={problem.nt} (base {args.nt}) nx={args.nx} ny={args.ny} "
          f"vareps={args.vareps:g}")
    print(f"dt_vec: min={dt_vec.min():.6e} max={dt_vec.max():.6e} "
          f"ratio={dt_vec.max() / dt_vec.min():.2f}")
    print(f"marginals: mu0=({args.mu0}, {args.mu0}) -> mu1=({args.mu1}, {args.mu1}) "
          f"on the line x=y, sigma={args.sigma}")

    bp = precomp_banded_proj_nu_2d(problem, args.vareps)
    import os as _os
    print(f"JAX backend: {BACKEND}  devices={jax.devices()}  "
          f"x64={_PRE.x64}  plotting={'on' if DO_PLOT else 'off'}  "
          f"CUDA_VISIBLE_DEVICES={_os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    cfg = LadmmConfig(gamma=args.gamma, tau=tau, max_iter=args.max_iter,
                      eps_abs=args.eps_abs, eps_rel=args.eps_rel)
    result = discretize_then_optimize_nu_2d(
        problem, proj_fokker_planck_banded_nu_2d, bp, args.vareps, cfg)

    info = result.info
    print(f"iters={info.iters} converged={info.converged} walltime={info.walltime:.2f}s "
          f"({info.walltime / max(info.iters, 1) * 1000:.3f} ms/iter)")

    rho_ana, mx_ana, my_ana = analytical_sb_gaussian_nu_2d(problem, args.vareps)
    rho_num = np.asarray(result.rho_stag)
    rho_ana = np.asarray(rho_ana)

    def rel(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))

    print(f"relative error vs analytical SB:  rho={rel(result.rho_stag, rho_ana):.3e}  "
          f"mx={rel(result.mx_stag, mx_ana):.3e}  my={rel(result.my_stag, my_ana):.3e}")

    mass_t = rho_num.sum(axis=(1, 2)) * problem.dx * problem.dy
    print(f"mass drift: min={mass_t.min():.6f} max={mass_t.max():.6f}")

    # Symmetry diagnostic: the problem is invariant under x <-> y, so the
    # solution must be too -- an asymmetry here means the x and y paths of
    # the solver disagree, which the diagonal profile alone would not reveal.
    asym = float(np.abs(rho_num - np.swapaxes(rho_num, 1, 2)).max() / np.abs(rho_num).max())
    print(f"x<->y symmetry: max|rho - rho^T| / max|rho| = {asym:.3e}")

    t_rho = np.asarray(problem.t_edges)[1:-1]
    idx = _snapshot_indices(rho_num.shape[0], args.n_snapshots)

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    stem = (f"sb_gaussian_nonuniform_2d_{args.grid}_nt{args.nt}_nx{args.nx}_ny{args.ny}"
            f"_eps{args.vareps:g}{args.tag}")

    if DO_PLOT:
        plot_density_snapshots(rho_num, rho_ana, t_rho, idx, problem,
                               out_dir / f"{stem}_density.png")
        print(f"saved {out_dir / f'{stem}_density.png'}")

        if args.nx == args.ny:
            plot_diagonal_profiles(rho_num, rho_ana, t_rho, idx, problem,
                                   out_dir / f"{stem}_diagonal.png")
            print(f"saved {out_dir / f'{stem}_diagonal.png'}")
        else:
            print("skipping the x=y profile: it needs nx == ny to be a grid diagonal")
    else:
        print(f"plotting off (backend={BACKEND}) -- writing the .npz only; "
              f"post-process it later")

    np.savez_compressed(
        out_dir / f"{stem}.npz",
        rho_stag=rho_num, mx_stag=np.asarray(result.mx_stag), my_stag=np.asarray(result.my_stag),
        rho_cc=np.asarray(result.rho_cc), rho_ana=rho_ana,
        mx_ana=np.asarray(mx_ana), my_ana=np.asarray(my_ana),
        t_edges=np.asarray(problem.t_edges), t_centers=np.asarray(problem.t_centers),
        dt_vec=dt_vec, xx=np.asarray(problem.xx), yy=np.asarray(problem.yy),
        vareps=args.vareps, grid=args.grid, iters=info.iters,
        # enough metadata to redraw both figures from this file alone
        snapshot_idx=np.asarray(idx), nt=problem.nt, nx=problem.nx, ny=problem.ny,
        dx=problem.dx, dy=problem.dy, sigma=args.sigma, mu0=args.mu0, mu1=args.mu1,
        converged=info.converged, walltime=info.walltime, backend=BACKEND,
        gpu_id=(-1 if _PRE.gpu_id is None else _PRE.gpu_id),
        x64=_PRE.x64, D=np.asarray(info.D), P=np.asarray(info.P),
    )
    print(f"saved {out_dir / f'{stem}.npz'}")


if __name__ == "__main__":
    main()
