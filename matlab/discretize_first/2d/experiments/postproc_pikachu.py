"""Post-process the Pikachu -> Raichu .mat files written by run_pikachu_raichu.m.

The solve runs in MATLAB on a GPU box and writes .mat (with -v7, so
scipy.io can read it); the figures are made here.

Per run:
  <stem>_evolution.png   density evolution, 2 rows x 4 columns
  <stem>_residuals.png   primal and dual residual vs iteration

Plus one combined figure comparing the residuals of all runs found.

Usage:
    python3 postproc_pikachu.py [results_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

NROW, NCOL = 2, 4


def scalar(d, k):
    """loadmat wraps every scalar in a 2-D array; unwrap without the
    deprecated implicit ndim>0 -> scalar conversion."""
    return np.asarray(d[k]).ravel()[0]


def label(d):
    return f"eps={scalar(d,'vareps'):g}, nt={int(scalar(d,'nt'))}"


def plot_evolution(d, path, spacing="linear"):
    """Density from t=0 to t=1, over NROW*NCOL panels.

    The FIRST and LAST panels are the PRESCRIBED marginals rho0 and rho1 --
    the boundary conditions at t=0 and t=1 themselves, not solver output.
    rho_cc only ever lives at the nt time-cell CENTRES, (k-0.5)*dt, so it
    never actually reaches t=0 or t=1; showing the given marginals at the
    ends makes the figure span the real interval and lets you check by eye
    that the interior solution arrives where it was told to.

    One shared colour scale across all panels (not per-panel autoscaling):
    the peak density genuinely changes during the transport, and per-panel
    scaling would hide exactly that.
    """
    rho = d["rho_cc"]
    nt = rho.shape[0]
    dt = scalar(d, "dt")
    rho0, rho1 = d["rho0"], d["rho1"]

    n_mid = NROW * NCOL - 2                       # interior panels
    if spacing == "endpoints":
        # Cluster the sample times near t=0 and t=1, geometrically from each
        # end. At large vareps the density diffuses to its Neumann
        # equilibrium (uniform) within a short time and stays there, so
        # evenly spaced panels spend most of the figure showing the same flat
        # field; all the structure is in the fast diffusion out and the
        # reassembly back in. Mirrored about t=1/2 so the two ends get equal
        # treatment.
        k = n_mid // 2
        early = np.unique(np.round(np.geomspace(1, max(nt // 4, 2), k)).astype(int)) - 1
        early = np.clip(early, 0, nt - 1)
        late = (nt - 1 - early)[::-1]
        idx = np.concatenate([early, late])
    else:
        idx = np.unique(np.linspace(0, nt - 1, n_mid).round().astype(int))
    # (array, time, is_given)
    panels = ([(rho0, 0.0, True)]
              + [(rho[k], (k + 0.5) * dt, False) for k in idx]
              + [(rho1, 1.0, True)])
    vmax = float(max(a.max() for a, _, _ in panels))

    fig, axes = plt.subplots(NROW, NCOL, figsize=(3.05 * NCOL, 3.45 * NROW))
    for ax, (arr, t, given) in zip(axes.ravel(), panels):
        im = ax.imshow(arr.T, origin="lower", extent=[0, 1, 0, 1],
                       cmap="magma", vmin=0.0, vmax=vmax)
        if given:
            name = r"$\rho_0$ given" if t == 0.0 else r"$\rho_1$ given"
            ax.set_title(f"t = {t:.0f}   ({name})", fontsize=10, pad=6,
                         fontweight="bold")
            for sp in ax.spines.values():
                sp.set_edgecolor("#1f77b4"); sp.set_linewidth(2.2)
        else:
            ax.set_title(f"t = {t:.3f}", fontsize=10, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(f"Pikachu $\\to$ Raichu:  $\\rho(t,x,y)$   ({label(d)}, "
                 f"{int(scalar(d,'iters'))} iters)", y=0.985, fontsize=13)
    # Explicit spacing rather than tight_layout: with equal-aspect image axes
    # tight_layout under-reserves the inter-row gap and row 2's titles land
    # underneath row 1's panels.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10,
                        wspace=0.06, hspace=0.18)
    cax = fig.add_axes([0.30, 0.045, 0.40, 0.020])
    fig.colorbar(im, cax=cax, orientation="horizontal", label=r"$\rho$")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(d, path):
    """Primal and dual residual vs iteration.

    NAMING: ladmm_solve's own docstring calls the constraint violation the
    "dual" residual, inverting the usual convention. These follow Boyd --
    ||A x + B y - b|| is the PRIMAL residual -- and print the formula, so
    each curve is identified by its mathematics rather than by a name.
    """
    p = np.asarray(d["res_primal"]).ravel()
    q = np.asarray(d["res_dual"]).ravel()
    it = np.arange(1, len(p) + 1)
    ms = scalar(d, "walltime") / max(int(scalar(d, "iters")), 1) * 1e3

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(it, p, lw=1.5, color="#1f77b4",
                label=r"primal  $\|Ax^k+By^k-b\|$")
    ax.semilogy(it, q, lw=1.5, color="#d62728",
                label=r"dual  $\|\tau(x^{k-1}-x^k)+\gamma A^{T}(D^{k-1}-D^k)\|$")
    ax.axhline(scalar(d, "tol"), color="k", ls=":", lw=1,
               label=f"tol = {scalar(d,'tol'):g}  (on $\\|x^k-x^{{k-1}}\\|$)")
    ax.set_xlabel("LADMM iteration")
    ax.set_ylabel("residual norm (grid-weighted $L^2$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    top = ax.secondary_xaxis("top", functions=(lambda i: i * ms / 1e3,
                                               lambda s: s * 1e3 / ms))
    top.set_xlabel(f"wall-clock (s) on GPU, at {ms:.1f} ms/iter")
    ax.set_title(f"Pikachu $\\to$ Raichu convergence  ({label(d)}, "
                 f"$\\gamma$={scalar(d,'gamma'):g}, $\\tau$={scalar(d,'tau'):g}, ETD)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all_residuals(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(runs)))
    for c, (_, d) in zip(colors, runs):
        p = np.asarray(d["res_primal"]).ravel()
        q = np.asarray(d["res_dual"]).ravel()
        it = np.arange(1, len(p) + 1)
        axes[0].semilogy(it, p, lw=1.4, color=c, label=label(d))
        axes[1].semilogy(it, q, lw=1.4, color=c, label=label(d))
    for ax, t in zip(axes, ["primal  $\\|Ax+By-b\\|$", "dual"]):
        ax.set_xlabel("LADMM iteration"); ax.set_ylabel("residual")
        ax.set_title(t); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
    fig.suptitle("Pikachu $\\to$ Raichu: all runs", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    here = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "results" / "pikachu"
    mats = sorted(here.glob("pikachu_raichu_*.mat"))
    if not mats:
        raise SystemExit(f"no .mat files in {here}")

    runs = []
    for m in mats:
        d = loadmat(m)
        runs.append((m, d))
        stem = m.with_suffix("")
        # Large vareps diffuses to a flat field mid-interval, so sample the
        # times near the endpoints instead of uniformly. Threshold on the
        # measured flatness rather than on vareps itself, so this adapts to
        # whatever a run actually did.
        rho = d["rho_cc"]
        mid = rho[rho.shape[0] // 2]
        flat = np.ptp(mid) / max(rho.max(), 1e-300) < 0.1
        spacing = "endpoints" if flat else "linear"
        plot_evolution(d, f"{stem}_evolution.png", spacing=spacing)
        plot_residuals(d, f"{stem}_residuals.png")
        rho = d["rho_cc"]
        dx, dy = scalar(d, "dx"), scalar(d, "dy")
        mass = rho.sum(axis=(1, 2)) * dx * dy
        print(f"{m.name:>40}  {label(d):<16} "
              f"primal {np.asarray(d['res_primal']).ravel()[-1]:.3e}  "
              f"dual {np.asarray(d['res_dual']).ravel()[-1]:.3e}  "
              f"mass [{mass.min():.6f},{mass.max():.6f}]  min rho {rho.min():.2e}  "
              f"[{spacing} times]")
        print(f"{'':>40}  -> {Path(stem).name}_evolution.png, _residuals.png")

    plot_all_residuals(runs, here / "pikachu_raichu_all_residuals.png")
    print(f"\n-> {here/'pikachu_raichu_all_residuals.png'}")


if __name__ == "__main__":
    main()
