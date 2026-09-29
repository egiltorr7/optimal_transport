"""Post-process an .npz written by sb_gaussian_nonuniform_2d.py.

Exists because a GPU run writes no figures: the solver script turns plotting
off on a GPU backend (a compute node need not have matplotlib, and importing
it is not free), so a DeltaAI run produces the .npz only. This draws the
figures from that file afterwards, on the laptop.

Two figures:
  <stem>_diagonal.png   rho along the line x = y, numerical vs the closed-form
                        reference, plus their difference.
  <stem>_residuals.png  the two LADMM residual histories on a log axis.

Usage:
    python3 postproc_2d_npz.py <path-to.npz> [--n-snapshots 5]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", type=Path, help="the .npz written by the solver script")
    p.add_argument("--n-snapshots", type=int, default=5,
                   help="how many time slices to draw in the diagonal figure")
    return p.parse_args()


def plot_diagonal(d, idx, path):
    """rho along x = y. See the solver script's own version for why this cut:
    the transport runs along this line, and it is aligned with neither grid
    axis, so an error in either axis' DCT eigenvalue shows up rather than
    cancelling."""
    rho_num, rho_ana, xx = d["rho_stag"], d["rho_ana"], d["xx"]
    t_rho = d["t_edges"][1:-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(idx)))
    for c, k in zip(colors, idx):
        dn, da = np.diagonal(rho_num[k]), np.diagonal(rho_ana[k])
        axes[0].plot(xx, dn, color=c, lw=1.8, label=f"t = {t_rho[k]:.4f}")
        axes[0].plot(xx, da, color=c, lw=1.1, ls="--")
        axes[1].plot(xx, dn - da, color=c, lw=1.4, label=f"t = {t_rho[k]:.4f}")

    axes[0].set_title("rho along x = y  (solid: numerical, dashed: analytical)")
    axes[0].set_ylabel("rho(t, x, x)")
    axes[1].set_title("numerical - analytical, along x = y")
    axes[1].set_ylabel("difference")
    axes[1].axhline(0.0, color="k", lw=0.6)
    for ax in axes:
        ax.set_xlabel("x  (along the line x = y)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{d['grid']} grid, nt={d['nt']} nx={d['nx']} ny={d['ny']}, "
        f"eps={d['vareps']:g}, sigma={d['sigma']:g}, {d['iters']} iters "
        f"({'converged' if d['converged'] else 'NOT converged'})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(d, path):
    """The two LADMM residual histories.

    NAMING, because ladmm.py's own comments invert the usual convention and a
    mislabelled axis is the kind of thing that reaches a draft: in Boyd's
    standard ADMM terminology the CONSTRAINT VIOLATION ||A x + B y - b|| is
    the PRIMAL residual and the ||tau*(x-) + gamma*A^T*(D-)|| combination is
    the DUAL residual. ladmm.py stores the former as `D` (calling it "dual"
    in its docstring) and the latter as `P` (calling it "primal"). The labels
    below follow Boyd and print the formula, so the curve is identified by
    its mathematics rather than by either name.
    """
    D, P = d["D"], d["P"]
    it = np.arange(1, len(D) + 1)
    ms_per_iter = float(d["walltime"]) / float(d["iters"]) * 1e3

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(it, D, lw=1.6, color="#1f77b4",
                label=r"primal  $\|Ax^k + By^k - b\|$   (constraint violation; stored as `D`)")
    ax.semilogy(it, P, lw=1.6, color="#d62728",
                label=r"dual  $\|\tau(x^{k-1}-x^k) + \gamma A^{T}(D^{k-1}-D^k)\|$   (stored as `P`)")

    ax.set_xlabel("LADMM iteration")
    ax.set_ylabel("residual norm (grid-weighted $L^2$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper right")

    # Second x-axis in wall-clock seconds -- "vs time" read the other way.
    top = ax.secondary_xaxis("top", functions=(lambda i: i * ms_per_iter / 1e3,
                                               lambda s: s * 1e3 / ms_per_iter))
    top.set_xlabel(f"wall-clock (s) on {d['backend']}, at {ms_per_iter:.1f} ms/iter")

    ax.set_title(
        f"LADMM convergence -- {d['grid']} grid, nt={d['nt']} nx={d['nx']} ny={d['ny']}, "
        f"eps={d['vareps']:g}\n"
        f"{d['iters']} iters, {'converged' if d['converged'] else 'NOT converged (iteration cap)'}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    d = np.load(args.npz, allow_pickle=True)
    stem = args.npz.with_suffix("")

    ntm = d["rho_stag"].shape[0]
    idx = np.unique(np.linspace(0, ntm - 1, min(args.n_snapshots, ntm)).round().astype(int))

    plot_diagonal(d, idx, f"{stem}_diagonal.png")
    print(f"saved {stem}_diagonal.png")
    plot_residuals(d, f"{stem}_residuals.png")
    print(f"saved {stem}_residuals.png")

    D, P = d["D"], d["P"]
    print(f"\nresiduals: D {D[0]:.3e} -> {D[-1]:.3e}   P {P[0]:.3e} -> {P[-1]:.3e}"
          f"   ({D[0]/D[-1]:.1f}x and {P[0]/P[-1]:.1f}x over {d['iters']} iters)")
    rel = np.linalg.norm(d["rho_stag"] - d["rho_ana"]) / np.linalg.norm(d["rho_ana"])
    print(f"rho relative error vs analytical SB: {rel:.3e}")


if __name__ == "__main__":
    main()
