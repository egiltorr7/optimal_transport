"""Density evolution from run_suite_2d.py's data. Post-processing only.

Three figures per (vareps, n):

  waterfall   rho along the diagonal x = y as an (s, t, rho) waterfall, s the
              arc length along the diagonal (s = sqrt(2) x), each selected time
              drawn at its true t -- the 2D analogue of post_density_profiles.py's
              x-t waterfall. The transport runs along this diagonal (both
              centres sit on x = y) so it is the informative cut. The reference
              is dashed. Times are picked by the same rule as the 1D script
              (uniform in t, or cosine-clustered to both ends once
              sigma^2/vareps < 1/2) from the rho_diag rows saved at EVERY time.
  diagonal    the same diagonal profiles at ~7 times on ONE 2D axes, LADMM solid
              and reference dashed per time, with a LADMM - reference panel
              underneath: the direct visual comparison, and where it goes wrong.
  snapshots   the full (x, y) plane at the representative times the runner
              stored (snap_rho), filled contours of the solution with the
              reference's contour lines over it.

Usage:
    python scripts/post_density_profiles_2d.py                 # n=256, every vareps
    python scripts/post_density_profiles_2d.py --n 64 --eps 1e-4 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SUITE = Path(__file__).resolve().parent.parent / "results" / "suite"
OUT = Path(__file__).resolve().parent.parent / "results" / "post"
EPS_LIST = [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SIGMA = 0.05
N_TIMES = 19
ELEV, AZIM = 28, -50
PLOT_REFS = ("analytic", "sinkhorn_same")


def fmt_eps(vareps):
    if vareps == 0:
        return "0"
    if vareps < 0.01:
        return rf"10^{{{int(round(np.log10(vareps)))}}}"
    return f"{vareps:g}"


def tag(vareps, n):
    grid = "clustered" if vareps >= 1e-2 else "uniform"
    return f"gaussian_eps{vareps:g}_n{n}_{grid}_ETD"


def pick_times(t_grid, vareps, n_times=N_TIMES):
    t_diff = np.inf if vareps <= 0 else SIGMA**2 / vareps
    if t_diff >= 0.5:
        want = np.linspace(t_grid[0], t_grid[-1], n_times)
    else:
        u = np.linspace(0.0, 1.0, n_times)
        want = t_grid[0] + (t_grid[-1] - t_grid[0]) * 0.5 * (1 - np.cos(np.pi * u))
    return np.unique([int(np.argmin(np.abs(t_grid - w))) for w in want]), t_diff


def pick_ref(d):
    names = [str(r) for r in d["ref_names"]]
    return next((r for r in PLOT_REFS if r in names), names[0] if names else None)


def waterfall(d, vareps, n, out):
    rho_diag, xx = d["rho_diag"], d["xx"]
    t_grid = d["t_edges"][1:-1]
    idx, t_diff = pick_times(t_grid, vareps)
    s = np.sqrt(2.0) * xx
    ref = pick_ref(d)
    ref_diag = d[f"ref_{ref}_rho_diag"] if ref else None

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(np.linspace(0, 1, len(idx)))
    for k, c in zip(idx, colors):
        t = t_grid[k]
        ax.plot(s, np.full_like(s, t), rho_diag[k], color=c, lw=1.3)
        if ref_diag is not None:
            ax.plot(s, np.full_like(s, t), ref_diag[k], color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("s along x = y"); ax.set_ylabel("t"); ax.set_zlabel(r"$\rho$")
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(rf"gaussian 2D, $\varepsilon = {fmt_eps(vareps)}$, n={n}: $\rho$ on the diagonal"
                 + (f"  (dashed: {ref})" if ref else "") + f"\n{len(idx)} times, "
                 + ("uniform in t" if t_diff >= 0.5 else rf"cosine-clustered ($\sigma^2/\varepsilon$={t_diff:.2g})"),
                 fontsize=10)
    fig.tight_layout()
    p = out / f"waterfall2d_gaussian_eps{vareps:g}_n{n}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {p.name}")


N_TIMES_OVERLAY = 7


def diagonal_overlay(d, vareps, n, out, n_times=N_TIMES_OVERLAY):
    """One 2D axes: rho on the diagonal x = y at a handful of times, LADMM solid
    and the reference dashed in the same colour, each pair labelled by its t --
    plus a lower panel with the pointwise difference, which is where a
    discrepancy that is invisible at the scale of rho shows up (and which
    time it belongs to). Fewer times than the waterfall on purpose: overlaid
    curves get unreadable past 7 or so."""
    rho_diag, xx = d["rho_diag"], d["xx"]
    t_grid = d["t_edges"][1:-1]
    idx, t_diff = pick_times(t_grid, vareps, n_times)
    s = np.sqrt(2.0) * xx
    ref = pick_ref(d)
    ref_diag = d[f"ref_{ref}_rho_diag"] if ref else None

    fig, (ax, axd) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.4]})
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(idx)))
    # Reference as a solid line, LADMM as hollow markers on every `every`-th
    # grid point (n=256 has too many points to mark them all).
    every = max(1, len(s) // 40)
    for k, c in zip(idx, colors):
        t = t_grid[k]
        if ref_diag is not None:
            ax.plot(s, ref_diag[k], color=c, lw=1.4, label=f"t = {t:.3f}")
            axd.plot(s, rho_diag[k] - ref_diag[k], color=c, lw=1.2)
        ax.plot(s[::every], rho_diag[k][::every], color=c, ls="none", marker="o", ms=4,
                mfc="none", mew=1.0, label=None if ref_diag is not None else f"t = {t:.3f}")
    if ref:
        ax.plot([], [], color="0.3", lw=1.4, label=f"{ref} (line)")
    ax.plot([], [], color="0.3", ls="none", marker="o", ms=4, mfc="none", label="LADMM (markers)")
    ax.set_ylabel(r"$\rho$ on x = y")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title(rf"gaussian 2D, $\varepsilon = {fmt_eps(vareps)}$, n={n}: diagonal profiles, "
                 + ("uniform in t" if t_diff >= 0.5 else rf"cosine-clustered times ($\sigma^2/\varepsilon$={t_diff:.2g})"),
                 fontsize=10)
    axd.axhline(0, color="0.5", lw=0.8)
    axd.set_xlabel("s along x = y   (s = sqrt(2) x)")
    axd.set_ylabel(r"LADMM $-$ ref")
    axd.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out / f"diagonal2d_gaussian_eps{vareps:g}_n{n}.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  wrote {p.name}")


def snapshots(d, vareps, n, out):
    snap, idx, xx = d["snap_rho"], d["snap_idx"], d["xx"]
    t_grid = d["t_edges"][1:-1]
    ref = pick_ref(d)
    ref_snap = d[f"ref_{ref}_snap_rho"] if ref else None
    k = len(idx)
    ncol = min(5, k)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow), squeeze=False)
    X, Y = np.meshgrid(xx, xx, indexing="ij")
    for ax, i, r in zip(axes.flat, range(k), snap):
        # Per-panel colour scale: at large vareps the interior states are
        # near-uniform at rho ~ 1 while the endpoints peak at ~64, and a shared
        # scale renders every interior panel as one flat colour.
        vmax = float(max(r.max(), 1e-300))
        ax.contourf(X, Y, r, levels=np.linspace(0, vmax, 21), cmap="viridis", extend="both")
        if ref_snap is not None:
            ax.contour(X, Y, ref_snap[i], levels=np.linspace(0.1 * vmax, vmax, 6), colors="w", linewidths=0.6)
        ax.plot([0, 1], [0, 1], color="w", lw=0.4, ls=":", alpha=0.6)
        ax.set_title(f"t = {t_grid[idx[i]]:.3f}   max {vmax:.3g}, min {float(r.min()):.2g}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[k:]:
        ax.axis("off")
    fig.suptitle(rf"gaussian 2D, $\varepsilon = {fmt_eps(vareps)}$, n={n}: $\rho(t,x,y)$"
                 + (f"  (white contours: {ref})" if ref else ""), fontsize=10)
    fig.tight_layout()
    p = out / f"snapshots2d_gaussian_eps{vareps:g}_n{n}.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  wrote {p.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite", default=str(SUITE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n", type=int, nargs="+", default=[256])
    ap.add_argument("--eps", type=float, nargs="+", default=EPS_LIST)
    a = ap.parse_args()
    suite, out = Path(a.suite), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for vareps in a.eps:
        for n in a.n:
            f = suite / f"{tag(vareps, n)}.npz"
            if not f.exists():
                print(f"  missing {f.name}")
                continue
            d = np.load(f, allow_pickle=True)
            waterfall(d, vareps, n, out)
            diagonal_overlay(d, vareps, n, out)
            snapshots(d, vareps, n, out)


if __name__ == "__main__":
    main()
