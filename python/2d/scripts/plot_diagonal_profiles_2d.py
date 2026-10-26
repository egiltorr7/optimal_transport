"""rho along the line x = y, numerical vs the closed-form reference.

Same cut and the same solid/dashed convention as postproc_2d_npz.py and
sb_gaussian_nonuniform_2d.py's plot_diagonal_profiles, and for their reason:
both marginal centres sit on x = y, so the transport runs along this line and
the diagonal carries essentially the whole solution -- while being aligned
with NEITHER grid axis, so an error in either axis' DCT eigenvalue shows up
here instead of cancelling.

Reads the .npz that etd_eps_sweep_2d.py already wrote (it stores the full
rho_stag and rho_a arrays, not just scalars), so this draws without re-solving.

Time is an ORDINAL variable here, so the snapshots use a single-hue ramp
light -> dark with increasing t, rather than the multi-hue viridis the older
scripts use: a rainbow implies category changes between steps that an ordered
sequence does not have, and its mid-range steps are not ordered by lightness.

Note rho(t, x, x) is a SLICE of the 2D density, not a 1D marginal -- it does
not integrate to 1, and its peak is the 2D peak (~25 here), not a 1D one.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
TIME_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # light -> dark = early -> late
RES = Path(__file__).resolve().parent.parent / "results"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#d5d4d0")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=RES / "etd_2d_eps_sweep.npz")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--grids", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--n-snapshots", type=int, default=5)
    ap.add_argument("--by", choices=("grid", "eps"), default="grid",
                    help="what the columns vary: grid resolution at fixed eps "
                         "(default), or eps at fixed grid")
    ap.add_argument("--eps-list", type=float, nargs="+", default=None,
                    help="with --by eps: the eps values to face")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    d = np.load(args.npz)
    ns = args.n_snapshots

    if args.by == "grid":
        cols = [(n, args.eps, f"n = {n}") for n in args.grids]
    else:
        eps_list = args.eps_list if args.eps_list is not None else [args.eps]
        n_fix = args.grids[-1]
        cols = [(n_fix, e, rf"$\varepsilon$ = {e:g}") for e in eps_list]

    fig, axes = plt.subplots(2, len(cols), figsize=(4.6 * len(cols), 7.2),
                             facecolor=SURFACE, squeeze=False)
    for col, (n, eps_c, col_label) in enumerate(cols):
        rho = d[f"rho|{eps_c:g}|{n}"]         # (ntm, nx, ny)
        ana = d[f"rho_a|{eps_c:g}|{n}"]
        ntm, nx, _ = rho.shape
        dt, dx = 1.0 / n, 1.0 / n
        xx = (np.arange(nx) + 0.5) * dx
        t_rho = (np.arange(1, ntm + 1)) * dt          # interior time edges
        idx = np.unique(np.linspace(0, ntm - 1, ns).round().astype(int))

        ax_p, ax_e = axes[0][col], axes[1][col]
        _style(ax_p)
        _style(ax_e)
        for c, k in zip(TIME_RAMP, idx):
            dn, da = np.diagonal(rho[k]), np.diagonal(ana[k])
            # Reference as the LINE, numerical as MARKERS on top. The older
            # solid/dashed pair is unreadable here: the two agree to <1% of
            # peak, so the dashed reference vanishes under the solid numerical
            # curve and the figure shows one curve instead of a comparison.
            ax_p.plot(xx, da, color=c, linewidth=1.6, zorder=2)
            ax_p.plot(xx, dn, linestyle="none", marker="o", markersize=4.5,
                      markerfacecolor="none", markeredgewidth=1.3,
                      markeredgecolor=c, zorder=3, label=f"t = {t_rho[k]:.3f}")
            ax_e.plot(xx, dn - da, color=c, linewidth=1.8, marker="o",
                      markersize=3, markerfacecolor="none")
        ax_e.axhline(0.0, color=INK2, linewidth=0.8, linestyle=":")
        ax_p.set_title(col_label, fontsize=10)
        ax_p.set_ylabel(r"$\rho(t,\,x,\,x)$")
        ax_e.set_ylabel("numerical $-$ closed form")
        for ax in (ax_p, ax_e):
            ax.set_xlabel("x   (along the line $x=y$)")
        if col == 0:
            ax_p.legend(frameon=False, fontsize=8, labelcolor=INK2, title="line: closed form\nmarkers: numerical")
            ax_p.get_legend().get_title().set_color(INK2)
            ax_p.get_legend().get_title().set_fontsize(8)

    # Shared y-limits per row ONLY when the columns are the same problem at
    # different resolutions. Faceting by eps, the peak density varies by more
    # than 2x (diffusion flattens the bridge), so a shared scale would render
    # the large-eps panels as flat lines.
    if args.by == "grid":
        for row in axes:
            lo = min(ax.get_ylim()[0] for ax in row)
            hi = max(ax.get_ylim()[1] for ax in row)
            for ax in row:
                ax.set_ylim(lo, hi)

    head = (rf"$\varepsilon$ = {args.eps:g}" if args.by == "grid"
            else f"n = {cols[0][0]}")
    fig.suptitle(rf"$\rho$ along $x=y$, ETD, {head}  "
                 r"$-$ the transport runs along this line, and it is aligned with neither grid axis",
                 fontsize=10, color=INK)
    fig.tight_layout()
    out = RES / (f"diagonal_profiles_2d_eps{args.eps:g}{args.tag}.png" if args.by == "grid"
                 else f"diagonal_profiles_2d_byeps_n{cols[0][0]}{args.tag}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")

    for n, eps_c, _lbl in cols:
        rho, ana = d[f"rho|{eps_c:g}|{n}"], d[f"rho_a|{eps_c:g}|{n}"]
        dn = np.stack([np.diagonal(rho[k]) for k in range(rho.shape[0])])
        da = np.stack([np.diagonal(ana[k]) for k in range(ana.shape[0])])
        print(f"  n={n:3d} eps={eps_c:<7g} diagonal slice: max|num-ana| = {np.abs(dn - da).max():.4e}"
              f"   peak rho = {da.max():.3f}   rel = {np.abs(dn - da).max() / da.max():.3e}")


if __name__ == "__main__":
    main()
