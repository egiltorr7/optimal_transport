"""Re-draw etd_eps_sweep_2d.py's vs-grid figure from its saved .npz.

Separate from the sweep so the figure can be fixed without re-running 9 solves.
Three panels rather than two, because the first draft hid its own result: the
three vareps curves overlap to within the line width (at sigma=0.08 the SB's
alpha = sqrt(sigma^4+eps^2)-sigma^2 is ~1.2e-7 even at eps=1e-4, so all three
are numerically the same problem), which made the eps=0 and eps=1e-8 series
invisible under eps=1e-4. Panel 3 plots the ratio to eps=0 so that overlap is
shown as a measurement instead of hidden as an occlusion.

Also: the reference slope is anchored to the SOLVER's coarsest point, not to
the reference floor. Anchoring it to the floor made it coincide with the floor
line -- two lines on top of each other testing nothing, since the floor is
itself O(h^2) by construction (validate_expsemi_2d.py check 5).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
EPS_CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]

RES = Path(__file__).resolve().parent.parent / "results"
d = np.load(RES / "etd_2d_eps_sweep.npz")
EPSS = [0.0, 1e-8, 1e-4]
GRIDS = [16, 24, 32]
g = lambda k, e, n: float(d[f"{k}|{e:g}|{n}"])


def _style(ax, xt):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#d5d4d0")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    ax.set_xscale("log")
    ax.set_xticks(xt)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())   # kills the 3x10^1 collision
    ax.set_xlabel("n   (nt = nx = ny)")


fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), facecolor=SURFACE)

for ax, (key, lbl) in zip(axes[:2], [("e_rho", r"$\rho$"), ("e_mx", r"$m_x$")]):
    _style(ax, GRIDS)
    ax.set_yscale("log")
    for k, e in enumerate(EPSS):
        ys = [g(key, e, n) for n in GRIDS]
        ax.plot(GRIDS, ys, marker=MARKERS[k], markersize=7, linewidth=2,
                color=EPS_CAT[k], label=f"$\\varepsilon$ = {e:g}",
                markerfacecolor="none" if k else None, zorder=3 + k)
    ys0 = [g(key, EPSS[0], n) for n in GRIDS]
    gg = np.asarray(GRIDS, float)
    ax.plot(gg, ys0[0] * (gg / gg[0]) ** -2.0, ":", linewidth=1.4, color=INK2,
            zorder=2, label=r"slope $-2$ from the coarsest point")
    fl = [g("floor", e, n) if f"floor|{e:g}|{n}" in d.files else np.nan
          for e, n in zip([EPSS[0]] * 3, GRIDS)]
    if not np.isnan(fl).any():
        ax.plot(GRIDS, fl, "--", linewidth=1.5, color="#a8a7a2", zorder=1,
                label="reference's own FP residual")
    ax.set_ylabel(f"relative error in {lbl}")

# panel 3: the eps dependence the overlap hides
ax = axes[2]
_style(ax, GRIDS)
for k, e in enumerate(EPSS[1:], start=1):
    for key, ls in [("e_rho", "-"), ("e_mx", "--")]:
        ys = [g(key, e, n) / g(key, 0.0, n) for n in GRIDS]
        ax.plot(GRIDS, ys, ls, marker=MARKERS[k], markersize=7, linewidth=2,
                color=EPS_CAT[k], markerfacecolor="none",
                label=f"{'ρ' if key == 'e_rho' else 'mx'},  $\\varepsilon$={e:g}")
ax.axhline(1.0, color=INK2, linewidth=1.0, linestyle=":")
ax.set_ylabel(r"error($\varepsilon$) / error($\varepsilon=0$)")
ax.set_title("eps-dependence (1.0 = same as pure OT)", fontsize=9)

axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")
axes[2].legend(frameon=False, fontsize=7.5, labelcolor=INK2)
fig.suptitle("ETD-2D vs the closed form. n=32 is still under-converged at 20k iterations "
             "(see the convergence figure)", fontsize=9.5, color=INK)
fig.tight_layout()
out = RES / "etd_2d_eps_sweep_vs_grid.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"saved {out}")

for key in ("e_rho", "e_mx"):
    print(f"\n{key}:")
    for e in EPSS:
        ys = [g(key, e, n) for n in GRIDS]
        ords = [np.log(ys[i] / ys[i + 1]) / np.log(GRIDS[i + 1] / GRIDS[i]) for i in range(2)]
        print(f"  eps={e:<8g} " + "  ".join(f"n={n}: {y:.4e}" for n, y in zip(GRIDS, ys))
              + f"   orders {ords[0]:.2f}, {ords[1]:.2f}")
