"""Re-draw scheme_grid_eps_compare_2d.py's figure from its .npz.

Two fixes over the first draft, both found by looking at the render:
  * min(rho)/peak was on a LINEAR axis, which crushed every eps <= 1e-2 onto
    zero and showed only the eps=1e-1 dive. It spans 5 decades, so it needs a
    log axis on |min rho| (every value is negative; the sign is stated in the
    label rather than drawn).
  * the four series overlap to within the line width on the error panels --
    the differences between CN/ETD and uniform/clustered are ~1%, which is the
    actual result but was rendered as occlusion. Panel 4 plots each series
    relative to CN-uniform so those percentages are readable.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
COLOR = {"CN": "#2a78d6", "ETD": "#eb6834"}
STYLE = {"uniform": "-", "clustered": "--"}
MARK = {"uniform": "o", "clustered": "s"}
EPSS = [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1]
N = 24

RES = Path(__file__).resolve().parent.parent / "results"
d = np.load(RES / f"scheme_grid_eps_compare_2d_n{N}.npz")
g = lambda s, gr, e, k: float(d[f"{s}|{gr}|{e:g}|{k}"])
x = [max(e, 1e-9) for e in EPSS]


def _style(ax):
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
    ax.set_xlabel(r"$\varepsilon$    ($\varepsilon=0$ drawn at $10^{-9}$)")


SERIES = [(s, gr) for s in ("CN", "ETD") for gr in ("uniform", "clustered")]
fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.0), facecolor=SURFACE)

for ax, (key, ylab, logy) in zip(axes, [
    ("e_rho", r"relative error in $\rho$", True),
    ("e_mx", r"relative error in $m_x$", True),
    ("min_rho_abs", r"$|\min\rho| / \rho_{\rm peak}$   (all values negative)", True),
]):
    _style(ax)
    if logy:
        ax.set_yscale("log")
    for s, gr in SERIES:
        if key == "min_rho_abs":
            ys = [abs(g(s, gr, e, "min_rho")) / g(s, gr, e, "peak") for e in EPSS]
        else:
            ys = [g(s, gr, e, key) for e in EPSS]
        ax.plot(x, ys, STYLE[gr], marker=MARK[gr], markersize=6, linewidth=2,
                color=COLOR[s], markerfacecolor="none" if gr == "clustered" else None,
                label=f"{s}, {gr}")
    ax.set_ylabel(ylab)

ax = axes[3]
_style(ax)
for s, gr in SERIES:
    if (s, gr) == ("CN", "uniform"):
        continue
    ys = [g(s, gr, e, "e_rho") / g("CN", "uniform", e, "e_rho") for e in EPSS]
    ax.plot(x, ys, STYLE[gr], marker=MARK[gr], markersize=6, linewidth=2,
            color=COLOR[s], markerfacecolor="none" if gr == "clustered" else None,
            label=f"{s}, {gr}")
ax.axhline(1.0, color=INK2, linewidth=1.0, linestyle=":")
ax.set_ylabel(r"$\rho$ error relative to CN-uniform")
ax.set_title("below 1.0 = better than CN-uniform", fontsize=9)

axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
axes[3].legend(frameon=False, fontsize=8, labelcolor=INK2)
fig.suptitle(f"CN vs ETD, uniform vs clustered, n={N}, 8000 iterations (none converged). "
             r"At $\varepsilon=10^{-1}$ the closed-form reference is invalid "
             "(9.6% of peak density at the wall), so read that column as scheme-vs-scheme only.",
             fontsize=9.5, color=INK)
fig.tight_layout()
out = RES / f"scheme_grid_eps_compare_2d_n{N}.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"saved {out}\n")

hdr = f"{'eps':>8s} | " + " | ".join(f"{s+','+gr[:4]:>16s}" for s, gr in SERIES)
print("rho error" + "\n" + hdr)
for e in EPSS:
    print(f"{e:>8g} | " + " | ".join(f"{g(s,gr,e,'e_rho'):>16.4e}" for s, gr in SERIES))
print("\nmin(rho)/peak" + "\n" + hdr)
for e in EPSS:
    print(f"{e:>8g} | " + " | ".join(
        f"{g(s,gr,e,'min_rho')/g(s,gr,e,'peak'):>16.2e}" for s, gr in SERIES))
