"""Grid-convergence plots from the run_suite.py data. Post-processing only.

Reads results/suite/*.npz and plots the final-iterate density error against
resolution, one panel per vareps, with O(h) and O(h^2) guides and a fitted
order. Re-runs nothing.

THREE error metrics are plotted, deliberately, because on this problem they do
not agree and the difference is the interesting part:

  slab      space-time weighted L2 over the whole slab. This is the headline
            number, and at large vareps it is dominated by the t=1 boundary
            layer -- the terminal FP constraint row is O(dt) while the interior
            is O(dt^2), so one row can set the "order" the plot reports.
  interior  the same, with the first and last TIME rows dropped. This is the
            scheme's own order, with the boundary layer excluded.
  max_t     max over time of the per-time L2(x). Tracks the worst row, so it
            follows the boundary layer almost exactly.

Reading slab and interior together tells you whether a given vareps is
boundary-limited or genuinely limited by the interior discretization.

Above SINKHORN_VAREPS_THRESHOLD two references are available and BOTH are
plotted, since they measure different things (docs/sinkhorn_reference.tex):
  sinkhorn_same   fitted on the run's own nx -- spatially consistent with the
                  solver, so it isolates the TIME error.
  sinkhorn_fine   fitted at nx=4096 + spectral interpolant -- approximates the
                  continuum, so it measures TOTAL discretization error.
Below the threshold the closed-form "analytic" reference is the only one.

Usage:
    python scripts/post_grid_convergence.py                 # gaussian (default)
    python scripts/post_grid_convergence.py --case bimodal
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

SUITE = Path(__file__).resolve().parent.parent / "results" / "suite"
OUT = Path(__file__).resolve().parent.parent / "results" / "post"
EPS_LIST = [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
RESOLUTIONS = [32, 64, 128, 256]
# Which error norm to plot. All three are still COMPUTED (see errors() and the
# printed table) -- this only picks the one drawn.
#   slab      every time row, weighted by its time-quadrature weight. The actual
#             total error of the solution; the default because it needs no
#             justification and, on the clustered grid, differs from `interior`
#             by <0.5% except at vareps=1, n<=64 (the boundary rows carry only
#             0.02-0.96% of the total time weight, so the clustering already
#             suppresses the t=1 layer).
#   interior  same, dropping the first and last TIME rows -- excludes the t=1
#             boundary layer to expose the interior scheme order.
#   max_t     max over time of the per-time L2(x); tracks the worst row.
METRIC = "slab"

# Which reference to draw. Both are still computed and tabulated.
PLOT_REFS = ("analytic", "sinkhorn_same")

METRICS = ("slab", "interior", "max_t")
REF_COLOR = {"analytic": "tab:blue", "sinkhorn_same": "tab:orange",
             "sinkhorn_fine": "tab:green"}
GUIDE_OFFSET = 0.40   # h^2 guide's closest approach, as a fraction of the data


def fmt_eps(vareps):
    """Powers of ten in scientific notation below 1e-2, plain decimals above.

    "1e-08" and "0.0001" read badly next to each other in a legend; every
    vareps below the threshold here is an exact power of ten, so 10^{-8} is
    both shorter and easier to scan."""
    if vareps == 0:
        return "0"
    if vareps < 0.01:
        return rf"10^{{{int(round(np.log10(vareps)))}}}"
    return f"{vareps:g}"


def tag(case, vareps, n):
    grid = "clustered" if vareps >= 1e-2 else "uniform"
    return f"{case}_eps{vareps:g}_n{n}_{grid}_ETD"


def row_weights(d):
    """Time-quadrature weight per staggered rho row (see pipeline_nu's
    _weighted_sq_sum): rho_stag lives on the nt-1 interior interfaces, whose
    natural weight is the centre-to-centre dual width. Constant dt reduces this
    to plain dt, so the uniform runs are handled by the same expression."""
    dt_vec = d["dt_vec"]
    return 0.5 * (dt_vec[:-1] + dt_vec[1:])


def errors(d, ref_name):
    """{metric: value} for the final iterate against one reference."""
    rho, ref = d["rho_stag"], d[f"ref_{ref_name}_rho_stag"]
    dx, w = float(d["dx"]), row_weights(d)
    per_row_sq = dx * np.sum((rho - ref) ** 2, axis=1)      # (ntm,)
    return {
        "slab": float(np.sqrt(np.sum(w * per_row_sq))),
        "interior": float(np.sqrt(np.sum(w[1:-1] * per_row_sq[1:-1]))),
        "max_t": float(np.sqrt(per_row_sq.max())),
    }


def fit_order(ns, errs):
    """Least-squares slope of log(err) vs log(n), negated -> order in h."""
    ns, errs = np.asarray(ns, float), np.asarray(errs, float)
    ok = errs > 0
    if ok.sum() < 2:
        return np.nan
    return -np.polyfit(np.log(ns[ok]), np.log(errs[ok]), 1)[0]


def collect(case):
    """{vareps: {ref: {metric: [err per n]}}} plus the per-run diagnostics."""
    data, diag = {}, {}
    for vareps in EPS_LIST:
        per_ref, rows = {}, []
        for n in RESOLUTIONS:
            f = SUITE / f"{tag(case, vareps, n)}.npz"
            if not f.exists():
                rows.append(None)
                continue
            d = np.load(f, allow_pickle=True)
            rows.append((int(d["iters"]), bool(d["converged"]),
                         float(d["D"][-1]), float(d["P"][-1])))
            for ref in [r for r in d["ref_names"]]:
                e = errors(d, ref)
                for m in METRICS:
                    per_ref.setdefault(ref, {}).setdefault(m, []).append(e[m])
        data[vareps] = per_ref
        diag[vareps] = rows
    return data, diag


def panel(ax, vareps, per_ref):
    ax.set_xscale("log"); ax.set_yscale("log")
    # Plain log-log with ticks ONLY at the resolutions actually run: the default
    # LogLocator adds labelled minor ticks (3x10^1, 4x10^1, ...) that collide
    # with them.
    ax.xaxis.set_major_locator(FixedLocator(RESOLUTIONS))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())

    shown = {k: v for k, v in per_ref.items() if k in PLOT_REFS}
    if not shown:
        ax.text(0.5, 0.5, "no reference\n(bimodal below 5e-3)", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="0.4")
        ax.set_title(f"$\\varepsilon$ = {vareps:g}")
        return

    ns = np.array(RESOLUTIONS, float)
    last = None
    for ref, per_metric in sorted(shown.items()):
        e = np.array(per_metric[METRIC], float)
        if e.size != ns.size:
            continue
        ax.plot(ns, e, "-o", color=REF_COLOR.get(ref, "0.3"), ms=5, lw=1.6,
                label="LADMM")
        last = e

    # Second-order guide, scaled so its CLOSEST approach to the data is a fixed
    # factor below it. Anchoring at a data point instead makes the two overlap
    # there, and anchoring at a fixed offset from one end still lets them cross
    # whenever the data is steeper than h^2 (which it is at small vareps). This
    # way the gap is guaranteed at every n, whatever the data's slope, while
    # staying close enough for the slopes to be compared.
    if last is not None:
        unit = (ns[0] / ns) ** 2
        ax.plot(ns, GUIDE_OFFSET * np.min(last / unit) * unit,
                linestyle=(0, (4, 3)), color="0.45", lw=1.2,
                label=r"$O(h^2)=O(n^{-2})$")

    ax.set_title(rf"$\varepsilon$ = ${fmt_eps(vareps)}$")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="gaussian")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data, diag = collect(a.case)

    print(f"=== {a.case}: fitted order in h (n = {RESOLUTIONS}) ===")
    hdr = f"  {'vareps':>8} {'reference':>15} " + " ".join(f"{m:>9}" for m in METRICS)
    print(hdr + "   |  final D (per n)")
    for vareps in EPS_LIST:
        if not data[vareps]:
            print(f"  {vareps:8g} {'(none)':>15}")
            continue
        for ref, per_metric in sorted(data[vareps].items()):
            orders = [fit_order(RESOLUTIONS, per_metric[m]) for m in METRICS]
            line = (f"  {vareps:8g} {ref:>15} "
                    + " ".join(f"{o:9.2f}" for o in orders))
            if ref == sorted(data[vareps])[0]:
                line += "   |  " + " ".join(
                    f"{r[2]:.1e}" for r in diag[vareps] if r)
            print(line)

    # per-eps figures
    for vareps in EPS_LIST:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        panel(ax, vareps, data[vareps])
        ax.set_title("")          # the suptitle below already names case + vareps
        ax.set_xlabel("n  (nt = nx)")
        ax.set_ylabel(r"$\|\rho-\rho_{ref}\|$")
        fig.suptitle(rf"{a.case}: grid convergence, $\varepsilon = {fmt_eps(vareps)}$",
                     y=0.99)
        fig.tight_layout()
        p = OUT / f"gridconv_{a.case}_eps{vareps:g}.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"  wrote {p.name}")

    # all vareps on one axes
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(RESOLUTIONS))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())

    ns = np.array(RESOLUTIONS, float)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(EPS_LIST)))
    finest = []
    for vareps, c in zip(EPS_LIST, colors):
        shown = {k: v for k, v in data[vareps].items() if k in PLOT_REFS}
        if not shown:
            continue
        ref = sorted(shown)[0]
        e = np.array(shown[ref][METRIC], float)
        if e.size != ns.size:
            continue
        # marker distinguishes which reference the curve is measured against,
        # since that switches with vareps at SINKHORN_VAREPS_THRESHOLD
        mk = "o" if ref == "analytic" else "s"
        ax.plot(ns, e, "-", marker=mk, color=c, ms=5, lw=1.6,
                label=rf"$\varepsilon = {fmt_eps(vareps)}$")
        finest.append(e[-1])

    if finest:
        # lifted clear of the family rather than through its topmost point
        anchor = max(finest) * 1.9
        ax.plot(ns, anchor * (ns[-1] / ns) ** 2, linestyle=(0, (4, 3)),
                color="0.45", lw=1.3, label=r"$O(h^2)=O(n^{-2})$")

    ax.set_xlabel("n  (nt = nx)")
    ax.set_ylabel(r"$\|\rho-\rho_{ref}\|$")
    # (mathtext has no \square, so say it in words)
    ax.set_title(f"{a.case}: grid convergence\n"
                 "circles = vs closed form,  squares = vs Sinkhorn", fontsize=10)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    p = OUT / f"gridconv_{a.case}_all.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p.name}")


if __name__ == "__main__":
    main()
