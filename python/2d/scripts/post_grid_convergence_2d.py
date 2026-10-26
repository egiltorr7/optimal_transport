"""Grid-convergence plots from run_suite_2d.py's data. Post-processing only.

2D sibling of scripts/post_grid_convergence.py, same three metrics built from
the per-time-row errors the runner saved (rows_<ref>_rho: dx*dy*sum over the
plane of (rho-ref)^2, one value per staggered time row):

  slab      sqrt(sum_rows w_row * rows)   -- whole space-time slab, w_row the
            time-quadrature weight of the row (dual width on the clustered
            grid, dt on the uniform one). The headline number.
  interior  the same with the first and last time rows dropped.
  max_t     sqrt(max_rows rows)           -- worst time row.

References: "analytic" below SINKHORN_VAREPS_THRESHOLD (5e-3), "sinkhorn_same"
and "sinkhorn_fine" above it (product of two 1D Sinkhorn bridges, exact for
this problem -- see run_suite_2d.py). Both are tabulated; PLOT_REFS picks
what is drawn.

Also draws the per-iteration density error against the reference, one panel
per vareps and one curve per n, which is what says whether 40k iterations
converged (the 1D suite reads that off residual_history.py).

Usage:
    python scripts/post_grid_convergence_2d.py
    python scripts/post_grid_convergence_2d.py --suite /path/to/suite --n 32 64
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

SUITE = Path(__file__).resolve().parent.parent / "results" / "suite"
OUT = Path(__file__).resolve().parent.parent / "results" / "post"
EPS_LIST = [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
RESOLUTIONS = [32, 64, 128, 256]
METRIC = "slab"
PLOT_REFS = ("analytic", "sinkhorn_same")
METRICS = ("slab", "interior", "max_t")
REF_COLOR = {"analytic": "tab:blue", "sinkhorn_same": "tab:orange", "sinkhorn_fine": "tab:green"}
GUIDE_OFFSET = 0.40


def fmt_eps(vareps):
    if vareps == 0:
        return "0"
    if vareps < 0.01:
        return rf"10^{{{int(round(np.log10(vareps)))}}}"
    return f"{vareps:g}"


def tag(vareps, n):
    grid = "clustered" if vareps >= 1e-2 else "uniform"
    return f"gaussian_eps{vareps:g}_n{n}_{grid}_ETD"


def row_weights(d):
    dt_vec = d["dt_vec"]
    return 0.5 * (dt_vec[:-1] + dt_vec[1:])


def errors(d, ref):
    rows, w = d[f"rows_{ref}_rho"], row_weights(d)
    return {"slab": float(np.sqrt(np.sum(w * rows))),
            "interior": float(np.sqrt(np.sum(w[1:-1] * rows[1:-1]))),
            "max_t": float(np.sqrt(rows.max()))}


def fit_order(ns, errs):
    ns, errs = np.asarray(ns, float), np.asarray(errs, float)
    ok = errs > 0
    if ok.sum() < 2:
        return np.nan
    return -np.polyfit(np.log(ns[ok]), np.log(errs[ok]), 1)[0]


def collect(suite, resolutions):
    data, diag, hist = {}, {}, {}
    for vareps in EPS_LIST:
        per_ref, rows, hs = {}, [], {}
        for n in resolutions:
            f = suite / f"{tag(vareps, n)}.npz"
            if not f.exists():
                rows.append(None)
                continue
            d = np.load(f, allow_pickle=True)
            rows.append((int(d["iters"]), bool(d["converged"]), float(d["D"][-1]), float(d["P"][-1]),
                         float(d["walltime"]), float(d["min_rho_t"].min()),
                         float(np.abs(d["mass_t"] - 1).max())))
            refs = [str(r) for r in d["ref_names"]]
            for ref in refs:
                e = errors(d, ref)
                for m in METRICS:
                    per_ref.setdefault(ref, {}).setdefault(m, []).append(e[m])
            hs[n] = {ref: d[f"err_rho_{ref}"] for ref in refs}
        data[vareps], diag[vareps], hist[vareps] = per_ref, rows, hs
    return data, diag, hist


def panel(ax, vareps, per_ref, resolutions):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(resolutions))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    shown = {k: v for k, v in per_ref.items() if k in PLOT_REFS}
    ns = np.array(resolutions, float)
    last = None
    for ref, per_metric in sorted(shown.items()):
        e = np.array(per_metric[METRIC], float)
        if e.size != ns.size:
            continue
        ax.plot(ns, e, "-o", color=REF_COLOR.get(ref, "0.3"), ms=5, lw=1.6, label=f"LADMM vs {ref}")
        last = e
    if last is not None:
        unit = (ns[0] / ns) ** 2
        ax.plot(ns, GUIDE_OFFSET * np.min(last / unit) * unit, linestyle=(0, (4, 3)), color="0.45", lw=1.2,
                label=r"$O(h^2)=O(n^{-2})$")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite", default=str(SUITE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n", type=int, nargs="+", default=RESOLUTIONS)
    a = ap.parse_args()
    suite, out, resolutions = Path(a.suite), Path(a.out), sorted(a.n)
    out.mkdir(parents=True, exist_ok=True)

    data, diag, hist = collect(suite, resolutions)

    print(f"=== gaussian 2D: fitted order in h (n = {resolutions}) ===")
    print(f"  {'vareps':>8} {'reference':>15} " + " ".join(f"{m:>9}" for m in METRICS)
          + "   |  per n: iters/conv, final D, min rho, |mass-1|")
    for vareps in EPS_LIST:
        if not data[vareps]:
            print(f"  {vareps:8g} {'(no runs)':>15}")
            continue
        for i, (ref, per_metric) in enumerate(sorted(data[vareps].items())):
            line = (f"  {vareps:8g} {ref:>15} "
                    + " ".join(f"{fit_order(resolutions[:len(per_metric[m])], per_metric[m]):9.2f}" for m in METRICS))
            if i == 0:
                line += "   |  " + "  ".join(f"{r[0]}/{'Y' if r[1] else 'N'} {r[2]:.1e} {r[5]:.1e} {r[6]:.0e}"
                                            for r in diag[vareps] if r)
            print(line)
            print(f"  {'':8} {'':>15} " + " ".join(f"{'':>9}" for _ in METRICS)
                  + "   |  slab err per n: " + "  ".join(f"{e:.3e}" for e in per_metric["slab"]))

    for vareps in EPS_LIST:
        if not data[vareps]:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        panel(ax, vareps, data[vareps], resolutions)
        ax.set_xlabel("n  (nt = nx = ny)")
        ax.set_ylabel(r"$\|\rho-\rho_{ref}\|$")
        fig.suptitle(rf"gaussian 2D: grid convergence, $\varepsilon = {fmt_eps(vareps)}$", y=0.99)
        fig.tight_layout()
        p = out / f"gridconv2d_gaussian_eps{vareps:g}.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"  wrote {p.name}")

    # all vareps on one axes
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(resolutions))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ns = np.array(resolutions, float)
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
        ax.plot(ns, e, "-", marker="o" if ref == "analytic" else "s", color=c, ms=5, lw=1.6,
                label=rf"$\varepsilon = {fmt_eps(vareps)}$")
        finest.append(e[-1])
    if finest:
        anchor = max(finest) * 1.9
        ax.plot(ns, anchor * (ns[-1] / ns) ** 2, linestyle=(0, (4, 3)), color="0.45", lw=1.3,
                label=r"$O(h^2)=O(n^{-2})$")
    ax.set_xlabel("n  (nt = nx = ny)")
    ax.set_ylabel(r"$\|\rho-\rho_{ref}\|$")
    ax.set_title("gaussian 2D: grid convergence\ncircles = vs closed form,  squares = vs Sinkhorn (product)",
                 fontsize=10)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    p = out / "gridconv2d_gaussian_all.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p.name}")

    # convergence history: err_rho vs iteration, one panel per vareps, one curve per n
    epss = [e for e in EPS_LIST if hist[e]]
    if epss:
        ncol = min(4, len(epss))
        nrow = int(np.ceil(len(epss) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
        ncolors = plt.cm.plasma(np.linspace(0, 0.85, len(resolutions)))
        for ax, vareps in zip(axes.flat, epss):
            for n, c in zip(resolutions, ncolors):
                if n not in hist[vareps]:
                    continue
                hs = hist[vareps][n]
                ref = next((r for r in PLOT_REFS if r in hs), sorted(hs)[0])
                e = hs[ref]
                k = np.arange(1, e.size + 1)
                ax.loglog(k, e, color=c, lw=1.3, label=f"n={n}")
            ax.set_title(rf"$\varepsilon = {fmt_eps(vareps)}$  (vs {ref})", fontsize=9)
            ax.set_xlabel("iteration"); ax.set_ylabel(r"$\|\rho^k-\rho_{ref}\|$")
            ax.grid(True, which="major", alpha=0.3); ax.legend(fontsize=7)
        for ax in axes.flat[len(epss):]:
            ax.axis("off")
        fig.suptitle("gaussian 2D: density error vs iteration (does 40k converge?)", y=0.995)
        fig.tight_layout()
        p = out / "convhist2d_gaussian.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"  wrote {p.name}")


if __name__ == "__main__":
    main()
