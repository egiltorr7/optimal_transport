"""Computational cost and accuracy of a reduced-precision suite (fp32 or
mixed) against the fp64 one. Post-processing only; reads results/suite (fp64)
and results/suite_<label> (--label fp32 | mixed, default fp32).

Per (vareps, n) it tabulates, from the two .npz files:
  ms/iter            steady-state, compile excluded (walltime - first chunk)
  wall               whole solve, compile included
  peak GPU memory    fp32 recorded by run_suite_2d.py; fp64 runs predate that
                     field, so their column is filled from bench_2d.py's
                     measurements where available and left blank otherwise
  err                slab density error against the reference, both precisions
  |fp32 - fp64|      relative L2 distance between the two SOLUTIONS on the
                     diagonal x = y, over all times -- the precision cost
                     itself, independent of any reference

Writes cost_<label>_vs_fp64.csv and .md into --out (default
results/post_<label>), and a figure overlaying the two grid-convergence curve
sets; --extra <suite> --extra-label <name> adds a third set (dotted) to the
figure and its error / distance-to-fp64 columns to the table, e.g. to see
fp64 / mixed / fp32 side by side.

Usage:
    python scripts/post_cost_2d.py                       # fp32 vs fp64
    python scripts/post_cost_2d.py --label mixed         # mixed vs fp64
    python scripts/post_cost_2d.py --label mixed --extra results/suite_fp32 --extra-label fp32
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

ROOT = Path(__file__).resolve().parent.parent / "results"
EPS_LIST = [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
RESOLUTIONS = [32, 64, 128, 256]
PLOT_REFS = ("analytic", "sinkhorn_same")
# bench_2d.py, GH200, fp64, ETD, dct modes + matmul DCT (uniform) / scan Thomas
# (clustered), 2026-09-24. Peak MB as reported by jax memory_stats.
FP64_PEAK_MB_BENCH = {("uniform", 32): 104, ("uniform", 64): 183, ("uniform", 128): 1273,
                      ("uniform", 256): 9033, ("clustered", 32): 105, ("clustered", 128): 983,
                      ("clustered", 256): 8164}


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


def slab_err(d):
    names = [str(r) for r in d["ref_names"]]
    ref = next((r for r in PLOT_REFS if r in names), names[0])
    return float(np.sqrt(np.sum(row_weights(d) * d[f"rows_{ref}_rho"]))), ref


def load(suite, vareps, n):
    f = suite / f"{tag(vareps, n)}.npz"
    return np.load(f, allow_pickle=True) if f.exists() else None


def ms_per_iter(d):
    if "ms_per_iter" in d.files:
        return float(d["ms_per_iter"])
    ce = int(d["check_every"]) if "check_every" in d.files else 1
    return 1e3 * (float(d["walltime"]) - float(d["walltime_first_chunk"])) / max(int(d["iters"]) - ce, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="fp32", help="reduced-precision suite label (fp32 | mixed)")
    ap.add_argument("--suite64", default=str(ROOT / "suite"))
    ap.add_argument("--suite-lo", default=None, help="default results/suite_<label>")
    ap.add_argument("--out", default=None, help="default results/post_<label>")
    ap.add_argument("--extra", default=None, help="a third suite to overlay (figure + err columns)")
    ap.add_argument("--extra-label", default="extra")
    ap.add_argument("--n", type=int, nargs="+", default=RESOLUTIONS)
    a = ap.parse_args()
    LO = a.label
    s64 = Path(a.suite64)
    slo = Path(a.suite_lo) if a.suite_lo else ROOT / f"suite_{LO}"
    out = Path(a.out) if a.out else ROOT / f"post_{LO}"
    sx, XL = (Path(a.extra) if a.extra else None), a.extra_label
    out.mkdir(parents=True, exist_ok=True)

    def rel_diag(dref, d):
        a_, b_ = dref["rho_diag"].astype(np.float64), d["rho_diag"].astype(np.float64)
        return float(np.linalg.norm(a_ - b_) / np.linalg.norm(a_))

    rows = []
    for vareps in EPS_LIST:
        for n in a.n:
            d64, dlo = load(s64, vareps, n), load(slo, vareps, n)
            dx = load(sx, vareps, n) if sx else None
            if d64 is None and dlo is None:
                continue
            r = dict(vareps=vareps, n=n, grid="clustered" if vareps >= 1e-2 else "uniform")
            for lab, d in (("fp64", d64), (LO, dlo), (XL, dx)):
                if d is None:
                    continue
                e, ref = slab_err(d)
                r[f"iters_{lab}"] = int(d["iters"])
                r[f"ms_iter_{lab}"] = ms_per_iter(d)
                r[f"wall_s_{lab}"] = float(d["walltime"])
                r[f"err_{lab}"] = e
                r["ref"] = ref
                pk = float(d["peak_gpu_mem_mb"]) if "peak_gpu_mem_mb" in d.files else np.nan
                if not np.isfinite(pk) and lab == "fp64":
                    pk = FP64_PEAK_MB_BENCH.get((r["grid"], n), np.nan)
                r[f"peak_mb_{lab}"] = pk
            if d64 is not None and dlo is not None:
                r["diag_rel_diff"] = rel_diag(d64, dlo)
                r["speedup"] = r["ms_iter_fp64"] / r[f"ms_iter_{LO}"]
                if np.isfinite(r.get("peak_mb_fp64", np.nan)) and np.isfinite(r.get(f"peak_mb_{LO}", np.nan)):
                    r["mem_ratio"] = r["peak_mb_fp64"] / r[f"peak_mb_{LO}"]
            if d64 is not None and dx is not None:
                r[f"diag_rel_diff_{XL}"] = rel_diag(d64, dx)
            rows.append(r)

    cols = ["vareps", "n", "grid", "iters_fp64", f"iters_{LO}", "ms_iter_fp64", f"ms_iter_{LO}", "speedup",
            "wall_s_fp64", f"wall_s_{LO}", "peak_mb_fp64", f"peak_mb_{LO}", "mem_ratio",
            "err_fp64", f"err_{LO}", "diag_rel_diff", "ref"]
    if sx:
        cols += [f"ms_iter_{XL}", f"err_{XL}", f"diag_rel_diff_{XL}"]
    with open(out / f"cost_{LO}_vs_fp64.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                              for c in cols) + "\n")

    def g(r, k, fmt):
        v = r.get(k)
        return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)

    xh = f" err {XL} | \\|{XL}-fp64\\| |" if sx else ""
    lines = [f"# 2D Gaussian suite: {LO} vs fp64 cost and accuracy", "",
             "GH200, one GPU, ETD scheme, gamma=0.1, jit. ms/iter is steady state (compile excluded). "
             "fp64 peak memory comes from bench_2d.py where the suite run predates that field. "
             f"`|{LO}-fp64|` is the relative L2 distance between the two solutions on the diagonal x=y over all times.",
             "",
             f"| eps | n | grid | iters | ms/iter fp64 | ms/iter {LO} | speedup | wall fp64 | wall {LO} | "
             f"peak MB fp64 | peak MB {LO} | err fp64 | err {LO} | \\|{LO}-fp64\\| |" + xh,
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if sx else "")]
    for r in rows:
        xc = f" {g(r, f'err_{XL}', '.3e')} | {g(r, f'diag_rel_diff_{XL}', '.2e')} |" if sx else ""
        lines.append(f"| {r['vareps']:g} | {r['n']} | {r['grid']} | {g(r, f'iters_{LO}', 'd') or g(r, 'iters_fp64', 'd')} | "
                     f"{g(r, 'ms_iter_fp64', '.2f')} | {g(r, f'ms_iter_{LO}', '.2f')} | {g(r, 'speedup', '.2f')} | "
                     f"{g(r, 'wall_s_fp64', '.0f')} s | {g(r, f'wall_s_{LO}', '.0f')} s | "
                     f"{g(r, 'peak_mb_fp64', '.0f')} | {g(r, f'peak_mb_{LO}', '.0f')} | "
                     f"{g(r, 'err_fp64', '.3e')} | {g(r, f'err_{LO}', '.3e')} | {g(r, 'diag_rel_diff', '.2e')} |" + xc)
    tot64 = sum(r.get("wall_s_fp64", 0) for r in rows if "wall_s_fp64" in r)
    totlo = sum(r.get(f"wall_s_{LO}", 0) for r in rows if f"wall_s_{LO}" in r)
    lines += ["", f"Total solve wall time over the runs present in both: fp64 {tot64 / 60:.1f} min, {LO} {totlo / 60:.1f} min."]
    (out / f"cost_{LO}_vs_fp64.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure: fp64 (solid) vs LO (dashed) [vs extra (dotted)] grid-convergence curves
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(a.n)); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(EPS_LIST)))
    series = [("fp64", "-", "o"), (LO, "--", "s")] + ([(XL, ":", "^")] if sx else [])
    for vareps, c in zip(EPS_LIST, colors):
        rs = [r for r in rows if r["vareps"] == vareps]
        for lab, ls, mk in series:
            pts = [(r["n"], r[f"err_{lab}"]) for r in rs if f"err_{lab}" in r and np.isfinite(r[f"err_{lab}"])]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                        color=c, ms=4.5, lw=1.5, label=rf"$\varepsilon={fmt_eps(vareps)}$ {lab}")
    ax.set_xlabel("n  (nt = nx = ny)"); ax.set_ylabel(r"$\|\rho-\rho_{ref}\|$")
    ttl = f"gaussian 2D: grid convergence, fp64 (solid) vs {LO} (dashed)" + (f" vs {XL} (dotted)" if sx else "")
    ax.set_title(ttl, fontsize=10)
    ax.grid(True, which="major", alpha=0.3); ax.legend(fontsize=6.5, ncol=3 if sx else 2, loc="lower left")
    fig.tight_layout()
    p = out / f"gridconv2d_{LO}_vs_fp64.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p.name}")


if __name__ == "__main__":
    main()
