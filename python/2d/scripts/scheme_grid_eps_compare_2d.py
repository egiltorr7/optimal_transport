"""CN vs ETD, uniform vs clustered time grid, swept over vareps in 2D.

The four-way comparison the 1D work established and the 2D tree could not run
until projection_expsemi_2d.py / projection_expsemi_nu_2d.py existed:

    scheme in {CN (projection_2d / projection_nu_2d),
               ETD (projection_expsemi_2d / projection_expsemi_nu_2d)}
    grid   in {uniform, clustered}

Errors are measured in the grid's OWN quadrature norm,
sqrt(sum_n w_n |g-a|^2) / sqrt(sum_n w_n |a|^2) with w_n the time weight
(dt for uniform, dt_vec/dt_dual for clustered), so the uniform and clustered
numbers are comparable rather than differing by a quadrature convention.

min(rho) is reported alongside the error on purpose. Neither scheme enforces
rho >= 0 -- both are plain projections -- so accuracy is not where they part
company; positivity is, and it is the quantity that degrades with vareps/dt.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import jax_config

jax_config.configure(device="cpu", x64=True)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setup.grid_2d import setup_problem_2d
from setup.grid_nu_2d import setup_problem_nu_2d
from setup.operators_2d import build_operators_2d
from setup.operators_nu_2d import build_operators_nu_2d
from setup.pipeline_2d import LadmmConfig as Cfg2D
from setup.pipeline_2d import discretize_then_optimize_2d
from setup.pipeline_nu_2d import LadmmConfig as CfgNU
from setup.pipeline_nu_2d import discretize_then_optimize_nu_2d
from setup.problems_2d import analytical_sb_gaussian_2d, prob_gaussian_2d
from setup.problems_nu_2d import analytical_sb_gaussian_nu_2d
from setup.projection_2d import (precomp_banded_proj_2d,
                                 proj_fokker_planck_banded_2d)
from setup.projection_expsemi_2d import (precomp_expsemi_proj_2d,
                                         proj_fokker_planck_expsemi_2d)
from setup.projection_expsemi_nu_2d import (precomp_expsemi_proj_nu_2d,
                                            proj_fokker_planck_expsemi_nu_2d)
from setup.projection_nu_2d import (precomp_banded_proj_nu_2d,
                                    proj_fokker_planck_banded_nu_2d)
from setup.time_grid import clustered_time_grid

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SCHEME_COLOR = {"CN": "#2a78d6", "ETD": "#eb6834"}     # categorical slots 1-2
GRID_STYLE = {"uniform": "-", "clustered": "--"}
GRID_MARKER = {"uniform": "o", "clustered": "s"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="both", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#d5d4d0")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def wrel(got, want, w):
    """Relative error in the grid's own time-quadrature norm."""
    g, a = np.asarray(got), np.asarray(want)
    ww = np.asarray(w).reshape((-1,) + (1,) * (g.ndim - 1))
    return float(np.sqrt((ww * (g - a) ** 2).sum()) / np.sqrt((ww * a ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--max-iter", type=int, default=8000)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    n = args.n
    pd = prob_gaussian_2d(mu0=(0.4, 0.4), mu1=(0.6, 0.6), sigma=0.08)
    kw = dict(gamma=1.0, tau=1.01, max_iter=args.max_iter, eps_abs=1e-12, eps_rel=1e-9)
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    p_u = setup_problem_2d(pd, nt=n, nx=n, ny=n)
    p_u.ops = build_operators_2d(p_u)
    tg = clustered_time_grid(n, strength=args.strength)
    p_c = setup_problem_nu_2d(pd, tg, nx=n, ny=n)
    p_c.ops = build_operators_nu_2d(p_c)

    print(f"n={n}  max_iter={args.max_iter}  clustered strength={args.strength}")
    print(f"clustered dt_vec in [{float(p_c.dt_vec.min()):.4e}, {float(p_c.dt_vec.max()):.4e}] "
          f"(uniform dt = {p_u.dt:.4e}); dt ratio {float(p_c.dt_vec.max()/p_c.dt_vec.min()):.1f}x\n")

    CASES = {
        ("CN", "uniform"): (p_u, precomp_banded_proj_2d, proj_fokker_planck_banded_2d),
        ("ETD", "uniform"): (p_u, precomp_expsemi_proj_2d, proj_fokker_planck_expsemi_2d),
        ("CN", "clustered"): (p_c, precomp_banded_proj_nu_2d, proj_fokker_planck_banded_nu_2d),
        ("ETD", "clustered"): (p_c, precomp_expsemi_proj_nu_2d, proj_fokker_planck_expsemi_nu_2d),
    }

    res = {}
    for (scheme, grid), (p, precomp, proj) in CASES.items():
        for vareps in args.eps:
            st = precomp(p, vareps)
            jp = jax.jit(lambda s, _p=p, _v=vareps, _st=st, _pr=proj: _pr(s, _p, _v, _st))
            if grid == "uniform":
                rho_a, mx_a, my_a = analytical_sb_gaussian_2d(p, vareps)
                w_rho = np.full(p.nt - 1, p.dt)
                w_m = np.full(p.nt, p.dt)
                r = discretize_then_optimize_2d(p, lambda s, a, b, c: jp(s), st, vareps,
                                                Cfg2D(**kw))
            else:
                rho_a, mx_a, my_a = analytical_sb_gaussian_nu_2d(p, vareps)
                w_rho = np.asarray(p.dt_dual)
                w_m = np.asarray(p.dt_vec)
                r = discretize_then_optimize_nu_2d(p, lambda s, a, b, c: jp(s), st, vareps,
                                                   CfgNU(**kw))
            d = dict(
                e_rho=wrel(r.rho_stag, rho_a, w_rho),
                e_mx=wrel(r.mx_stag, mx_a, w_m),
                e_my=wrel(r.my_stag, my_a, w_m),
                min_rho=float(jnp.min(r.rho_stag)),
                min_rho_cc=float(jnp.min(r.rho_cc)),
                peak=float(jnp.max(rho_a)),
                iters=r.info.iters, conv=r.info.converged, D=float(r.info.D[-1]),
            )
            res[(scheme, grid, vareps)] = d
            print(f"{scheme:<4s} {grid:<10s} eps={vareps:<8g} iters={d['iters']:<6d} "
                  f"conv={str(d['conv']):<5s} rho={d['e_rho']:.3e} mx={d['e_mx']:.3e}  "
                  f"min_rho={d['min_rho']:+.3e} ({d['min_rho']/d['peak']:+.1e} of peak)",
                  flush=True)

    # --------- plots ---------
    eps_x = [max(e, 1e-9) for e in args.eps]   # 0 plotted at the decade floor
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), facecolor=SURFACE)
    panels = [("e_rho", r"relative error in $\rho$"),
              ("e_mx", r"relative error in $m_x$"),
              ("min_rho_frac", r"$\min\rho \,/\, \rho_{\rm peak}$   (negative = violation)")]
    for ax, (key, ylab) in zip(axes, panels):
        _style(ax)
        for scheme in ("CN", "ETD"):
            for grid in ("uniform", "clustered"):
                if key == "min_rho_frac":
                    ys = [res[(scheme, grid, e)]["min_rho"] / res[(scheme, grid, e)]["peak"]
                          for e in args.eps]
                else:
                    ys = [res[(scheme, grid, e)][key] for e in args.eps]
                ax.plot(eps_x, ys, GRID_STYLE[grid], marker=GRID_MARKER[grid],
                        markersize=6, linewidth=2, color=SCHEME_COLOR[scheme],
                        label=f"{scheme}, {grid}")
        ax.set_xscale("log")
        if key != "min_rho_frac":
            ax.set_yscale("log")
        else:
            ax.axhline(0.0, color=INK2, linewidth=1.0, linestyle=":")
        ax.set_xlabel(r"$\varepsilon$   ($\varepsilon=0$ drawn at $10^{-9}$)")
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"CN vs ETD, uniform vs clustered, n = {n}", fontsize=10, color=INK)
    fig.tight_layout()
    f1 = out_dir / f"scheme_grid_eps_compare_2d_n{n}{args.tag}.png"
    fig.savefig(f1, dpi=150, facecolor=SURFACE)

    npz = out_dir / f"scheme_grid_eps_compare_2d_n{n}{args.tag}.npz"
    np.savez_compressed(npz, **{f"{s}|{g}|{e:g}|{k}": v
                                for (s, g, e), d in res.items() for k, v in d.items()})
    print(f"\nsaved {f1}\nsaved {npz}")


if __name__ == "__main__":
    main()
