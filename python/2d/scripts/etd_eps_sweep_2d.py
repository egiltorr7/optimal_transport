"""ETD-2D on a UNIFORM grid: error vs grid and vs iteration, swept over vareps.

Companion to validate_expsemi_2d.py. That script checks the projection and the
discretization in isolation (and is where the O(h^2) claim is established);
this one runs the full LADMM solve and plots how close it actually gets.

Two things are plotted on purpose rather than one, because they answer different
questions and are easy to confuse:

  error vs ITERATION  -- has LADMM converged? A curve still falling at max_iter
                         means the number in the vs-grid plot is a solver
                         artifact, not a discretization error.
  error vs GRID       -- is the discretization second order? Only meaningful
                         where the iteration curves have flattened, and only
                         down to the reference's own O(h^2) floor, which is
                         drawn alongside (the free-space closed form does not
                         satisfy the discrete no-flux walls; see 3d/README.md).

Marginals are the boundary-free ones (means 0.4/0.6, sigma=0.08), NOT
prob_gaussian_2d's defaults -- with those the reference is not a valid target
at all (its own FP residual stalls near 3.5e-2 and then rises under
refinement).
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
from matplotlib.colors import LinearSegmentedColormap

from setup.grid_2d import setup_problem_2d
from setup.operators_2d import build_operators_2d
from setup.pipeline_2d import LadmmConfig, discretize_then_optimize_2d
from setup.problems_2d import analytical_sb_gaussian_2d, prob_gaussian_2d
from setup.projection_2d import dct2_xy, idct2_xy
from setup.projection_expsemi_2d import (precomp_expsemi_proj_2d,
                                         proj_fokker_planck_expsemi_2d)
from setup.state_2d import State2D

# --- design tokens (dataviz skill: validated ordinal blue ramp + cat slots) ---
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID_RAMP = ["#86b6ef", "#2a78d6", "#104281"]          # ordinal: coarse -> fine
EPS_CAT = ["#2a78d6", "#eb6834", "#1baf7a"]            # categorical slots 1-3
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#3987e5", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_br", ["#184f95", "#f0efec", "#c0392f"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="both", color="#e6e5e1", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#d5d4d0")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def reference_floor(p, vareps, ep, rho_a, mx_a, my_a):
    """||R(analytical)|| / ||d_t-ish scale|| under the ETD discretization."""
    ops = p.ops
    zx, zy = jnp.zeros((p.nt, p.ny)), jnp.zeros((p.nt, p.nx))
    mu_prev = jnp.concatenate([p.rho0[None], rho_a], axis=0)
    mu_curr = jnp.concatenate([rho_a, p.rho1[None]], axis=0)
    S_prev = idct2_xy(ep.c_vals * dct2_xy(mu_prev))
    div_m = ops.deriv_x_at_phi(mx_a, zx, zx) + ops.deriv_y_at_phi(my_a, zy, zy)
    R = (mu_curr - S_prev) / p.dt + idct2_xy(ep.phi_vals * dct2_xy(div_m))
    return float(jnp.linalg.norm(R)) / float(jnp.linalg.norm((mu_curr - S_prev) / p.dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 1e-8, 1e-4])
    ap.add_argument("--max-iter", type=int, default=20_000)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    pd = prob_gaussian_2d(mu0=(0.4, 0.4), mu1=(0.6, 0.6), sigma=0.08)
    cfg_kw = dict(gamma=args.gamma, tau=args.gamma * 1.01, max_iter=args.max_iter,
                  eps_abs=1e-12, eps_rel=1e-9)
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    res = {}      # (eps, n) -> dict
    for vareps in args.eps:
        for n in args.grids:
            p = setup_problem_2d(pd, nt=n, nx=n, ny=n)
            p.ops = build_operators_2d(p)
            ep = precomp_expsemi_proj_2d(p, vareps)
            rho_a, mx_a, my_a = analytical_sb_gaussian_2d(p, vareps)

            # jit the projection only; verified against eager before use, so
            # this changes cost and not arithmetic.
            jp = jax.jit(lambda s: proj_fokker_planck_expsemi_2d(s, p, vareps, ep))
            rng = np.random.default_rng(0)
            probe = State2D(jnp.asarray(rng.standard_normal((n - 1, n, n))),
                            jnp.asarray(rng.standard_normal((n, n - 1, n))),
                            jnp.asarray(rng.standard_normal((n, n, n - 1))))
            a, b = proj_fokker_planck_expsemi_2d(probe, p, vareps, ep), jp(probe)
            jit_dev = max(float(jnp.abs(a.rho - b.rho).max()),
                          float(jnp.abs(a.mx - b.mx).max()),
                          float(jnp.abs(a.my - b.my).max()))
            assert jit_dev < 1e-12, f"jit != eager ({jit_dev:.2e})"

            t0 = time.perf_counter()
            r = discretize_then_optimize_2d(
                p, lambda st, pr, ve, stt: jp(st), ep, vareps, LadmmConfig(**cfg_kw),
                x_ref=State2D(rho_a, mx_a, my_a),
            )
            wall = time.perf_counter() - t0
            rel = lambda g, w: float(jnp.linalg.norm(g - w) / jnp.linalg.norm(w))
            e_rho, e_mx, e_my = (rel(r.rho_stag, rho_a), rel(r.mx_stag, mx_a),
                                 rel(r.my_stag, my_a))
            floor = reference_floor(p, vareps, ep, rho_a, mx_a, my_a)
            res[(vareps, n)] = dict(
                err_iter=np.asarray(r.info.err_x), D=np.asarray(r.info.D),
                e_rho=e_rho, e_mx=e_mx, e_my=e_my, floor=floor,
                iters=r.info.iters, converged=r.info.converged,
                rho=np.asarray(r.rho_stag), rho_a=np.asarray(rho_a),
                ms_per_it=r.info.walltime / r.info.iters * 1e3,
            )
            print(f"eps={vareps:<8g} n={n:<3d} iters={r.info.iters:<6d} "
                  f"conv={str(r.info.converged):<5s} D={r.info.D[-1]:.2e}  "
                  f"rho={e_rho:.3e} mx={e_mx:.3e} my={e_my:.3e}  "
                  f"ref_floor={floor:.3e}  [{wall:.0f}s, {res[(vareps,n)]['ms_per_it']:.1f} ms/it]",
                  flush=True)

    # ================= Figure A: error vs iteration =================
    fig, axes = plt.subplots(1, len(args.eps), figsize=(4.2 * len(args.eps), 3.6),
                             facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, vareps in zip(axes, args.eps):
        _style(ax)
        for k, n in enumerate(args.grids):
            d = res[(vareps, n)]
            ax.loglog(np.arange(1, d["iters"] + 1), d["err_iter"], linewidth=2,
                      color=GRID_RAMP[k % len(GRID_RAMP)], label=f"n = {n}", zorder=3)
        ax.set_title(f"$\\varepsilon$ = {vareps:g}", fontsize=10)
        ax.set_xlabel("LADMM iteration")
    axes[0].set_ylabel(r"$\|x^k - x_{\rm ref}\|$")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Convergence: flat tail = discretization-limited, still falling = under-converged",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fa = out_dir / f"etd_2d_eps_sweep_convergence{args.tag}.png"
    fig.savefig(fa, dpi=150, facecolor=SURFACE)

    # ================= Figure B: error vs grid =================
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE)
    for ax, (key, lbl) in zip(axes, [("e_rho", r"$\rho$"), ("e_mx", r"$m_x$")]):
        _style(ax)
        for k, vareps in enumerate(args.eps):
            ys = [res[(vareps, n)][key] for n in args.grids]
            ax.loglog(args.grids, ys, "o-", linewidth=2, markersize=6,
                      color=EPS_CAT[k % len(EPS_CAT)],
                      label=f"$\\varepsilon$ = {vareps:g}", zorder=3)
        fl = [res[(args.eps[0], n)]["floor"] for n in args.grids]
        ax.loglog(args.grids, fl, "--", linewidth=1.6, color=INK2, zorder=2,
                  label="reference's own $O(h^2)$ floor")
        g = np.asarray(args.grids, dtype=float)
        ax.loglog(g, fl[0] * (g / g[0]) ** -2.0, ":", linewidth=1.2,
                  color="#a8a7a2", zorder=1, label="slope $-2$")
        ax.set_xlabel("n  (nt = nx = ny)")
        ax.set_ylabel(f"relative error in {lbl}")
        ax.set_xticks(args.grids)
        ax.set_xticklabels([str(n) for n in args.grids])
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Solver vs closed form, against the floor the closed form itself sets",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fb = out_dir / f"etd_2d_eps_sweep_vs_grid{args.tag}.png"
    fig.savefig(fb, dpi=150, facecolor=SURFACE)

    # ================= Figure C: density at the finest grid =================
    nfin = args.grids[-1]
    fig, axes = plt.subplots(len(args.eps), 3, figsize=(10, 3.1 * len(args.eps)),
                             facecolor=SURFACE, squeeze=False)
    for i, vareps in enumerate(args.eps):
        d = res[(vareps, nfin)]
        mid = (nfin - 1) // 2
        num, ana = d["rho"][mid], d["rho_a"][mid]
        vmax = float(max(num.max(), ana.max()))
        dif = num - ana
        lim = float(np.abs(dif).max())
        for ax, (data, ttl, cm, kw) in zip(axes[i], [
            (num, "numerical", SEQ, dict(vmin=0, vmax=vmax)),
            (ana, "closed form", SEQ, dict(vmin=0, vmax=vmax)),
            (dif, "difference", DIV, dict(vmin=-lim, vmax=lim)),
        ]):
            im = ax.imshow(data.T, origin="lower", extent=[0, 1, 0, 1], cmap=cm, **kw)
            ax.set_title(f"$\\varepsilon$={vareps:g}  {ttl}", fontsize=9, color=INK)
            ax.tick_params(colors=INK2, labelsize=7)
            cb = fig.colorbar(im, ax=ax, fraction=0.046)
            cb.ax.tick_params(colors=INK2, labelsize=7)
    fig.suptitle(f"$\\rho$ at mid-time, n = {nfin}", fontsize=10, color=INK)
    fig.tight_layout()
    fc = out_dir / f"etd_2d_eps_sweep_density{args.tag}.png"
    fig.savefig(fc, dpi=150, facecolor=SURFACE)

    npz = out_dir / f"etd_2d_eps_sweep{args.tag}.npz"
    np.savez_compressed(npz, **{
        f"{k}|{eps:g}|{n}": v for (eps, n), d in res.items()
        for k, v in d.items() if isinstance(v, np.ndarray) or np.isscalar(v)
    })
    for f in (fa, fb, fc, npz):
        print(f"saved {f}")


if __name__ == "__main__":
    main()
