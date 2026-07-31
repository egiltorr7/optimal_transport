"""Static Schrodinger bridge between two 1D Gaussians, solved with POT.

Convention (fixed so POT's `reg` argument IS the physical epsilon):
  Reference process: Brownian motion with diffusivity sigma^2 run for time T.
  Its transition density is exp(-|x-y|^2 / (2 sigma^2 T)).
  POT's Gibbs kernel is exp(-C(x,y) / reg).
  Matching these requires C(x,y) = |x-y|^2 / 2 and reg = sigma^2 * T.
  Grid/Gaussian/solver parameters are grouped into small config objects
  below so a sweep can vary any of them independently.
"""
from dataclasses import dataclass
import time
import warnings

import numpy as np
import ot


@dataclass
class GridConfig:
    n_points: int = 200
    x_min: float = 0.0
    x_max: float = 1.0


@dataclass
class GaussianConfig:
    mean1: float = 0.3
    std1: float = 0.05
    mean2: float = 0.7
    std2: float = 0.05
    T: float = 1.0  # reference process time horizon; reg passed to POT = sigma^2 * T


@dataclass
class SolverConfig:
    num_iter_max: int = 20000
    stop_thr: float = 1e-12
    num_inner_iter_max: int = 200  # only used by sinkhorn_epsilon_scaling


METHODS = ["sinkhorn", "sinkhorn_log", "sinkhorn_stabilized", "sinkhorn_epsilon_scaling"]


def make_grid(cfg: GridConfig):
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.n_points)
    dx = x[1] - x[0]
    return x, dx


def gaussian_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def make_marginals(x, cfg: GaussianConfig):
    a = gaussian_pdf(x, cfg.mean1, cfg.std1)
    b = gaussian_pdf(x, cfg.mean2, cfg.std2)
    a /= a.sum()
    b /= b.sum()
    return a, b


def make_cost(x):
    return ot.dist(x.reshape(-1, 1), x.reshape(-1, 1), metric="sqeuclidean") / 2.0


def solve(a, b, C, sigma2, method, solver_cfg: SolverConfig, T: float = 1.0):
    """Solve one instance. `sigma2` is the physical diffusivity; the
    regularization passed to POT is reg = sigma2 * T."""
    reg = sigma2 * T
    t0 = time.time()
    status = "ok"
    P = None
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            if method == "sinkhorn_epsilon_scaling":
                P, _ = ot.bregman.sinkhorn_epsilon_scaling(
                    a, b, C, reg=reg, log=True,
                    numItermax=solver_cfg.num_iter_max,
                    numInnerItermax=solver_cfg.num_inner_iter_max,
                )
            else:
                P, _ = ot.sinkhorn(
                    a, b, C, reg=reg, method=method, log=True,
                    numItermax=solver_cfg.num_iter_max,
                    stopThr=solver_cfg.stop_thr,
                )
        except Exception as e:
            status = f"exception:{type(e).__name__}"
    dt = time.time() - t0

    if P is None or np.any(np.isnan(P)) or np.any(np.isinf(P)):
        if status == "ok":
            status = "nan/inf"
        marg_err = np.nan
    else:
        marg_err = max(np.abs(P.sum(1) - a).max(), np.abs(P.sum(0) - b).max())
        if status == "ok" and wlist:
            status = "warning:" + wlist[-1].category.__name__

    return dict(P=P, status=status, marg_err=marg_err, time=dt)


def sweep(sigma2_values, methods=METHODS, grid_cfg=GridConfig(), gauss_cfg=GaussianConfig(),
          solver_cfg=SolverConfig(), verbose=True):
    x, dx = make_grid(grid_cfg)
    a, b = make_marginals(x, gauss_cfg)
    C = make_cost(x)

    results = []
    if verbose:
        print(f"grid: N={grid_cfg.n_points}, dx={dx:.3e}, dx^2={dx**2:.3e}, T={gauss_cfg.T}")
        header = f"{'method':<26}{'sigma^2':>10}{'reg':>10}  {'status':<20}{'marg_err':>12}{'time(s)':>9}"
        print(header)

    for method in methods:
        for sigma2 in sigma2_values:
            r = solve(a, b, C, sigma2, method, solver_cfg, T=gauss_cfg.T)
            r.update(method=method, sigma2=sigma2, reg=sigma2 * gauss_cfg.T)
            results.append(r)
            if verbose:
                print(f"{method:<26}{sigma2:>10.1e}{r['reg']:>10.1e}  "
                      f"{r['status']:<20}{r['marg_err']:>12.2e}{r['time']:>9.3f}")

    return results, dict(x=x, dx=dx, a=a, b=b, C=C)


def plot_convergence(results, dx, methods=METHODS, save_path="sb_1d_convergence.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for m in methods:
        pts = [r for r in results if r["method"] == m]
        sigma2 = np.array([p["sigma2"] for p in pts])
        err = np.clip(np.array([p["marg_err"] for p in pts]), 1e-16, None)
        ax.loglog(sigma2, err, "o-", label=m, linewidth=1.8, markersize=4)

    eps_wall = dx ** 2
    ax.axvline(eps_wall, color="black", linestyle=":", linewidth=1)
    ax.text(eps_wall, 2e-16, "  sqrt(sigma^2 T)=dx\n  (grid wall)", fontsize=8, va="bottom")

    ax.set_xlabel("sigma^2 (T=1)")
    ax.set_ylabel("max marginal constraint violation")
    ax.set_title("POT Sinkhorn variants: 1D Gaussian-to-Gaussian SB")
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"saved {save_path}")


if __name__ == "__main__":
    # ---- edit these to explore -------------------------------------------
    grid_cfg = GridConfig(n_points=200, x_min=0.0, x_max=1.0)
    gauss_cfg = GaussianConfig(mean1=0.3, std1=0.05, mean2=0.7, std2=0.05, T=1.0)
    solver_cfg = SolverConfig(num_iter_max=20000, stop_thr=1e-12)
    sigma2_values = [1e0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4,
                      3e-5, 1e-5, 3e-6, 1e-6, 1e-7, 1e-8]
    methods = METHODS
    # ------------------------------------------------------------------------

    results, data = sweep(sigma2_values, methods=methods, grid_cfg=grid_cfg,
                           gauss_cfg=gauss_cfg, solver_cfg=solver_cfg)
    plot_convergence(results, data["dx"], methods=methods)
