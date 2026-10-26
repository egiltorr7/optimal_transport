# Pikachu -> Raichu transport

## Run it (on a CUDA machine)

    cd matlab/discretize_first/2d/experiments
    bash run_pikachu_raichu_all.sh /path/to/matlab

Or one epsilon at a time:

    matlab -batch "cd('..'); setup_paths; run_pikachu_raichu(0)"
    matlab -batch "cd('..'); setup_paths; run_pikachu_raichu(1e-2)"

Output: `matlab/discretize_first/2d/results/pikachu/pikachu_raichu_eps<E>_128.mat`

## Settings

| | |
|---|---|
| marginals | `images/mollified/pikachu_raichu_w0.04_128.mat` (precomputed) |
| grid | nt=32, nx=ny=128 |
| projection | ETD (`proj_fokker_planck_expsemi_gpu`) |
| vareps | 0 (pure OT) and 1e-2 |
| gamma / tau | 1 / 1.05 |
| max_iter / tol | 20000 / 1e-8 |

## Reading the output in Python

    from scipy.io import loadmat
    d = loadmat('pikachu_raichu_eps0_128.mat')
    rho = d['rho_cc']              # (nt, nx, ny) density over time
    p, q = d['res_primal'].ravel(), d['res_dual'].ravel()

Saved with `-v7` specifically so `scipy.io.loadmat` can read it; `-v7.3` is
HDF5 and scipy cannot.

Fields: `rho_cc, mx_cc, my_cc` (nt x nx x ny), `rho_stag` (ntm x nx x ny),
`rho0, rho1`, `res_primal, res_dual, res_x, res_y, iter_times`,
`xx, yy, dx, dy, dt, nt, nx, ny`, and run metadata (`vareps, gamma, tau,
alpha, tol, max_iter, iters, converged, walltime, projection, cache_file,
created`).

## Two things to know before reading the numbers

**`res_dual` is new.** `ladmm_solve` previously tracked only `res_x`,
`res_y` and `res_primal`. `res_y = ||y-y_prev||` is *not* the dual residual
-- linearizing the x-step changes the optimality condition. The added
`res_dual` is

    P = tau*(x^t - x^{t+1}) + gamma*A^T*(D^t - D^{t+1}),   D = A*x + B*y - b

matching the 1D/Python `ladmm.py` so all three implementations report the
same quantity. It costs one extra `A^T` per iteration. Both additions are
backward compatible -- nothing else changed.

**No reference-error tracking, and none was removed.** There is no closed
form for this problem. `ladmm_solve` never tracked a reference error in the
first place, so there was no cost to save there; `res_dual` is the only
diagnostic overhead added.

## Expect it not to converge to 1e-8

On the Gaussian problem the dual residual decayed algebraically (exponent
~1.2), needing far more than 20000 iterations for 1e-8. This problem is
harder -- sharp-ish edges, ~55-70% of the domain at exactly zero. The
morph will very likely *look* right long before the residuals say so, so
judge it visually and report the residuals as they are rather than calling
it converged. `stalled` and `diverged` in the .mat say which way it ended.
