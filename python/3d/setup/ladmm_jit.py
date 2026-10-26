"""jax.jit'd version of ladmm.py's over-relaxed linearized ADMM.

Same algorithm, same LadmmInfo, same stopping rule as ladmm.py -- the
difference is purely how the work is dispatched:

  * ladmm.py runs eagerly: every jnp operation inside A_fn/At_fn/prox_f1/
    solve_y and every norm is its own kernel launch, and every `float(...)`
    in the norms is a device->host sync. At nt=nx=ny=8 that costs ~35 ms
    per iteration on a CPU that could do the arithmetic in microseconds;
    on a GPU the same per-op overhead (plus the Thomas sweep's 3*nt
    dispatches) put the 128^3 run at ~365 ms/iter.
  * here one whole iteration (x-step, y-step, dual update, all seven
    residual/scale norms) is traced once and compiled into a single XLA
    executable, and `check_every` iterations are run back-to-back inside a
    lax.scan before the scalar diagnostics come back to the host for the
    stopping test. With check_every=1 the iterates are those of ladmm.py
    to rounding (XLA may fuse a*b+c into an FMA, so not bit-identical);
    with check_every=k the stop is detected up to k-1 iterations late and
    the returned state is the end-of-chunk one -- every extra iteration is
    a further ADMM step, so this only ever moves the answer closer.

Callable contracts differ from ladmm.py in two ways, both forced by jit:

  prox_f1(v, step, aux)   takes the projection precompute `aux` as an
                          explicit pytree argument instead of closing over
                          it, so the (nt, nx, ny[, nz]) precomputed arrays are
                          runtime inputs rather than constants baked into
                          the compiled executable.
  norm_fn / dual_norm_fn  must return a jnp scalar, NOT a Python float --
                          a float() on a tracer fails under jit.

Mixed precision with master copies (`master_dtype`, see precision.py): the
two quantities LADMM accumulates across iterations, x and delta, are kept in
master_dtype (float64) while y, By, D and everything A_fn / solve_y / the
norms compute stay in the dtype of `b` (float32). Needed because the late
LADMM steps are tiny -- P/tau ~ 1e-8 relative at n >= 128 -- and a float32 x
cannot represent an increment below its own ulp (6e-8): the step is rounded
away and the iterate stalls, at 4e-4 absolute error in the 2D suite, even
with the projection in float64. With the master copy in float64 the only
casts are x -> float32 before A_fn (a fresh rounding each iteration, white
noise, harmless) and the prox argument in solve_y. At_fn then runs in
float64 (its input mixes the float32 residual with the float64 dual) -- a
few slice-averages, memory-bound, cheap -- and prox_f1 must accept and
return master_dtype (precision.mixed_projection with lo_dtype=None).

Master copies alone are NOT enough (measured 2026-09-28, n=128: 5.6e-4 vs
fp64 3.4e-4, and a float64 solve with only the KE prox / y in float32 gives
the same 6.0e-4, while rounding x to float32 before A_fn changes nothing).
The reason: the dual update delta - gamma*(z_hat - y) equals -grad KE(y) at
the EXACT prox output; with y stored in float32 the dual is the gradient at a
point 6e-8 away from the y the constraint A x = y actually uses. That frozen
inconsistency in the KKT system is amplified by the ill-conditioned (slow)
modes of the problem -- the ones LADMM needs 1e5 iterations for -- by ~1e3.
Rounding x has no such inconsistency (the prox is evaluated at the same
rounded x the constraint sees), so it is harmless. `dual_fn` fixes it by
computing the dual as -grad KE at the STORED float32 y, in float64: the KKT
conditions then hold at the stored iterate and float32 only perturbs which
nearby point is found.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .ladmm import LadmmInfo, LadmmOpts


@dataclass
class _Carry:
    x: object
    y: object
    By: object
    delta: object
    D_prev: object


jax.tree_util.register_dataclass(_Carry, data_fields=["x", "y", "By", "delta", "D_prev"], meta_fields=[])

_N_DIAG = 9   # dx, dy, dz, D, P, scale_D, scale_P, err_x, err_delta -- then 2 per x_refs entry


def _cast(v, dtype):
    if dtype is None or v.rho.dtype == dtype:
        return v
    return type(v)(*(getattr(v, f).astype(dtype) for f in v.__dataclass_fields__))


def _make_step(prox_f1, solve_y, A_fn, At_fn, B_fn, b, opts, norm_fn, dual_norm_fn,
               diag_level: str = "full", lo_dtype=None, dual_fn=None):
    """diag_level "full": every LadmmInfo history is filled. "stopping": only
    what the stopping rule needs (D, P, scale_D, scale_P) plus err_x/err_delta
    when references are given; dx/dy/dz come back NaN. Each norm is a full
    reduction over the state, and on a GPU the seven of them cost about as
    much as the KE prox, so the three purely descriptive ones are optional."""
    tau, gamma, alpha = opts.tau, opts.gamma, opts.alpha
    x_ref, delta_ref, delta_mask = opts.x_ref, opts.delta_ref, opts.delta_mask
    x_refs = opts.x_refs or {}
    ref_names = sorted(x_refs)
    rho_norm_fn = opts.rho_norm_fn
    nan = jnp.asarray(np.nan)
    full = diag_level == "full"
    if diag_level not in ("full", "stopping"):
        raise ValueError(f"diag_level must be 'full' or 'stopping' (got {diag_level!r})")

    def step(c: _Carry, aux):
        x_prev, y_prev, By_prev, delta_prev, D_prev = c.x, c.y, c.By, c.delta, c.D_prev

        # ladmm.py recomputes r = A_fn(x_prev) + By_prev - b here, but that is
        # exactly D_prev (the previous iteration's D = A x + B y - b was built
        # from this same x_prev, By_prev and b), so reuse it: one fewer A
        # application per iteration, bitwise the same numbers.
        r = D_prev
        g = At_fn(gamma * r - delta_prev)
        x = prox_f1(x_prev - g * (1.0 / tau), 1.0 / tau, aux)

        # master-copy mode: A_fn and everything downstream see the float32
        # rounding of x; x itself stays float64 in the carry (module docstring)
        Ax = A_fn(_cast(x, lo_dtype))
        z_hat = Ax * alpha + (b - By_prev) * (1.0 - alpha)

        y = solve_y(delta_prev, z_hat, gamma)
        By = B_fn(y)
        D = Ax + By - b
        delta = delta_prev - (z_hat + By - b) * gamma
        if dual_fn is not None:
            # consistent dual (module docstring): the KE gradient at the y
            # actually stored, rather than the algebraically equal
            # delta_prev - gamma*(z_hat - y) whose rounding does not match y's
            delta = dual_fn(y, delta)
        P = (x_prev - x) * tau + At_fn(D_prev - D) * gamma

        err_x = norm_fn(x - x_ref) if x_ref is not None else nan
        if delta_ref is not None:
            diff = delta - delta_ref
            if delta_mask is not None:
                diff = type(diff)(*(getattr(diff, f) * getattr(delta_mask, f)
                                    for f in diff.__dataclass_fields__))
            err_delta = norm_fn(diff)
        else:
            err_delta = nan

        diag = jnp.stack([
            norm_fn(x - x_prev) if full else nan,
            norm_fn(y - y_prev) if full else nan,
            norm_fn(delta - delta_prev) if full else nan,
            norm_fn(D), norm_fn(P),
            jnp.maximum(norm_fn(Ax), norm_fn(y)), dual_norm_fn(At_fn(delta)),
            err_x, err_delta,
        ] + [v for nm in ref_names
             for v in ((lambda d: (norm_fn(d), rho_norm_fn(d) if rho_norm_fn is not None else nan))(x - x_refs[nm]))])
        return _Carry(x, y, By, delta, D), diag

    return step


def ladmm_solve_jit(
    prox_f1: Callable,
    solve_y: Callable,
    A_fn: Callable,
    At_fn: Callable,
    B_fn: Callable,
    b,
    x0,
    y0,
    opts: LadmmOpts,
    aux=None,
    check_every: int = 1,
    diag_level: str = "full",
    master_dtype=None,
    round_x_dtype=None,
    dual_fn=None,
) -> tuple[object, object, object, LadmmInfo]:
    """See ladmm.ladmm_solve; `aux` is forwarded to prox_f1 as its third
    argument, `check_every` is how many iterations run per compiled call,
    `diag_level` is _make_step's (which histories to compute), `master_dtype`
    keeps x and delta in that dtype while the rest runs in b's (module
    docstring); None means everything in b's dtype. `round_x_dtype` is the
    dtype x is rounded to before A_fn (default: b's dtype when master_dtype is
    set, else no rounding) -- settable on its own, e.g. float32 with a float64
    b, to inject exactly that one rounding into an otherwise float64 solve.
    `dual_fn(y, delta_std)` replaces the dual update by a value computed from
    the stored y (the pipelines pass the negative KE gradient, see the module
    docstring's mixed-precision note).

    The compiled chunk is built from this call's closures, so it is compiled
    afresh on every call (~0.5 s on a GH200, reported in
    info.walltime_first_chunk); it is not shared between solves."""
    norm_fn = opts.norm_fn if opts.norm_fn is not None else (lambda v: v.norm_jnp())
    dual_norm_fn = opts.dual_norm_fn if opts.dual_norm_fn is not None else norm_fn
    ref_names = sorted(opts.x_refs or {})
    n_diag = _N_DIAG + 2 * len(ref_names)
    if check_every < 1:
        raise ValueError("check_every must be >= 1")

    if round_x_dtype is None and master_dtype is not None:
        round_x_dtype = b.rho.dtype
    step = _make_step(prox_f1, solve_y, A_fn, At_fn, B_fn, b, opts, norm_fn, dual_norm_fn, diag_level,
                      lo_dtype=round_x_dtype, dual_fn=dual_fn)

    @partial(jax.jit, static_argnums=2)
    def run_chunk(carry, aux, n):
        return lax.scan(lambda c, _: step(c, aux), carry, None, length=n)

    max_iter = opts.max_iter
    delta0 = _cast(type(b).zeros_like(b), master_dtype)
    By0 = B_fn(y0)
    carry = _Carry(_cast(x0, master_dtype), y0, By0, delta0, A_fn(x0) + By0 - b)

    diag_hist = np.zeros((max_iter, n_diag))
    n_iters = max_iter
    converged = False

    t_start = time.perf_counter()
    t_first = 0.0
    it = 0
    while it < max_iter:
        n = min(check_every, max_iter - it)
        carry, diag = run_chunk(carry, aux, n)
        diag = np.asarray(diag)          # (n, n_diag) -- the one host sync per chunk
        if it == 0:
            t_first = time.perf_counter() - t_start
        diag_hist[it:it + n] = diag
        D_k, P_k, sD, sP = diag[:, 3], diag[:, 4], diag[:, 5], diag[:, 6]
        ok = (D_k <= opts.eps_abs + opts.eps_rel * sD) & (P_k <= opts.eps_abs + opts.eps_rel * sP)
        if ok.any():
            n_iters = it + int(np.argmax(ok)) + 1
            converged = True
            break
        it += n
    walltime = time.perf_counter() - t_start

    h = diag_hist[:n_iters]
    info = LadmmInfo(
        dx=h[:, 0].copy(), dy=h[:, 1].copy(), dz=h[:, 2].copy(),
        D=h[:, 3].copy(), P=h[:, 4].copy(),
        scale_D=h[:, 5].copy(), scale_P=h[:, 6].copy(),
        err_x=h[:, 7].copy(), err_delta=h[:, 8].copy(),
        err_x_refs={nm: h[:, _N_DIAG + 2 * i].copy() for i, nm in enumerate(ref_names)},
        err_rho_refs=({nm: h[:, _N_DIAG + 2 * i + 1].copy() for i, nm in enumerate(ref_names)}
                      if opts.rho_norm_fn is not None else {}),
        tau=np.full(n_iters, opts.tau), gamma=np.full(n_iters, opts.gamma),
        iters=n_iters, converged=converged, walltime=walltime,
        walltime_first_chunk=t_first,
    )
    return carry.x, carry.y, carry.delta, info
