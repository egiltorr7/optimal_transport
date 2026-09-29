"""Mixed-precision helpers for the 2D LADMM solver.

Three precision modes are supported by the pipelines (via the dtype of the
Problem they are handed) and by scripts/run_suite_2d.py:

  fp64    everything float64 (the reference).
  fp32    everything float32 (x64 off): 1.05-1.6x faster and 2.2x less
          memory than fp64, but the LADMM iteration stalls at ~1e-4 relative
          at n=256 and the clustered grid blows up at vareps >= 0.1 (memory:
          fp32-2d-suite-results).
  mixed   the LADMM state (x, y, delta, D), the consensus maps A/A^T, the KE
          prox and the norms in float32; the FP projection -- residual with
          its 1/dt time differences, the per-mode time solve, the R^T
          correction -- in float64. `mixed_projection` below is the whole
          mechanism: the float32 iterate is cast up, projected with the
          float64 precompute on the float64 problem, and the result cast back.

Why the projection is the piece that needs float64 (and the rest does not):
the projection's output is x + R^T T^{-1}(-R x), and the R / R^T factors are
first differences in time divided by dt. Both amplify rounding by 1/dt: the
residual's noise is harmless (T^{-1} R^T smooths it, the 1/j^2 tail of the
time spectrum sums to a constant), but R^T applied to phi -- a large, smooth
potential of size ~|g|/tau -- turns float32 rounding of phi into an error of
order eps32 * |phi| / dt in the iterate, i.e. growing linearly in n. That is
the n-dependent stall the pure fp32 suite measured (6e-6, 1.3e-5, 4e-5,
1.4e-4 at n = 32..256). The KE prox, the interpolations and the dual update
are pointwise or O(1)-conditioned, so their float32 error is a white
eps32-relative noise the iteration averages out. On the clustered grid the
Thomas sweep on the N-weighted per-mode system (entries spanning
1/(dt_min^2 dx dy) to 1/dt_max^2) is where fp32 actually diverged; it is
float64 here too.

  mixedm  mixed + float64 "master copies" of the two accumulators x and
          delta (LadmmConfig.master_dtype=jnp.float64, ladmm_jit.py). Also
          stalls (n=128: 5.6e-4): the floor is not the storage of x.
  mixedc  the working design. Additionally the KE prox ARITHMETIC runs in
          float64 (prox_dtype) with its output stored float32 (y_dtype), and
          the dual is computed as -grad KE at the stored float32 y, in
          float64 (consistent_dual, ladmm_jit.py's dual_fn). Matches fp64 to
          all printed digits at n=128 (3.377e-4, eps=0) and on the clustered
          grid (2.485e-3, eps=0.1, 150k), at ~1.1-1.3x the speed and 60% of
          the memory of fp64. What stays float32: y, By, D, the consensus map
          A on the rounded x, the reference-error diagnostics.

Why a float32 y alone costs 5e-4 (diagnosed 2026-09-28 with --precision
fp64rx / fp64px in run_suite_2d.py): the standard dual update
delta - gamma*(z_hat - y) equals -grad KE(y) at the exact prox output. When y
is stored float32 the dual is the gradient at a point 6e-8 away from the y
the constraint A x = y actually uses -- a frozen inconsistency in the KKT
system that the problem's ill-conditioned slow modes (the ones LADMM needs
1e5 iterations for) amplify a thousandfold. Rounding x before A has no such
inconsistency (the prox is evaluated at the same rounded x the constraint
sees) and is measurably harmless. The rule for any further mixed-precision
work (3D): a variable may be stored in float32 only if every optimality
condition is evaluated AT the stored value.

Mixed precision needs x64 ENABLED (jax_enable_x64=True); the float32 half is
float32 because every array it touches is float32, not because of a global
flag. The pipelines and operators build their zero planes and initial
iterates in the dtype of problem.rho0 for exactly this reason.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from .state_2d import State2D


def cast_state(s: State2D, dtype) -> State2D:
    return State2D(s.rho.astype(dtype), s.mx.astype(dtype), s.my.astype(dtype))


def cast_problem(problem, dtype):
    """Copy of a Problem2D / ProblemNU2D with every jnp array field cast.
    `ops` is NOT carried over: rebuild it with build_operators_(nu_)2d on the
    returned problem so its boundary planes take the new dtype."""
    changes = {f.name: getattr(problem, f.name).astype(dtype)
               for f in dataclasses.fields(problem)
               if isinstance(getattr(problem, f.name), jnp.ndarray)}
    if "ops" in {f.name for f in dataclasses.fields(problem)}:
        changes["ops"] = None
    return dataclasses.replace(problem, **changes)


def mixed_projection(projection, problem_hi, lo_dtype=jnp.float32, hi_dtype=jnp.float64):
    """Wrap `projection(x, problem, vareps, ps)` so it runs in hi_dtype on
    problem_hi (whose arrays and `ops` are hi_dtype) and returns lo_dtype
    (lo_dtype=None: returns hi_dtype -- the master-copy mode, where the
    iterate handed in is already hi_dtype and the input cast is a no-op).

    The returned callable has the projection's signature; the `problem`
    argument the pipeline passes (the lo_dtype one) is ignored in favour of
    problem_hi, and `ps` must be the hi_dtype precompute. Under jit the two
    casts are fused into the neighbouring elementwise ops."""
    def proj_mixed(x_in: State2D, problem, vareps: float, ps) -> State2D:
        out = projection(cast_state(x_in, hi_dtype), problem_hi, vareps, ps)
        return out if lo_dtype is None else cast_state(out, lo_dtype)
    return proj_mixed
