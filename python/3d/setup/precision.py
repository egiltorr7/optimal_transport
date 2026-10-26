"""Mixed-precision helpers for the 3D LADMM solver -- 2d/setup/precision.py
made rank-agnostic (State3D has four components; everything below goes
through the dataclass fields instead of naming them).

Precision modes, as established on the 2D grid-convergence suite
(docs/gpu_mixed_precision_2d.tex, Section 6):

  fp64    everything float64 (the reference).
  fp32    everything float32: 1.05-1.6x faster, 2.2x less memory, but the
          density error floors at ~1e-3 on fine grids, mass conservation
          degrades to 1e-4..1e-2 and the clustered time grid diverges.
  mixed   float32 iterate, float64 projection (`mixed_projection`): fixes mass
          and the divergence, still floors at ~5e-4.
  mixedm  mixed + float64 master copies of x and delta: same floor.
  mixedc  the working design: float32 for y, By, D and the consensus map A
          on the rounded x; float64 for x, delta, the projection, A^T and
          the KE prox ARITHMETIC (output stored float32); and the dual
          computed as -grad KE at the STORED float32 y, in float64
          (LadmmConfig(master_dtype=f64, prox_dtype=f64, y_dtype=f32,
          consistent_dual=True)). Matches float64 to every printed digit on
          all 28 2D suite points at 60-71% of the memory.

Why only mixedc works: the standard dual update delta - gamma*(z_hat - y)
equals -grad KE(y) at the EXACT prox output; with y stored float32 the dual
is the gradient at a point 6e-8 away from the y the constraint A x = y uses,
a frozen inconsistency in the KKT system that the problem's slow modes
amplify ~1e3x. Rounding x before A has no such inconsistency (the prox is
evaluated at the same rounded x the constraint sees) and is harmless. Rule:
a variable may be stored in reduced precision only if every optimality
condition involving it is evaluated at the stored value.

Mixed precision needs x64 ENABLED (jax_enable_x64=True); the float32 half is
float32 because every array it touches is float32 -- the pipeline and the
operators build their zero planes and initial iterates in problem.rho0.dtype
for exactly this reason. A single float64 zero plane promotes the whole solve.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp


def cast_state(s, dtype):
    """Copy of a State2D/State3D with every component cast (no-op if already dtype)."""
    if getattr(s.rho, "dtype", None) == dtype:
        return s
    return type(s)(*(getattr(s, f).astype(dtype) for f in s.__dataclass_fields__))


def cast_problem(problem, dtype):
    """Copy of a Problem3D (or any problem dataclass) with every jnp array field
    cast. `ops` is NOT carried over: rebuild it with build_operators_3d on the
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
    (lo_dtype=None: returns hi_dtype -- the master-copy modes, where the
    iterate handed in is already hi_dtype and the input cast is a no-op).

    The `problem` argument the pipeline passes (the lo_dtype one) is ignored
    in favour of problem_hi, and `ps` must be the hi_dtype precompute. Under
    jit the casts fuse into the neighbouring elementwise kernels."""
    def proj_mixed(x_in, problem, vareps: float, ps):
        out = projection(cast_state(x_in, hi_dtype), problem_hi, vareps, ps)
        return out if lo_dtype is None else cast_state(out, lo_dtype)
    return proj_mixed


def configure_precision(precision: str, problem, ps, proj, build_ops):
    """One-stop setup of a precision mode for a solve.

    Given the float64 problem (with ops), its float64 precompute `ps`, the
    projection callable and the operator builder, returns
    (problem_for_pipeline, ps, proj, ladmm_kwargs) where ladmm_kwargs are the
    LadmmConfig fields (master_dtype, prox_dtype, y_dtype, consistent_dual,
    round_x_dtype) for that mode. For "fp32" the caller must switch x64 OFF
    afterwards (the projection modules build float64 constants internally);
    every other mode keeps x64 on.
    """
    f32, f64 = jnp.float32, jnp.float64
    kw = dict(master_dtype=None, prox_dtype=None, y_dtype=None, consistent_dual=False, round_x_dtype=None)
    if precision == "fp64":
        return problem, ps, proj, kw
    if precision == "fp32":
        p32 = cast_problem(problem, f32)
        p32.ops = build_ops(p32)
        ps32 = _cast_tree(ps, f32)
        return p32, ps32, proj, kw
    if precision in ("mixed", "mixedm", "mixedc"):
        p32 = cast_problem(problem, f32)
        p32.ops = build_ops(p32)
        master = f64 if precision in ("mixedm", "mixedc") else None
        proj_m = mixed_projection(proj, problem, lo_dtype=(None if master else f32), hi_dtype=f64)
        kw["master_dtype"] = master
        if precision == "mixedc":
            kw.update(prox_dtype=f64, y_dtype=f32, consistent_dual=True)
        return p32, ps, proj_m, kw
    if precision == "fp64rx":     # diagnostic: float64 but x rounded to float32 before A
        kw.update(master_dtype=f64, round_x_dtype=f32)
        return problem, ps, proj, kw
    if precision == "fp64px":     # diagnostic: float64 but the KE prox / y in float32
        kw.update(master_dtype=f64, round_x_dtype=f64, prox_dtype=f32, y_dtype=f32)
        return problem, ps, proj, kw
    raise ValueError(f"unknown precision {precision!r}")


PRECISIONS = ("fp64", "fp32", "mixed", "mixedm", "mixedc", "fp64rx", "fp64px")


def _cast_tree(tree, dtype):
    import jax
    return jax.tree_util.tree_map(lambda a: a.astype(dtype) if hasattr(a, "astype") else a, tree)
