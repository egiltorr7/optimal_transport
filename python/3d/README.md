# 3D optimal transport / Schrödinger bridge solver

Port of `python/2d` to three spatial dimensions, following the same
copy-and-extend convention that `2d` follows from the 1D code in
`python/optimal_transport`. Uniform grids only — the non-uniform-time
(`*_nu_2d`) track has no 3D sibling yet.

## Layout

```
setup/
  jax_config.py     device (CPU/GPU) + float64 switch -- call FIRST
  state_3d.py       State3D(rho, mx, my, mz), a JAX pytree
  grid_3d.py        Problem3D: grid, marginals, DCT eigenvalues
  problems_3d.py    prob_gaussian_3d + closed-form analytical_sb_gaussian_3d
  operators_3d.py   staggered interp/deriv operators in t, x, y, z
  projection_3d.py  FP-constraint projection: DCT in (x,y,z) + Thomas in t
  prox_3d.py        exact KE prox (cubic in rho)
  pipeline_3d.py    wires the above into linearized ADMM
  ladmm.py          the LADMM loop itself -- dimension-agnostic
  state.py          1D State, imported by ladmm.py for type hints only
scripts/
  sb_gaussian_3d.py 3D Schrödinger bridge demo vs the closed form
  validate_3d.py    correctness suite (see below)
```

## What actually changed from 2D

Very little, which is the point:

- **operators**: one more axis (`_node_z`, `*_z_at_{phi,m}`). Each operator
  is still a two-point average/difference along a single axis.
- **projection**: `lambda_xyz = lambda_x + lambda_y + lambda_z`. The discrete
  Laplacian's spectrum is additive across axes, so the per-mode tridiagonal
  system in *t* is unchanged — only the number of terms in that sum differs.
  The singular DC mode moves from `(0,0)` to `(0,0,0)`.
- **prox**: `|m|² = mx²+my²+mz²`. `solve_cubic` is untouched.
- **ladmm.py**: no change, beyond making the optional `delta_mask` branch
  field-generic (the 1D/2D copies hardcode `rho`/`mx` there and would
  silently drop `my`/`mz`).

## Running

```bash
../../.venv/bin/python scripts/validate_3d.py       # ~15 min on CPU
../../.venv/bin/python scripts/sb_gaussian_3d.py    # demo + figure
../../.venv/bin/python scripts/sb_gaussian_3d.py --nx 32 --ny 32 --nz 32 --nt 32
```

## Validation

`scripts/validate_3d.py` checks, in order:

1. `_dct_axis` against `scipy.fft.dct` on 4-D arrays along every axis, and
   `dct3_xyz` round-trip.
2. Adjoint identities for every interp/deriv pair.
3. The batched Thomas solve against a dense `np.linalg.solve`.
4. The FP projection lands on the constraint set and is idempotent.
5. **Reduction to 2D.** On a problem whose marginals are uniform in *z*, the
   exact 3D solution is the 2D one, constant along *z*, with `mz ≡ 0`. The
   3D solver reproduces the already-validated 2D solver to ~1e-15. This is
   the check that actually pins down the z-axis operators, the extra
   divergence term, `lambda_xyz`, and `mz`'s coupling in the prox — an
   isotropic problem would pass even with the axes mixed up.
6. `analytical_sb_gaussian_3d` satisfies the discrete FP equation to O(h²),
   checked under refinement and *without* the solver.
7. The solve against that reference.

## Test-problem caveat (carried over from 2D, and fixed here)

`analytical_sb_gaussian_3d` is the **free-space** SB solution and does not
satisfy the no-flux walls the discretization imposes. How much that matters
is set by how much mass reaches the wall, and it puts a floor under every
comparison against the reference that grid refinement cannot cross.

With 2D's defaults (means 0.35/0.65, σ=0.1) the wall density is ~2e-2 of the
peak and the reference's FP residual stalls at ~5e-2, then *rises* past
n=32. `prob_gaussian_3d` therefore defaults to means 0.4/0.6 and σ=0.08,
putting the wall density at ~1e-4 and restoring clean O(h²).

## Memory

This is the one genuinely new constraint. Every working array is
`(nt, nx, ny, nz)`; the projection holds roughly eight live, so float64 peak
memory is about `64 · nt·nx·ny·nz` bytes:

| grid | per array | rough peak |
|------|-----------|------------|
| 16⁴  | 0.5 MB    | ~4 MB      |
| 32⁴  | 8 MB      | ~64 MB     |
| 64⁴  | 134 MB    | ~1 GB      |
| 96⁴  | 680 MB    | ~5.4 GB    |

Two knobs, both deliberately left for later:

- `jax_config.configure(x64=False)` halves it (but see that module on why
  float64 is the right default while validating).
- The Thomas sweeps in `projection_3d.py` build Python lists of length `nt`;
  a `jax.lax.scan` version would trace once instead of unrolling `nt` steps
  and let XLA reuse buffers. Already flagged as a follow-up in the 2D code.
