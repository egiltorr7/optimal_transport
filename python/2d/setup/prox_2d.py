"""Kinetic-energy proximal operator on the 2D collocated (cell-centre) grid.

2D analogue of prox.py's prox_ke_cc / solve_cubic. KE(rho,mx,my) =
(mx^2+my^2)/(2*rho) has exactly the same closed-form cubic-in-rho prox as
the 1D case (matlab/shared/2d/prox/prox_ke_exact.m makes the same
substitution: |m|^2 = mx^2+my^2 in place of the 1D m^2), so only the
combined squared-momentum term changes; solve_cubic itself is untouched
math.

Rewritten against jax.numpy rather than reusing prox.py's numpy version
because JAX arrays are immutable: prox.py's solve_cubic writes into `x`
with boolean-mask assignment (`x[ind] = ...`), which has no JAX
equivalent that traces under jit. Every branch below uses jnp.where
instead -- evaluated for all elements and then selected, rather than only
computed for the masked subset, which is the standard (and jit-safe)
vectorized-branching idiom in JAX.
"""
from __future__ import annotations

import jax.numpy as jnp

from .state_2d import State2D


def solve_cubic(b: jnp.ndarray, c: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Real root of x^3 + b*x^2 + c*x + d = 0, vectorized via jnp.where.

    Same depressed-cubic (Cardano) derivation as prox.py's solve_cubic,
    with `a` dropped (always 1 here), including its cancellation-safe
    single-real-root branch (computing only the well-conditioned cube
    root and recovering the other via u*v = -p/3, rather than the naive
    cbrt(-q/2-s)+cbrt(-q/2+s) that cancels catastrophically when |p|<<|q|)
    and its complex-arccos trick for the three-real-roots branch --
    replaced here by an equivalent (see scratchpad derivation) real
    jnp.clip(s, -1, 1) before jnp.arccos, since real(arccos(s+0j)) for
    |s|>1 equals arccos(clip(s,-1,1)) exactly (0 or pi at the clipped
    ends), and clipping needs no complex dtype.
    """
    p = c - b**2 / 3
    q = 2 * b**3 / 27 - b * c / 3 + d
    delta = q**2 / 4 + p**3 / 27

    is_p0 = p == 0
    p_safe = jnp.where(is_p0, 1.0, p)

    x_p0 = -jnp.cbrt(q)

    x_delta0 = jnp.maximum(3 * q / p_safe, -3 * q / p_safe / 2)

    s_pos = jnp.sqrt(jnp.where(delta > 0, delta, 0.0))
    a3 = jnp.where(q >= 0, -q / 2 - s_pos, -q / 2 + s_pos)
    cr = jnp.cbrt(a3)
    cr_safe = jnp.where(cr == 0, 1.0, cr)
    x_deltapos = cr - p_safe / (3 * cr_safe)

    p_neg_safe = jnp.where(p < 0, p, -1.0)  # forced strictly negative -> safe sqrt below
    r = 2 * jnp.sqrt(-p_neg_safe / 3)
    r_safe = jnp.where(r == 0, 1.0, r)
    s_arg = jnp.clip(3 * q / (p_neg_safe * r_safe), -1.0, 1.0)
    theta = jnp.arccos(s_arg) / 3
    x_deltaneg = r * jnp.cos(theta)

    x = jnp.where(
        is_p0,
        x_p0,
        jnp.where(delta == 0, x_delta0, jnp.where(delta > 0, x_deltapos, x_deltaneg)),
    )
    return x - b / 3


def prox_ke_cc_2d(x_in: State2D, sigma: float) -> State2D:
    """Exact prox of KE(rho,mx,my) = sum (mx^2+my^2)/(2 rho) on the collocated grid.

    Solves, pointwise:
      argmin_{rho,mx,my}  (mx^2+my^2)/(2 rho) + (1/(2 sigma))*||(.)-x_in||^2
    via the closed-form cubic in rho; mx, my follow in closed form from rho.
    """
    rho_c = x_in.rho
    mx_c = x_in.mx
    my_c = x_in.my
    m2_c = mx_c**2 + my_c**2

    b = 2 * sigma - rho_c
    c = sigma**2 - 2 * sigma * rho_c
    d = -sigma * (sigma * rho_c + 0.5 * m2_c)

    rho_new = solve_cubic(b, c, d)
    scale = rho_new / (rho_new + sigma)
    mx_new = scale * mx_c
    my_new = scale * my_c

    neg = rho_new <= 1e-12
    rho_new = jnp.where(neg, 0.0, rho_new)
    mx_new = jnp.where(neg, 0.0, mx_new)
    my_new = jnp.where(neg, 0.0, my_new)

    return State2D(rho_new, mx_new, my_new)
