"""Kinetic-energy proximal operator on the cell-centered grid.

Port of matlab/shared/utils/solve_cubic.m and
matlab/discretize_first/1d/prox/prox_ke_cc.m.

(prox_ke_cc.m takes a `problem` argument but never uses it in the body --
dropped here.)
"""
import numpy as np

from .state import State


def solve_cubic(a, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Real root of a*x^3 + b*x^2 + c*x + d = 0, vectorized elementwise.

    `a` may be a scalar (as used by prox_ke_cc, where a=1 always) or an
    array broadcastable against b/c/d. Follows the case split in
    solve_cubic.m (depressed cubic via Cardano's formula, with the
    trigonometric form for the three-real-roots case).
    """
    b = b / a
    c = c / a
    d = d / a

    p = c - b**2 / 3
    q = 2 * b**3 / 27 - b * c / 3 + d
    delta = q**2 / 4 + p**3 / 27

    x = np.zeros_like(b, dtype=float)

    ind = p == 0
    x[ind] = -np.cbrt(q[ind])

    ind = (p != 0) & (delta == 0)
    x[ind] = np.maximum(3 * q[ind] / p[ind], -3 * q[ind] / p[ind] / 2)

    # delta>0 (single real root): the naive cbrt(-q/2-s) + cbrt(-q/2+s) from
    # solve_cubic.m suffers catastrophic cancellation whenever |p| << |q|,
    # since s = sqrt(q^2/4 + p^3/27) then rounds to exactly |q|/2 and one of
    # the two cbrt arguments rounds to 0. Fix: compute only the
    # well-conditioned (larger-magnitude) cube root directly, and recover
    # the other root via the exact identity u*v = -p/3 (both cube roots u,v
    # satisfy u^3+v^3=-q, uv=-p/3 by construction of Cardano's substitution).
    ind = (p != 0) & (delta > 0)
    pi, qi = p[ind], q[ind]
    s = np.sqrt(delta[ind])
    a3 = np.where(qi >= 0, -qi / 2 - s, -qi / 2 + s)
    a = np.cbrt(a3)
    x[ind] = a - pi / (3 * a)

    ind = (p != 0) & (delta < 0)
    r = 2 * np.sqrt(-p[ind] / 3)
    s = 3 * q[ind] / (p[ind] * r)
    theta = np.real(np.arccos(s.astype(complex)) / 3)
    x[ind] = r * np.cos(theta)

    x = x - b / 3
    return x


def prox_ke_cc(x_in: State, sigma: float) -> State:
    """Exact prox of KE(rho,m) = sum m^2/(2 rho) on the collocated cell-centre grid.

    Solves, pointwise:
      argmin_{rho,m}  m^2/(2 rho) + (1/(2 sigma)) * ||(rho,m) - x_in||^2
    via the closed-form cubic in rho; m follows in closed form from rho.
    """
    rho_c = x_in.rho
    mx_c = x_in.mx

    b = 2 * sigma - rho_c
    c = sigma**2 - 2 * sigma * rho_c
    d = -sigma * (sigma * rho_c + 0.5 * mx_c**2)

    rho_new = solve_cubic(1.0, b, c, d)
    mx_new = rho_new * mx_c / (rho_new + sigma)

    neg = rho_new <= 1e-12
    rho_new = np.where(neg, 0.0, rho_new)
    mx_new = np.where(neg, 0.0, mx_new)

    return State(rho_new, mx_new)
