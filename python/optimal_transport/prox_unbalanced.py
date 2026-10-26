"""Prox of KE + reaction on the cell-centre grid:

    f2(rho, m, s) = sum (m^2 + s^2) / (2 rho),      s = delta * zeta.

This is the perspective function of |(m, s)|^2/2, so the pointwise prox is
prox_ke_cc's cubic with m^2 replaced by m^2 + s^2, and s follows like m.
"""
import numpy as np

from .prox import solve_cubic
from .state_unbalanced import StateU


def prox_ke_reaction_cc(x_in: StateU, sigma: float) -> StateU:
    rho_c, mx_c, s_c = x_in.rho, x_in.mx, x_in.s
    q2 = mx_c**2 + s_c**2

    b = 2 * sigma - rho_c
    c = sigma**2 - 2 * sigma * rho_c
    d = -sigma * (sigma * rho_c + 0.5 * q2)

    rho_new = solve_cubic(1.0, b, c, d)
    fac = rho_new / (rho_new + sigma)
    mx_new = fac * mx_c
    s_new = fac * s_c

    neg = rho_new <= 1e-12
    rho_new = np.where(neg, 0.0, rho_new)
    mx_new = np.where(neg, 0.0, mx_new)
    s_new = np.where(neg, 0.0, s_new)
    return StateU(rho_new, mx_new, s_new)
