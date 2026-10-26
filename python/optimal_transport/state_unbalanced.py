"""Primal/dual state (rho, mx, s) for the unbalanced (source-term) LADMM.

s = delta * zeta is the SCALED source: with it the reaction cost
(delta^2/2) zeta^2/rho becomes s^2/(2 rho), the same perspective function as
the kinetic energy, so the KE prox applies unchanged with s as one more
momentum component.  s lives on the cell-centre grid (nt, nx) in both the
x- and y-copies (the FP residual lives there too, so A is the identity on s).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StateU:
    rho: np.ndarray
    mx: np.ndarray
    s: np.ndarray

    def __add__(self, other: StateU) -> StateU:
        return StateU(self.rho + other.rho, self.mx + other.mx, self.s + other.s)

    def __sub__(self, other: StateU) -> StateU:
        return StateU(self.rho - other.rho, self.mx - other.mx, self.s - other.s)

    def __mul__(self, scalar: float) -> StateU:
        return StateU(scalar * self.rho, scalar * self.mx, scalar * self.s)

    __rmul__ = __mul__

    def norm(self) -> float:
        return float(np.sqrt(np.sum(self.rho**2) + np.sum(self.mx**2) + np.sum(self.s**2)))

    @classmethod
    def zeros_like(cls, other: StateU) -> StateU:
        return StateU(np.zeros_like(other.rho), np.zeros_like(other.mx), np.zeros_like(other.s))
