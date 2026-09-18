"""Primal/dual state (rho, mx) with arithmetic, for the LADMM iteration.

Replaces the s_add/s_sub/s_scale/s_zeros/s_norm generic-struct helpers in
matlab/shared/1d/utils/{ladmm_solve,admm_solve}.m: those exist only because
MATLAB structs don't support operator overloading. Since our state always
has exactly the two fields (rho, mx), a small dataclass with __add__/__sub__/
__mul__/norm() replaces that whole utility layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    rho: np.ndarray
    mx: np.ndarray

    def __add__(self, other: State) -> State:
        return State(self.rho + other.rho, self.mx + other.mx)

    def __sub__(self, other: State) -> State:
        return State(self.rho - other.rho, self.mx - other.mx)

    def __mul__(self, scalar: float) -> State:
        return State(scalar * self.rho, scalar * self.mx)

    __rmul__ = __mul__

    def norm(self) -> float:
        return float(np.sqrt(np.sum(self.rho**2) + np.sum(self.mx**2)))

    @classmethod
    def zeros_like(cls, other: State) -> State:
        return State(np.zeros_like(other.rho), np.zeros_like(other.mx))
