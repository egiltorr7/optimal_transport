"""Primal/dual state (rho, mx, my, mz) for the 3D LADMM iteration.

3D analogue of state_2d.py's State2D -- adds the third momentum component
mz for the extra spatial axis. Backed by jax.numpy arrays, so the same
arithmetic dispatches to CPU or GPU depending on how jax_config.configure()
set up the JAX backend; see that module.

Registered as a JAX pytree (register_dataclass) so State3D values can be
passed through jax.jit/vmap/grad transparently later, even though the
current ladmm_solve loop (ladmm.py, shared with the 1D/2D code -- see the
duck-typing note there) runs eagerly rather than jitted.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class State3D:
    rho: jnp.ndarray
    mx: jnp.ndarray
    my: jnp.ndarray
    mz: jnp.ndarray

    def __add__(self, other: State3D) -> State3D:
        return State3D(self.rho + other.rho, self.mx + other.mx,
                       self.my + other.my, self.mz + other.mz)

    def __sub__(self, other: State3D) -> State3D:
        return State3D(self.rho - other.rho, self.mx - other.mx,
                       self.my - other.my, self.mz - other.mz)

    def __mul__(self, scalar: float) -> State3D:
        return State3D(scalar * self.rho, scalar * self.mx,
                       scalar * self.my, scalar * self.mz)

    __rmul__ = __mul__

    def norm(self) -> float:
        return float(jnp.sqrt(jnp.sum(self.rho**2) + jnp.sum(self.mx**2)
                              + jnp.sum(self.my**2) + jnp.sum(self.mz**2)))

    @classmethod
    def zeros_like(cls, other: State3D) -> State3D:
        return cls(jnp.zeros_like(other.rho), jnp.zeros_like(other.mx),
                   jnp.zeros_like(other.my), jnp.zeros_like(other.mz))


jax.tree_util.register_dataclass(State3D, data_fields=["rho", "mx", "my", "mz"], meta_fields=[])
