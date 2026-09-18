"""Primal/dual state (rho, mx, my) for the 2D LADMM iteration.

2D analogue of state.py's State -- adds the second momentum component my
for the extra spatial axis. Backed by jax.numpy arrays instead of numpy,
so the same arithmetic dispatches to CPU or GPU depending on how
jax_config.configure() set up the JAX backend; see that module.

Registered as a JAX pytree (register_dataclass) so State2D values can be
passed through jax.jit/vmap/grad transparently later, even though the
current ladmm_solve loop (ladmm.py, shared with the 1D code -- see the
duck-typing note there) runs eagerly rather than jitted.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class State2D:
    rho: jnp.ndarray
    mx: jnp.ndarray
    my: jnp.ndarray

    def __add__(self, other: State2D) -> State2D:
        return State2D(self.rho + other.rho, self.mx + other.mx, self.my + other.my)

    def __sub__(self, other: State2D) -> State2D:
        return State2D(self.rho - other.rho, self.mx - other.mx, self.my - other.my)

    def __mul__(self, scalar: float) -> State2D:
        return State2D(scalar * self.rho, scalar * self.mx, scalar * self.my)

    __rmul__ = __mul__

    def norm(self) -> float:
        return float(jnp.sqrt(jnp.sum(self.rho**2) + jnp.sum(self.mx**2) + jnp.sum(self.my**2)))

    @classmethod
    def zeros_like(cls, other: State2D) -> State2D:
        return cls(jnp.zeros_like(other.rho), jnp.zeros_like(other.mx), jnp.zeros_like(other.my))


jax.tree_util.register_dataclass(State2D, data_fields=["rho", "mx", "my"], meta_fields=[])
