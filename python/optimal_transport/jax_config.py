"""Device and precision configuration for the JAX-backed 2D solver.

JAX's platform and x64-precision flags are process-wide settings that only
affect computations that happen *after* they are set. Call configure() as
the very first thing in your script -- before creating any array, and
before importing other `_2d` modules (state_2d, grid_2d, ...) if you can
help it -- so the device/precision choice is in effect everywhere.

Why x64 matters (this is the main GPU memory knob): JAX defaults to
float32 for every array, even one built from a float64 numpy/Python
literal -- silently downcasting. That default is exactly what you want
for GPU memory (half the bytes of float64, matching the mixed-precision
approach in the advisor's Schrodinger-solver code: fp64 only where
precision-critical, fp32 elsewhere), but it will NOT match the float64
reference solutions (analytical_sb_gaussian_2d, the 1D solver, etc.) used
to validate correctness -- float32 alone typically caps agreement at
~1e-6/1e-7 relative error regardless of grid resolution. configure()
defaults to float64 (x64=True) so a fresh port validates against those
references first; switch to x64=False once correctness is established and
you want the smaller memory footprint on GPU.

Why "switch to run on GPU or CPU" needs no array-level code changes: every
`_2d` module below is written against `jax.numpy` directly (no separate
numpy/cupy code path to maintain) -- JAX dispatches each op to whatever
backend is configured, CPU or GPU, from the exact same Python. `device`
here only controls *which* backend that is.
"""
from __future__ import annotations


def configure(device: str = "auto", x64: bool = True) -> None:
    """Set the JAX backend platform and floating-point precision.

    device: "cpu", "gpu", "tpu", or "auto" (let JAX pick its default --
        a GPU if a CUDA-enabled jaxlib is installed and one is visible,
        else CPU).
    x64: enable float64 (see module docstring).
    """
    import jax

    jax.config.update("jax_enable_x64", x64)

    if device != "auto":
        if device not in ("cpu", "gpu", "tpu"):
            raise ValueError(f"device must be 'cpu', 'gpu', 'tpu', or 'auto' (got {device!r})")
        jax.config.update("jax_platform_name", device)
        got = jax.default_backend()
        if got != device:
            raise RuntimeError(
                f"Requested JAX device {device!r} but the available backend is "
                f"{got!r} (devices: {jax.devices()}). For device='gpu' you likely "
                f"need a CUDA-enabled jaxlib, e.g. `pip install jax[cuda12]` on a "
                f"machine with an NVIDIA GPU -- the CPU-only `jax` wheel installed "
                f"in this project's .venv cannot drive a GPU no matter what this "
                f"flag is set to."
            )
