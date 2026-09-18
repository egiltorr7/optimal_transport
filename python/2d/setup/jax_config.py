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


def configure(device: str = "auto", x64: bool = True, gpu_id: int | None = None,
              preallocate: bool | None = None, mem_fraction: float | None = None) -> None:
    """Set the JAX backend platform, the GPU to use, and floating-point precision.

    device: "cpu", "gpu", "tpu", or "auto" (let JAX pick its default --
        a GPU if a CUDA-enabled jaxlib is installed and one is visible,
        else CPU).
    x64: enable float64 (see module docstring).
    gpu_id: which physical GPU to use on a multi-GPU machine, by setting
        CUDA_VISIBLE_DEVICES before JAX initializes. This is the right lever
        rather than jax.devices()[i] + device_put, for two reasons. First, the
        _2d modules are written against jax.numpy with no explicit placement,
        so there is no single place to put a device argument. Second and more
        importantly on a SHARED machine: JAX preallocates a large fraction of
        memory on every GPU it can see, at initialization -- so a run that
        "uses GPU 2" while seeing four of them still squats on all four.
        Hiding the rest is what actually frees them for other people. After
        this, JAX sees exactly one GPU and numbers it 0, whatever its physical
        id. Overrides CUDA_VISIBLE_DEVICES if that is already set.
    preallocate: False disables JAX's preallocation entirely (memory is then
        grown on demand, slower but neighbourly). None leaves JAX's default.
    mem_fraction: cap preallocation at this fraction of the card, e.g. 0.4 to
        share one GPU with another job. None leaves JAX's default (~0.75).

    All three GPU settings are environment variables that JAX reads ONCE at
    initialization, so they are applied here before `import jax` -- which is why
    this function, and not the caller, does that import.
    """
    import os

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if preallocate is False:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)

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
