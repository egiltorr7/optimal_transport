# 1D Schrödinger Bridge / Optimal Transport — Documentation

## Contents

- [Problem Setup](problem_setup.md) — Continuous formulation, objective, constraints
- [Discretization](discretization.md) — Staggered grid, finite differences, interpolation
- [Algorithm](algorithm.md) — ADMM iteration, proximal step, projection step
- [Implementation](implementation.md) — MATLAB code structure, key functions, parameters

## Quick Summary

This code solves the **entropic optimal transport / Schrödinger bridge** problem in 1D using
an ADMM (Alternating Direction Method of Multipliers) algorithm on a **staggered space-time grid**.

The problem is to find a curve of probability densities `rho(t,x)` and a momentum field
`m(t,x)` that minimize the kinetic energy `∫∫ m²/rho dt dx` subject to the continuity
equation `∂_t rho + ∂_x m = 0` with prescribed boundary densities `rho(0,·) = rho0` and
`rho(1,·) = rho1`. A diffusion regularization (biharmonic) with parameter `vareps` is
optionally included.
