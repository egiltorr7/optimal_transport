# Prompt for Claude Code: 1D Schrödinger Bridge Solver

## Instructions

Please read `context.md` and `skill.md` carefully before writing any code. Then implement a 1D Schrödinger Bridge solver as described.

---

## Task

Implement `sb_admm_1d.py` — a clean, well-documented Python solver for the 1D dynamic Schrödinger Bridge problem using Linearized ADMM (Formulation 2, two-grid approach).

---

## Requirements

### Must have:
1. All discrete operators $D_t$, $A_t$, $D_x$, $L_x$ built correctly as described in `context.md`
2. Correct interpolation operators $A_\rho$ and $A_m$ from collocated to staggered grid
3. Pointwise proximal operator for the kinetic energy (cubic solve)
4. Efficient $M_{\text{SB}}$ solver using DCT in space + DST + Woodbury rank-2 correction in time
5. Full linearized ADMM loop with convergence tracking
6. Validation tests (operator sizes, solver accuracy, prox correctness)
7. All three test cases from `context.md`
8. All five plots described in `skill.md`

### Must verify:
- When $\varepsilon = 0$: solution matches standard Benamou-Brenier OT
- Primal and dual residuals converge to below $10^{-4}$
- Marginal constraints satisfied: $\rho(0,\cdot) \approx \rho_0$ and $\rho(1,\cdot) \approx \rho_1$
- Fokker-Planck constraint satisfied at convergence

### Parameters to use:
- $N_t = N_x = 32$ for initial testing
- $\varepsilon \in \{0.0, 0.05, 0.1\}$
- $\gamma = 1.0$, $\tau = 2.0$ (or tune if needed)
- Max iterations: 1000
- Tolerance: $10^{-5}$

---

## Code Quality

- Add docstrings to all functions
- Add inline comments explaining key steps
- Print convergence info every 50 iterations
- Use `numpy` vectorization — no Python loops over grid points
- Structure the file so each component can be tested independently

---

## Deliverables

1. `sb_admm_1d.py` — the full solver
2. A brief printout showing:
   - Operator sizes
   - Solver validation results
   - Convergence achieved (iterations, final residuals)
   - Objective value at convergence
3. Saved plots as PNG files

---

## Notes

- The proximal operator of the kinetic energy is the most delicate step — take care with the cubic solve and positivity of $\rho$
- The Woodbury correction handles the rank-2 boundary perturbation in the temporal operator — make sure the $2\times 2$ system is correctly assembled
- Test the $M_{\text{SB}}$ solver independently before plugging into ADMM
- If you get numerical issues for large $\varepsilon$, check eigenvalues of $T^k$ for potential indefiniteness
