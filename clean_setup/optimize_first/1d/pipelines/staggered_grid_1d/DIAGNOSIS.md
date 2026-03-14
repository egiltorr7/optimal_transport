# Diagnosis of 1D Optimal Transport Issues

## Problem Statement
- Gaussian translation not showing correct behavior
- Convergence stalling at ||u_{k+1} - u_k|| ≈ 1e-7
- Not reaching numerical optimizer

## Root Cause: INSUFFICIENT MARGIN IN STEP SIZE

### Correct Convergence Condition (Fang et al.)
Following "Generalized ADMM" paper, the condition is:
```
τ/γ > ||A^T A||_2
```
where A is the linear operator (interpolation operators).

### Your Original Parameters
```matlab
gamma = 10
tau   = gamma / 0.99  ≈ 10.1
τ/γ ≈ 1.0101
```

For interpolation operators: ||A||_2 ≈ 1, so ||A^T A||_2 ≈ 1.

**The condition is technically satisfied but with almost zero margin!**
- You need τ/γ > 1
- You have τ/γ = 1.0101
- Margin: only 1% above the boundary!

### Corrected Parameters
Need more margin for robust convergence:
```matlab
gamma = 10
tau   = 1.5*gamma  # tau/gamma = 1.5 > 1 (50% margin)
```

This gives τ/γ = 1.5 > 1 ✓ with comfortable margin

## Why This Causes Your Symptoms

1. **Convergence Stalling at 1e-7**:
   - Step sizes too large → algorithm oscillates instead of converging
   - Gets stuck in a limit cycle rather than finding the true minimum

2. **Wrong Gaussian Translation**:
   - Not reaching the true optimum → incorrect transport pattern
   - The kinetic energy is not being properly minimized
   - Continuity constraint is satisfied (machine zero) but energy minimization fails

## Other Findings

### ✓ Projection is CORRECT
- You verified machine-zero divergence
- The signs in the projection (+ not -) are correct for your formulation

### ✓ Operators are Correct
- Staggered grid structure properly implemented
- Dimensions match correctly:
  - Primal (rho, mx): (63, 127) at cell centers
  - Dual (rho_tilde, bx): (62, 127) and (63, 126) on faces

### ✓ Proximal Operator is Correct
- Cubic solve for kinetic energy minimization looks correct
- Truncation of negative densities is reasonable

## Recommendation

**Try the corrected step sizes:**
```matlab
gamma = 1.0;
tau   = 0.9;
```

If this still doesn't converge well, you can try:
- Smaller steps: `gamma = 0.5, tau = 0.5`
- Adaptive step sizes
- Check the actual operator norm ||K|| numerically and tune accordingly

## Testing

To verify the fix:
1. Run with corrected parameters
2. Check if ||u_{k+1} - u_k|| decreases below 1e-10
3. Verify Gaussian translates linearly from mean=1/3 to mean=2/3
4. Plot the velocity field - should be approximately constant in space
