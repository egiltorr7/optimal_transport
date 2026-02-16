# Detailed Analysis of ADMM Implementation

## Summary of Findings

After careful analysis of the code, I've identified several potential issues that could explain poor convergence:

---

## Issue #1: **CRITICAL SIGN ERROR** in Projection Operator

**Location:** `ot1d_admm.m`, lines 262-263

### Current Code:
```matlab
function [rho_out, m_out] = proj_div_free(mu, psi)
    dirichlet_bc  = zeros(nt,1);
    dmu_dt  = deriv_t_at_phi(mu,rho0,rho1);
    dpsi_dx = deriv_x_at_phi(psi,dirichlet_bc,dirichlet_bc);

    phi_temp = invert_neg_laplacian(dmu_dt + dpsi_dx);
    rho_out = mu + deriv_t_at_rho(phi_temp);     % ← WRONG SIGN
    m_out   = psi + deriv_x_at_m(phi_temp);       % ← WRONG SIGN
end
```

### Mathematical Derivation:

To project (μ, ψ) onto the constraint ∂ρ/∂t + ∂m/∂x = 0:

1. **Formulate the projection problem:**
   ```
   minimize (1/2)||ρ - μ||² + (1/2)||m - ψ||²
   subject to ∂ρ/∂t + ∂m/∂x = 0
   ```

2. **Lagrangian:**
   ```
   L = (1/2)||ρ - μ||² + (1/2)||m - ψ||² + ∫∫ φ(∂ρ/∂t + ∂m/∂x) dx dt
   ```

3. **First-order conditions:**
   ```
   ∂L/∂ρ: ρ - μ + ∂φ/∂t = 0  ⟹  ρ = μ - ∂φ/∂t
   ∂L/∂m: m - ψ + ∂φ/∂x = 0  ⟹  m = ψ - ∂φ/∂x
   ∂L/∂φ: ∂ρ/∂t + ∂m/∂x = 0
   ```

4. **Substitute into constraint:**
   ```
   ∂(μ - ∂φ/∂t)/∂t + ∂(ψ - ∂φ/∂x)/∂x = 0
   ∂μ/∂t + ∂ψ/∂x = ∂²φ/∂t² + ∂²φ/∂x²
   ```

5. **Poisson equation:**
   ```
   -Δφ = ∂μ/∂t + ∂ψ/∂x
   ```

### Correct Formula:
```matlab
rho_out = mu - deriv_t_at_rho(phi_temp);  % MINUS, not plus
m_out   = psi - deriv_x_at_m(phi_temp);    % MINUS, not plus
```

### Why This Matters:

With the **wrong signs** (+), the "projection" actually moves **away** from the constraint manifold instead of onto it! This would:
- Prevent convergence
- Produce physically incorrect results
- Cause the transport to look wrong

**However:** Your DIAGNOSIS.md claims you verified "machine-zero divergence", which contradicts this. This suggests either:
1. The divergence check was done incorrectly
2. There's a compensating sign error elsewhere
3. There's a different sign convention I'm missing

---

## Issue #2: Potential Problem with Moreau Identity Application

**Location:** `ot1d_admm.m`, lines 66-73

### Current Code:
```matlab
% Lines 66-67: Compute residual
tmp_rho = interp_t_at_phi(rho_tilde,rho0,rho1) - rho - delta_rho./gamma;
tmp_mx  = interp_x_at_phi(bx,dirichlet_bc_space,dirichlet_bc_space) - mx - delta_mx./gamma;

% Lines 70-71: Apply adjoint
tmp_rho = rho_tilde - (gamma/tau)*interp_t_at_rho(tmp_rho);
tmp_mx  = bx - (gamma/tau)*interp_x_at_m(tmp_mx);

% Line 73: Project
[rho_tilde_new, bx_new] = proj_div_free(tmp_rho, tmp_mx);
```

### Moreau's Identity:
For conjugate function f*, the proximal operator is:
```
prox_{τf*}(v) = v - τ · prox_{f/τ}(v/τ)
```

For indicator function g = δ_C (constraint set C):
```
g* = support function of C (not simple indicator)
prox_{τg*}(v) = v - τ · proj_C(v/τ)
```

### Potential Issues:

1. **Scaling in projection:** For affine constraints (not cone constraints), the projection doesn't scale simply:
   ```
   proj_C(v/τ) ≠ proj_C(v)/τ
   ```
   So the correct formula for Moreau's identity with affine constraints is more complex.

2. **The gradient step** (lines 70-71) uses `gamma/tau` as the step size, then projects. But for proper Moreau identity:
   ```
   prox_{τg*}(v) = v - τ · proj_C((v - something)/τ)
   ```
   The argument to proj_C should be **scaled by 1/τ**, but your code doesn't do this scaling before the projection.

### What to Check:

Does the algorithm actually implement Moreau's identity correctly for the specific case of affine constraints? Or is this a primal-dual method that doesn't directly use Moreau's identity?

---

## Issue #3: Boundary Conditions in Adjoint (May Already Be Fixed)

**Status:** According to ADJOINT_BUG_FIX.md, this should be fixed, but verify:

Line 70 uses:
```matlab
tmp_rho = rho_tilde - (gamma/tau)*interp_t_at_rho(tmp_rho);
```

The function `interp_t_at_rho` doesn't take BC arguments, so it implicitly uses zero BC. This is **correct** for the adjoint operator.

✓ This appears to be correctly implemented.

---

## Issue #4: Step Size Ratio

**Status:** According to DIAGNOSIS.md, this was fixed:

```matlab
gamma = 10;
tau = 1.1*gamma;  % tau/gamma = 1.1
```

For convergence, need: τ/γ > ||A^T A||_2

With interpolation operators: ||A||_2 ≈ 1, so ||A^T A||_2 ≈ 1.

Your ratio τ/γ = 1.1 satisfies this, but **barely** (only 10% margin).

### Recommendation:
Try **much larger margin** for robustness:
```matlab
gamma = 1.0;
tau = 2.0;  % tau/gamma = 2.0 (100% margin)
```

Or even:
```matlab
gamma = 1.0;
tau = 5.0;  % tau/gamma = 5.0
```

Larger τ/γ ratios often give better convergence, especially with staggered grids.

---

## Proximal Operator Verification

**Location:** Lines 82-85

### Mathematical Check:

For f(ρ, m) = m²/(2ρ), the proximal operator solves:
```
minimize m²/(2ρ) + (1/(2γ))[(ρ - tmp_ρ)² + (m - tmp_m)²]
```

**FOC for m:**
```
m/ρ + (1/γ)(m - tmp_m) = 0
m(1/ρ + 1/γ) = tmp_m/γ
m = ρ·tmp_m/(γ + ρ)
```

**FOC for ρ (cubic equation):**
```
-m²/(2ρ²) + (1/γ)(ρ - tmp_ρ) = 0
```

Substituting m and simplifying with σ = 1/γ leads to:
```
ρ³ + (2σ - tmp_ρ)ρ² + (σ² - 2σ·tmp_ρ)ρ - σ(σ·tmp_ρ + tmp_m²/2) = 0
```

### Code Check:
```matlab
sigma = 1/gamma;
rho = solve_cubic(1, 2*sigma-tmp_rho, sigma^2-2*sigma*tmp_rho, ...
                  -sigma*(sigma*tmp_rho + 0.5.*tmp_mx.^2));
mx = (rho.*tmp_mx)./(sigma+rho);
```

✓ **This matches perfectly!** The proximal operator is correctly implemented.

---

## Recommendations for Debugging

### 1. **FIRST: Fix the projection sign error**
Change lines 262-263 in `proj_div_free`:
```matlab
rho_out = mu - deriv_t_at_rho(phi_temp);  % Change + to -
m_out   = psi - deriv_x_at_m(phi_temp);    % Change + to -
```

### 2. **Verify divergence after projection**
Add this diagnostic at the end of `proj_div_free`:
```matlab
% Verify divergence is actually zero
div_check = deriv_t_at_phi(rho_out, rho0, rho1) + ...
            deriv_x_at_phi(m_out, dirichlet_bc, dirichlet_bc);
fprintf('  Divergence after projection: %.2e\n', max(abs(div_check(:))));
```

### 3. **Increase step size margin**
Try more conservative step sizes:
```matlab
gamma = 1.0;
tau = 2.0;  % or even 5.0
```

### 4. **Check if you're solving primal or dual**
Your directory is named "admm_on_dual" but the code structure looks like a **primal-dual method** (like Chambolle-Pock), not pure ADMM on the dual. Make sure the algorithm matches your formulation.

### 5. **Verify Moreau identity application**
For affine constraints, Moreau's identity for conjugates is subtle. Consider:
- Are you computing prox_{g*} correctly?
- Should there be additional scaling in the projection call?

---

## Questions to Investigate

1. **How did you verify "machine-zero divergence"?**
   - If the signs are wrong in projection, divergence should NOT be zero
   - Check your verification code carefully

2. **What does the transport look like?**
   - Does the density move in the right direction?
   - Does it preserve mass?
   - Is the velocity field physical?

3. **What's the actual convergence behavior?**
   - Does residual decrease monotonically?
   - Does it stall at a fixed value?
   - Does it oscillate?

4. **Is this truly ADMM on the dual, or primal-dual?**
   - The algorithm structure suggests primal-dual splitting
   - This affects how Moreau's identity should be applied
