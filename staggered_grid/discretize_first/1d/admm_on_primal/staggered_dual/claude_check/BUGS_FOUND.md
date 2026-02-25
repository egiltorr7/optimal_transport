# Bugs Found in 1D Optimal Transport Implementation

## Critical Bug #1: Wrong Signs in Projection (Line 243-244)

**Location:** `ot1d_admm.m`, lines 243-244 in `proj_div_free`

**Current Code:**
```matlab
rho_out = mu + deriv_t_at_rho(phi_temp);
m_out   = psi + deriv_x_at_m(phi_temp);
```

**Issue:** The projection formula should SUBTRACT the derivatives, not add them.

**Correct Formula:**
For the projection onto the divergence-free constraint ∂ρ/∂t + ∂m/∂x = 0:
- ρ = μ - ∂φ/∂t
- m = ψ - ∂φ/∂x

where -Δφ = 2(∂μ/∂t + ∂ψ/∂x)

**Fix:**
```matlab
rho_out = mu - deriv_t_at_rho(phi_temp);
m_out   = psi - deriv_x_at_m(phi_temp);
```

**Why this causes your symptoms:**
- Wrong signs mean the projection is moving AWAY from the constraint manifold instead of onto it
- This prevents convergence and produces physically incorrect transport
- The continuity equation ∂ρ/∂t + ∂m/∂x = 0 won't be satisfied

---

## Potential Bug #2: Missing Factor of 1/2

**Location:** Same as above

**Issue:** The standard projection formula includes a factor of 1/2:
- ρ = μ - (1/2)∂φ/∂t
- m = ψ - (1/2)∂φ/∂x

And the Laplacian solve should be:
- -Δφ = 2(∂μ/∂t + ∂ψ/∂x)

**Current code** doesn't have this factor. This might be intentional if using a different scaling, but worth checking.

---

## Potential Bug #3: Eigenvalue Formula (Lines 37-38)

**Location:** `ot1d_admm.m`, lines 37-38

**Current Code:**
```matlab
lambda_x = (2*ones(1,nx) - 2*cos(pi*dx.*(0:nxm)))/dx/dx;
```

**Issue:** For DCT with Neumann BC, the correct eigenvalues depend on the grid type:
- For cell centers: λ_k = (2/dx²)(1 - cos(πk/n)) where k = 0,...,n-1
- Current formula uses: λ_k = (2/dx²)(1 - cos(πk·dx))

With dx = 1/nx, we have πk·dx = πk/nx, so the formulas match! This is actually correct.

**Status:** NOT a bug (formula is correct)

---

## Potential Bug #4: Misnamed Operator

**Location:** `ot1d_admm.m`, line 100-110

**Current Code:** Function named `deriv_t_at_phi` but it computes:
```matlab
c = [1 1 zeros(1,ntm-1)];  % averaging pattern
r = [1 zeros(1,ntm-1)];
It = 0.5*toeplitz(c,r);
```

**Issue:** This is an INTERPOLATION operator (averages), not a derivative operator!

**Correct Usage:** Looking at line 239, it's used as:
```matlab
dmu_dt = deriv_t_at_phi(mu,rho0,rho1);
```

But it should compute ∂μ/∂t. Let me check if this is intentional or a bug...

Actually, looking more carefully at line 144-154, there IS a `deriv_t_at_phi` that computes differences:
```matlab
c = [1 -1 zeros(1,ntm-1)];  % difference pattern
```

So there are TWO functions with the same name? Need to check this.

**Status:** Need verification - possible naming confusion or duplicate functions

---

## Summary

**Must Fix:**
1. Sign error in projection (Bug #1) - this is definitely wrong

**Should Investigate:**
2. Missing factor of 1/2 (Bug #2) - check if this matches your formulation
3. Operator naming (Bug #4) - check for duplicate function definitions

**Not a Bug:**
- Eigenvalue formula is correct
