# Critical Bug Fix: Affine vs Linear Adjoint

## The Issue

When computing the adjoint of an **affine** operator (linear + boundary conditions), you were including the boundary condition terms in the adjoint operation, which is incorrect.

### Affine Operators

An affine operator has the form:
```
y = A*x + b
```
where `b` contains boundary condition contributions.

The **adjoint** of this operator acting on `y` should be:
```
A^T * y   (NOT  A^T * (y + b))
```

Or equivalently, when computing the adjoint, use **zero boundary conditions**.

## The Bug in the Code

### Line 61 (WRONG):
```matlab
tmp_rho = rho - (gamma/tau)*interp_t_at_phi(tmp_rho,rho0,rho1);
```
This uses `rho0` and `rho1` (non-zero BC) when computing the adjoint!

### Line 62 (CORRECT):
```matlab
tmp_mx = mx - (gamma/tau)*interp_x_at_phi(tmp_mx,dirichlet_bc_space,dirichlet_bc_space);
```
This correctly uses `dirichlet_bc_space = zeros(nt,1)` (zero BC)!

## Why This Matters

### In the Primal-Dual Updates (Lines 61-62):
- You're computing `A^T * (residual)` as part of the gradient step
- This should use the **pure linear adjoint** (zero BC)
- Using non-zero BC corrupts the gradient direction
- This causes:
  - Wrong optimization direction
  - Inability to reach true minimum
  - Convergence to incorrect solution

### In the Projection (Lines 239-240):
- You're solving the PDE: `-Δφ = ∂μ/∂t + ∂ψ/∂x`
- WITH boundary conditions: `μ(0,·) = rho0`, `μ(1,·) = rho1`
- Here you SHOULD use the actual boundary conditions
- This is correct in the projection!

## The Fix

Changed line 61 to:
```matlab
tmp_rho = rho - (gamma/tau)*interp_t_at_phi(tmp_rho,zeros(1,nx),zeros(1,nx));
```

Now both spatial and temporal adjoints correctly use zero BC.

## Expected Impact

This should:
1. **Fix the Gaussian translation** - optimization will now find correct transport
2. **Improve convergence** - gradient direction is now correct
3. **Reach machine precision** - can now converge to true minimum

## Mathematical Background

For an operator `L : U → V` defined by `L(u) = A*u + b` where `b` encodes boundary conditions:

The adjoint `L^* : V → U` is given by:
```
L^*(v) = A^T * v
```

Note that the adjoint does NOT include the boundary term `b`. This is because:
```
⟨L(u), v⟩_V = ⟨A*u + b, v⟩ = ⟨A*u, v⟩ + ⟨b, v⟩ = ⟨u, A^T*v⟩ + ⟨b, v⟩
```

For `L^*(v)` to satisfy `⟨L(u), v⟩ = ⟨u, L^*(v)⟩`, we need:
```
L^*(v) = A^T * v
```

The boundary term `⟨b, v⟩` must be handled separately as a constant offset, not part of the adjoint operator itself.
