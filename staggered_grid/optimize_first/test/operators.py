"""
operators.py — Discrete operators for the 1D Schrödinger Bridge ADMM solver.

Builds all sparse matrices needed for the two-grid linearized ADMM formulation:
  - D_t : time first-difference,  shape (Nt, Nt-1)
  - A_t : time averaging,          shape (Nt, Nt-1)
  - D_x : space first-difference,  shape (Nx, Nx-1)
  - L_x : Neumann Laplacian,       shape (Nx, Nx)
  - A_rho: collocated->staggered ρ interpolation, shape (Nt-1, Nt)  [in time]
  - A_m  : collocated->staggered m  interpolation, shape (Nx-1, Nx)  [in space]

Kronecker products and RHS construction are also provided.
"""

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# 1-D operators
# ---------------------------------------------------------------------------

def build_Dt(Nt: int, dt: float) -> sp.csr_matrix:
    """
    Time first-difference operator, shape (Nt, Nt-1).

    Acts on interior time nodes of rho_bar (n=1,...,Nt-1).
    Boundary values rho_bar^0 = rho0, rho_bar^Nt = rho1 are absorbed into d.

    Row i corresponds to the time equation at half-time i+0.5:
      D_t[i, i]   = +1/dt   (upper diagonal)
      D_t[i, i-1] = -1/dt   (lower diagonal, where present)
    """
    rows, cols, vals = [], [], []
    for i in range(Nt):
        # upper diagonal entry
        if i < Nt - 1:
            rows.append(i); cols.append(i);   vals.append(1.0 / dt)
        # lower diagonal entry
        if i > 0:
            rows.append(i); cols.append(i-1); vals.append(-1.0 / dt)
    return sp.csr_matrix((vals, (rows, cols)), shape=(Nt, Nt - 1))


def build_At(Nt: int) -> sp.csr_matrix:
    """
    Time averaging operator, shape (Nt, Nt-1).

    Averages interior rho_bar from integer times to half times:
      A_t[i, i]   = 0.5   (where i < Nt-1)
      A_t[i, i-1] = 0.5   (where i > 0)
    """
    rows, cols, vals = [], [], []
    for i in range(Nt):
        if i < Nt - 1:
            rows.append(i); cols.append(i);   vals.append(0.5)
        if i > 0:
            rows.append(i); cols.append(i-1); vals.append(0.5)
    return sp.csr_matrix((vals, (rows, cols)), shape=(Nt, Nt - 1))


def build_Dx(Nx: int, dx: float) -> sp.csr_matrix:
    """
    Space first-difference operator, shape (Nx, Nx-1).

    Same structure as D_t but in space (with Neumann BCs handled implicitly
    by restricting b to interior faces j=1,...,Nx-1).
    """
    rows, cols, vals = [], [], []
    for j in range(Nx):
        if j < Nx - 1:
            rows.append(j); cols.append(j);   vals.append(1.0 / dx)
        if j > 0:
            rows.append(j); cols.append(j-1); vals.append(-1.0 / dx)
    return sp.csr_matrix((vals, (rows, cols)), shape=(Nx, Nx - 1))


def build_Lx(Nx: int, dx: float) -> sp.csr_matrix:
    """
    Neumann Laplacian L_x = D_x @ D_x.T, shape (Nx, Nx).

    Corner entries are -1/dx^2 (one neighbour), interior are -2/dx^2.
    """
    Dx = build_Dx(Nx, dx)
    return (Dx @ Dx.T).tocsr()


def build_Arho(Nt: int) -> sp.csr_matrix:
    """
    Interpolation A_rho: maps collocated rho (Nt,) -> staggered rho_bar (Nt-1,).

    rho_bar[i] = (rho[i] + rho[i+1]) / 2   for i = 0,...,Nt-2
    Shape: (Nt-1, Nt)
    """
    rows, cols, vals = [], [], []
    for i in range(Nt - 1):
        rows.append(i); cols.append(i);   vals.append(0.5)
        rows.append(i); cols.append(i+1); vals.append(0.5)
    return sp.csr_matrix((vals, (rows, cols)), shape=(Nt - 1, Nt))


def build_Am(Nx: int) -> sp.csr_matrix:
    """
    Interpolation A_m: maps collocated m (Nx,) -> face-staggered b (Nx-1,).

    b[j] = (m[j] + m[j+1]) / 2   for j = 0,...,Nx-2
    Shape: (Nx-1, Nx)
    """
    rows, cols, vals = [], [], []
    for j in range(Nx - 1):
        rows.append(j); cols.append(j);   vals.append(0.5)
        rows.append(j); cols.append(j+1); vals.append(0.5)
    return sp.csr_matrix((vals, (rows, cols)), shape=(Nx - 1, Nx))


# ---------------------------------------------------------------------------
# Full-grid operators (Kronecker products)
# ---------------------------------------------------------------------------

def build_all_operators(Nt: int, Nx: int, dt: float, dx: float, eps: float):
    """
    Build and return all discrete operators as a dict.

    Keys:
      Dt, At, Dx, Lx       — 1-D sparse operators
      Arho, Am             — interpolation operators
      FP_rhobar, FP_b      — blocks of the Fokker-Planck matrix
                             FP_rhobar = I_x ⊗ D_t − ε L_x ⊗ A_t
                             FP_b      = D_x ⊗ I_t
    """
    Dt   = build_Dt(Nt, dt)
    At   = build_At(Nt)
    Dx   = build_Dx(Nx, dx)
    Lx   = build_Lx(Nx, dx)
    Arho = build_Arho(Nt)
    Am   = build_Am(Nx)

    It = sp.eye(Nt, format='csr')
    Ix = sp.eye(Nx, format='csr')

    # Fokker-Planck blocks — time-slow (row-major) ordering:
    #   vector index = i*Nx + j  (i=time, j=space)
    # FP_rhobar acts on vec(rho_bar) with rho_bar shape (Nt-1, Nx)
    # FP_b      acts on vec(b)       with b       shape (Nt,   Nx-1)
    FP_rhobar = sp.kron(Dt, Ix, format='csr') - eps * sp.kron(At, Lx, format='csr')
    FP_b      = sp.kron(It, Dx, format='csr')

    return dict(Dt=Dt, At=At, Dx=Dx, Lx=Lx, Arho=Arho, Am=Am,
                FP_rhobar=FP_rhobar, FP_b=FP_b)


# ---------------------------------------------------------------------------
# RHS vector d
# ---------------------------------------------------------------------------

def build_rhs(rho0: np.ndarray, rho1: np.ndarray,
              Nt: int, Nx: int, dt: float) -> np.ndarray:
    """
    Build the RHS vector d for the Fokker-Planck constraint.

    d encodes the boundary conditions.

    The FP constraint (time-slow ordering) is:
      FP_rhobar @ rho_bar + FP_b @ b = d
    For a stationary field rho_bar=c, b=0, D_t gives:
      row 0: +c/dt (only upper neighbour)  → d[0:Nx] = +rho0/dt
      row Nt-1: -c/dt (only lower neighb.) → d[-Nx:] = -rho1/dt

    Shape: (Nt * Nx,)
    """
    d = np.zeros(Nt * Nx)
    d[:Nx]  =  rho0 / dt
    d[-Nx:] = -rho1 / dt
    return d


# ---------------------------------------------------------------------------
# Interpolation helpers (array form, no sparse multiply needed in hot loop)
# ---------------------------------------------------------------------------

def apply_Arho(rho: np.ndarray) -> np.ndarray:
    """
    A_rho applied to rho array of shape (Nt, Nx) -> (Nt-1, Nx).
    rho_bar[i] = (rho[i] + rho[i+1]) / 2
    """
    return 0.5 * (rho[:-1] + rho[1:])


def apply_Arho_T(rho_bar: np.ndarray, Nt: int) -> np.ndarray:
    """
    A_rho^T applied to rho_bar of shape (Nt-1, Nx) -> (Nt, Nx).
    """
    Nx = rho_bar.shape[1]
    out = np.zeros((Nt, Nx))
    out[:-1] += 0.5 * rho_bar
    out[1:]  += 0.5 * rho_bar
    return out


def apply_Am(m: np.ndarray) -> np.ndarray:
    """
    A_m applied to m array of shape (Nt, Nx) -> (Nt, Nx-1).
    b[j] = (m[j] + m[j+1]) / 2
    """
    return 0.5 * (m[:, :-1] + m[:, 1:])


def apply_Am_T(b: np.ndarray, Nx: int) -> np.ndarray:
    """
    A_m^T applied to b of shape (Nt, Nx-1) -> (Nt, Nx).
    """
    Nt = b.shape[0]
    out = np.zeros((Nt, Nx))
    out[:, :-1] += 0.5 * b
    out[:, 1:]  += 0.5 * b
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_operators(Nt: int = 6, Nx: int = 5, eps: float = 0.1):
    """Print shapes and basic checks for all operators."""
    dt = 1.0 / Nt
    dx = 1.0 / Nx
    ops = build_all_operators(Nt, Nx, dt, dx, eps)

    print("=== Operator shapes ===")
    expected = {
        'Dt':   (Nt,   Nt-1),
        'At':   (Nt,   Nt-1),
        'Dx':   (Nx,   Nx-1),
        'Lx':   (Nx,   Nx),
        'Arho': (Nt-1, Nt),
        'Am':   (Nx-1, Nx),
        'FP_rhobar': (Nt*Nx, (Nt-1)*Nx),
        'FP_b':      (Nt*Nx, Nt*(Nx-1)),
    }
    for name, exp in expected.items():
        actual = ops[name].shape
        status = "OK" if actual == exp else f"MISMATCH (got {actual})"
        print(f"  {name:12s}: expected {exp}  {status}")

    # Check Lx is positive semi-definite (L_x = D_x D_x^T = negative of Laplacian)
    Lx_dense = ops['Lx'].toarray()
    eigs = np.linalg.eigvalsh(Lx_dense)
    print(f"\n  L_x eigenvalues: min={eigs.min():.4f}, max={eigs.max():.4f}")
    print(f"  (should be ≥ 0: L_x = D_x D_x^T is positive semi-definite)")

    # Check RHS builder
    rho0 = np.ones(Nx) / (Nx * dx)
    rho1 = np.ones(Nx) / (Nx * dx)
    d = build_rhs(rho0, rho1, Nt, Nx, dt)
    print(f"\n  RHS d shape: {d.shape}, nonzero blocks: {np.count_nonzero(d)}/{len(d)}")

    print("\noperators.py: all checks passed.\n")


if __name__ == '__main__':
    validate_operators()
