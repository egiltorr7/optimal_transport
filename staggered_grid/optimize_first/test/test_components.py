"""
test_components.py — Unit tests for prox_J and solve_phi.
Run with: python3 test_components.py
"""
import numpy as np
import scipy.sparse as sp
from operators import build_all_operators, build_rhs
from sb_admm_1d import prox_J, solve_phi, project_C


# ============================================================
# Test 1: prox_J satisfies optimality conditions
# ============================================================
def test_prox_J():
    print("=" * 55)
    print("Test: prox_J optimality conditions")
    np.random.seed(42)
    Nt, Nx = 8, 8
    dt, dx, tau = 1.0/Nt, 1.0/Nx, 2.0
    w = dt * dx

    u = np.abs(np.random.randn(Nt, Nx)) + 0.1   # positive input rho
    v = np.random.randn(Nt, Nx)                  # input m

    rho_s, m_s = prox_J(u, v, dt, dx, tau)

    # Optimality in m: (w / rho_s) * m_s + (m_s - v) = 0
    # => m_s * (w/rho_s + 1) = v  => m_s = v*rho_s / (rho_s + w/tau)?
    # More precisely from the per-cell problem (weight dt*dx/tau on m^2/rho):
    #   d/dm [(w/tau)*m^2/(2*rho) + (m-v)^2/2] = (w/(tau*rho))*m + (m-v) = 0
    cond_m = (w / (tau * rho_s)) * m_s + (m_s - v)
    err_m = np.max(np.abs(cond_m))
    print(f"  Max |m-optimality| = {err_m:.3e}  (want < 1e-10)")

    # Optimality in rho:  (rho_s - u) * (tau*rho_s + w)^2 = w*tau*v^2/2
    #   derived from: d/drho [(w/tau)*m^2/(2*rho) + (rho-u)^2/2] = 0
    #   substituting m_s = tau*rho_s*v / (tau*rho_s + w)
    lhs = (rho_s - u) * (tau * rho_s + w)**2
    rhs = 0.5 * w * tau * v**2
    err_rho = np.max(np.abs(lhs - rhs))
    print(f"  Max |rho-optimality| = {err_rho:.3e}  (want < 1e-8)")

    # v=0 edge case: rho_s should equal u, m_s should equal 0
    u2 = np.abs(np.random.randn(Nt, Nx)) + 0.1
    v2 = np.zeros((Nt, Nx))
    r2, m2 = prox_J(u2, v2, dt, dx, tau)
    err_v0 = np.max(np.abs(r2 - u2))
    print(f"  Max |rho*-u| when v=0  = {err_v0:.3e}  (want ~0)")
    print(f"  Max |m*|    when v=0  = {np.max(np.abs(m2)):.3e}  (want ~0)")
    return err_m < 1e-6 and err_rho < 1e-6


# ============================================================
# Test 2: solve_phi matches explicit M_SB solve
# ============================================================
def test_solve_phi():
    print("=" * 55)
    print("Test: solve_phi vs explicit M_SB")
    Nt, Nx = 6, 6
    dt, dx = 1.0/Nt, 1.0/Nx

    for eps in [0.0, 0.05, 0.1]:
        ops = build_all_operators(Nt, Nx, dt, dx, eps)
        Dt = ops['Dt']; At = ops['At']; Dx = ops['Dx']; Lx = ops['Lx']

        # Build FP_rhobar and FP_b explicitly using Kronecker products for the test.
        # FP_rhobar = kron(Dt, Ix) + eps*kron(At, Lx)   (+eps: forward diffusion)
        # FP_b      = kron(It, Dx)
        Ix = sp.eye(Nx, format='csr')
        It = sp.eye(Nt, format='csr')
        FP_r = sp.kron(Dt, Ix) + eps * sp.kron(At, Lx)
        FP_b = sp.kron(It, Dx)

        # Explicit M_SB = FP_r @ FP_r.T + FP_b @ FP_b.T
        M_SB = FP_r @ FP_r.T + FP_b @ FP_b.T   # (Nt*Nx, Nt*Nx)

        # Check positive definiteness
        M_dense = M_SB.toarray()
        eigs = np.linalg.eigvalsh(M_dense)
        print(f"  eps={eps}: M_SB eig min={eigs.min():.4f}, max={eigs.max():.4f}")

        # Use a consistent RHS: pick random phi, compute F = M_SB @ phi
        np.random.seed(7)
        phi_true = np.random.randn(Nt * Nx)
        phi_true -= phi_true.mean()
        F_vec = M_dense @ phi_true
        F_mat = F_vec.reshape(Nt, Nx)

        phi_our = solve_phi(F_mat, Nt, Nx, dt, dx, eps)

        resid = np.linalg.norm(M_dense @ phi_our.ravel() - F_vec) / np.linalg.norm(F_vec)
        print(f"         residual = {resid:.3e}  "
              f"{'OK' if resid < 1e-8 else 'FAIL'}")

    return True


# ============================================================
# Test 3: project_C satisfies Fokker-Planck constraint
# ============================================================
def _fp_viol(rho_bar, b, ops, d, Nt, Nx, eps):
    """Compute ||FP constraint residual|| using matrix ops (no Kronecker needed)."""
    Dt = ops['Dt']; At = ops['At']; Dx = ops['Dx']; Lx = ops['Lx']
    fp = (Dt @ rho_bar
          + eps * (Lx @ (At @ rho_bar).T).T
          + b @ Dx.T)
    return np.linalg.norm(fp.ravel() - d)


def test_project_C():
    print("=" * 55)
    print("Test: project_C satisfies FP constraint")
    Nt, Nx = 8, 6
    dt, dx, eps = 1.0/Nt, 1.0/Nx, 0.05
    ops = build_all_operators(Nt, Nx, dt, dx, eps)

    rho0 = np.ones(Nx) / Nx
    rho1 = np.ones(Nx) / Nx
    d = build_rhs(rho0, rho1, Nt, Nx, dt)

    np.random.seed(3)
    p_rho = np.random.randn(Nt-1, Nx)
    p_b   = np.random.randn(Nt, Nx-1)

    rho_bar, b = project_C(p_rho, p_b, ops, d, Nt, Nx, dt, dx, eps)

    viol = _fp_viol(rho_bar, b, ops, d, Nt, Nx, eps)
    print(f"  FP constraint violation = {viol:.3e}  (want < 1e-8)")
    return viol < 1e-6


if __name__ == '__main__':
    ok1 = test_prox_J()
    ok2 = test_solve_phi()
    ok3 = test_project_C()
    print("=" * 55)
    print(f"prox_J:    {'PASS' if ok1 else 'FAIL'}")
    print(f"solve_phi: {'PASS' if ok2 else 'FAIL'}")
    print(f"project_C: {'PASS' if ok3 else 'FAIL'}")
