"""Consistency checks for projection_expsemi_theta.py (the ETD-theta scheme).

Run this after touching projection_expsemi_theta.py.  Same philosophy as
validate_nu.py: the closed-form tridiagonal entries and the Theta/Theta^*
pair are hand-derived, so nothing here trusts that algebra -- every piece is
checked against an independently built dense reference, and the decisive
check (#5) would fail if ANY of the derivation were wrong, since a projection
built on a T that is not the true Gram R R^* cannot land on the constraint set.

  1. theta/omega weights are in range and match a high-precision quadrature
     of the ETD weight's centroid, across 15 decades of alpha.
  2. _apply_theta / _apply_theta_adj are exact Euclidean adjoints, and
     _apply_theta agrees with a densely built Theta matrix.
  3. The closed-form (main, sub) entries match a dense
     R_rho R_rho^* + phi^2*lam*Theta Theta^* built per mode, and every
     per-mode system is SPD (Cholesky succeeds) -- this is what rules out the
     final-row extrapolation making Theta singular.
  4. omega == 0 degeneracy: at vareps = 0 ETD-theta reproduces
     projection_expsemi.py's ETD1 projection to machine precision.
  5. FEASIBILITY: the ETD-theta FP residual of the projected point is ~0,
     computed with a dense Theta rather than the module's own helper.
  6. Idempotence: projecting an already-projected point changes nothing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.fft import dct, idct

from optimal_transport.grid import setup_problem
from optimal_transport.operators import build_operators
from optimal_transport.problems import prob_gaussian
from optimal_transport.projection_expsemi import (
    precomp_expsemi_proj,
    proj_fokker_planck_expsemi,
)
from optimal_transport.projection_expsemi_theta import (
    _apply_theta,
    _apply_theta_adj,
    omega_weights,
    precomp_expsemi_theta_proj,
    proj_fokker_planck_expsemi_theta,
    theta_centroid,
)
from optimal_transport.state import State

NT, NX = 32, 24
RNG = np.random.default_rng(0)
EPS_SWEEP = [0.0, 1e-8, 1e-3, 0.1, 1.0, 10.0, 1000.0]
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def mkproblem(nt=NT, nx=NX):
    p = setup_problem(prob_gaussian(), nt=nt, nx=nx)
    p.ops = build_operators(p)
    return p


def dense_theta(nt, w):
    """Theta for a single mode, built row by row straight from the definition."""
    T = np.zeros((nt, nt))
    for k in range(nt - 1):
        T[k, k] = 1 - w
        T[k, k + 1] = w
    T[nt - 1, nt - 1] = 1.0        # constant extrapolation -> lone midpoint sample
    return T


# ---------------------------------------------------------------- 1
print("\n1. theta/omega weights")
a = np.array([0.0, 1e-12, 1e-6, 1e-4, 1e-2, 0.5, 1.0, 5.0, 50.0, 1e4, 1e8])
# the GL reference underflows for very large alpha (exp(-a(1-u)) == 0 at every
# node but u==1), so compare only where it is computable
a_ref = a[a <= 500.0]
th, om = theta_centroid(a), omega_weights(a)
check("theta in [1/2, 1), omega in [0, 1/2)",
      bool(np.all((th >= 0.5) & (th < 1.0) & (om >= 0.0) & (om < 0.5))))
check("theta monotone increasing in alpha", bool(np.all(np.diff(th) > -1e-15)))
# reference: the DEFINITION -- centroid of the ETD weight exp(-a(1-u)) over
# u in [0,1] -- by adaptive quadrature.  (Fixed-node Gauss-Legendre is not
# accurate enough here: the weight is sharply peaked at u=1 for large a.)
from scipy.integrate import quad

a_ref = a[(a > 0) & (a <= 1e4)]
ref = np.array([
    quad(lambda u, x=x: u * np.exp(-x * (1 - u)), 0, 1, epsabs=1e-15, epsrel=1e-13)[0]
    / quad(lambda u, x=x: np.exp(-x * (1 - u)), 0, 1, epsabs=1e-15, epsrel=1e-13)[0]
    for x in a_ref
])
# Measure the error on OMEGA = theta - 1/2, not on theta: omega is what the
# scheme multiplies by, and theta ~ 1/2 masks a relative error in omega by
# three orders of magnitude.
err = np.max(np.abs(omega_weights(a_ref) - (ref - 0.5)) / (ref - 0.5))
check("omega matches adaptive-quadrature centroid", err < 1e-9,
      f"max rel err on omega {err:.3e}")
# large alpha: the weight collapses onto u=1, so theta -> 1 - 1/a
a_big = np.array([1e3, 1e6, 1e10])
err_asym = np.max(np.abs(theta_centroid(a_big) - (1 - 1 / a_big)) * a_big**2)
check("large-alpha asymptote theta ~ 1 - 1/a", err_asym < 1e-6,
      f"max |theta-(1-1/a)|*a^2 = {err_asym:.3e}")
check("alpha=0 gives exactly the midpoint", th[0] == 0.5)

# ---------------------------------------------------------------- 2
print("\n2. Theta / Theta^* adjointness")
worst_adj, worst_dense = 0.0, 0.0
for w_scalar in [0.0, 1e-6, 0.05, 0.2, 0.4999]:
    w = np.full(NX, w_scalar)
    g = RNG.standard_normal((NT, NX))
    v = RNG.standard_normal((NT, NX))
    lhs = np.sum(_apply_theta(g, w) * v)
    rhs = np.sum(g * _apply_theta_adj(v, w))
    worst_adj = max(worst_adj, abs(lhs - rhs) / max(abs(lhs), 1e-300))
    Td = dense_theta(NT, w_scalar)
    worst_dense = max(worst_dense, np.max(np.abs(_apply_theta(g, w) - Td @ g)))
check("<Theta g, v> == <g, Theta^* v>", worst_adj < 1e-13, f"max rel err {worst_adj:.3e}")
check("_apply_theta == dense Theta @ g", worst_dense < 1e-13, f"max abs err {worst_dense:.3e}")

# ---------------------------------------------------------------- 3
print("\n3. closed-form (main, sub) vs dense Gram, and SPD-ness")
worst_sys, min_eig_ratio, spd_ok = 0.0, np.inf, True
for vareps in EPS_SWEEP:
    p = mkproblem()
    ep = precomp_expsemi_theta_proj(p, vareps)
    nt, dt = p.nt, p.dt
    # rebuild main/sub from the dataclass' own inputs (main_mod is swept, so
    # reconstruct the pre-sweep diagonal the same way precomp does)
    c, phi, w = ep.c_vals[1:], ep.phi_vals[1:], ep.omega[1:]
    lx = p.lambda_x[1:]
    for j in range(lx.shape[0]):
        Arho = np.zeros((nt, nt - 1))
        for k in range(nt):
            if k <= nt - 2:
                Arho[k, k] = 1 / dt
            if k >= 1:
                Arho[k, k - 1] = -c[j] / dt
        Td = dense_theta(nt, w[j])
        T_ref = Arho @ Arho.T + phi[j] ** 2 * lx[j] * (Td @ Td.T)

        main_j = np.diag(T_ref).copy()
        sub_j = np.diag(T_ref, 1).copy()
        # module's closed form
        main_c = np.full(nt, (1 + c[j] ** 2) / dt**2)
        main_c[0] = 1 / dt**2
        main_c[-1] = c[j] ** 2 / dt**2
        gram = phi[j] ** 2 * lx[j]
        main_c += gram * ((1 - w[j]) ** 2 + w[j] ** 2)
        main_c[-1] = c[j] ** 2 / dt**2 + gram
        sub_c = np.full(nt - 1, -c[j] / dt**2) + gram * w[j] * (1 - w[j])
        sub_c[-1] = -c[j] / dt**2 + gram * w[j]

        scale = max(np.max(np.abs(main_j)), 1e-300)
        worst_sys = max(worst_sys,
                        np.max(np.abs(main_j - main_c)) / scale,
                        np.max(np.abs(sub_j - sub_c)) / scale)
        # T_ref symmetric tridiagonal + SPD?
        if np.max(np.abs(T_ref - np.diag(np.diag(T_ref))
                         - np.diag(np.diag(T_ref, 1), 1)
                         - np.diag(np.diag(T_ref, -1), -1))) > 1e-9 * scale:
            spd_ok = False
        ev = np.linalg.eigvalsh(T_ref)
        if ev.min() <= 0:
            spd_ok = False
        min_eig_ratio = min(min_eig_ratio, ev.min() / ev.max())
check("closed-form main/sub == dense Gram", worst_sys < 1e-12, f"max rel err {worst_sys:.3e}")
check("every per-mode T is tridiagonal and SPD", spd_ok,
      f"worst lam_min/lam_max over all modes & eps = {min_eig_ratio:.3e}")

# ---------------------------------------------------------------- 4
print("\n4. omega == 0 degeneracy (vareps = 0 => ETD-theta == ETD1)")
p = mkproblem()
x = State(RNG.standard_normal((NT - 1, NX)), RNG.standard_normal((NT, NX - 1)))
e1 = proj_fokker_planck_expsemi(x, p, 0.0, precomp_expsemi_proj(p, 0.0))
et = proj_fokker_planck_expsemi_theta(x, p, 0.0, precomp_expsemi_theta_proj(p, 0.0))
d = max(np.max(np.abs(e1.rho - et.rho)), np.max(np.abs(e1.mx - et.mx)))
sc = max(np.max(np.abs(e1.rho)), np.max(np.abs(e1.mx)))
check("matches ETD1 at vareps=0", d / sc < 1e-13, f"max rel diff {d/sc:.3e}")

# ---------------------------------------------------------------- 5
print("\n5. feasibility: ETD-theta residual after projection (dense Theta)")
worst_feas = 0.0
for vareps in EPS_SWEEP:
    p = mkproblem()
    ep = precomp_expsemi_theta_proj(p, vareps)
    x = State(RNG.standard_normal((NT - 1, NX)), RNG.standard_normal((NT, NX - 1)))
    xo = proj_fokker_planck_expsemi_theta(x, p, vareps, ep)

    def residual_dense(st):
        mu, psi = st.rho, st.mx
        z = np.zeros(p.nt)
        mu_prev = np.concatenate([p.rho0[None, :], mu], axis=0)
        mu_curr = np.concatenate([mu, p.rho1[None, :]], axis=0)
        sp = idct(dct(mu_prev, type=2, norm="ortho", axis=1) * ep.c_vals,
                  type=2, norm="ortho", axis=1)
        fh = dct((mu_curr - sp) / p.dt, type=2, norm="ortho", axis=1)
        gh = dct(p.ops.deriv_x_at_phi(psi, z, z), type=2, norm="ortho", axis=1)
        gt = np.empty_like(gh)
        for j in range(p.nx):                       # dense Theta, column by column
            gt[:, j] = dense_theta(p.nt, ep.omega[j]) @ gh[:, j]
        return fh + ep.phi_vals * gt

    r_in = np.linalg.norm(residual_dense(x))
    r_out = np.linalg.norm(residual_dense(xo))
    worst_feas = max(worst_feas, r_out / r_in)
    print(f"        vareps={vareps:<9g} ||R|| before {r_in:.3e} -> after {r_out:.3e}"
          f"   (ratio {r_out/r_in:.2e})")
check("residual driven to ~0 at every vareps", worst_feas < 1e-10,
      f"worst after/before ratio {worst_feas:.3e}")

# ---------------------------------------------------------------- 6
print("\n6. idempotence")
worst_idem = 0.0
for vareps in EPS_SWEEP:
    p = mkproblem()
    ep = precomp_expsemi_theta_proj(p, vareps)
    x = State(RNG.standard_normal((NT - 1, NX)), RNG.standard_normal((NT, NX - 1)))
    x1 = proj_fokker_planck_expsemi_theta(x, p, vareps, ep)
    x2 = proj_fokker_planck_expsemi_theta(x1, p, vareps, ep)
    n = max(np.max(np.abs(x1.rho)), np.max(np.abs(x1.mx)))
    worst_idem = max(worst_idem,
                     max(np.max(np.abs(x2.rho - x1.rho)),
                         np.max(np.abs(x2.mx - x1.mx))) / n)
check("P(P(x)) == P(x)", worst_idem < 1e-10, f"max rel drift {worst_idem:.3e}")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
