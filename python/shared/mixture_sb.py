"""Closed-form Schrodinger bridge between two Gaussian *mixtures*, built the way
Gushchin et al. (NeurIPS 2023, arXiv:2306.10161) build their benchmark pairs,
but with both Schrodinger factors chosen as Gaussian mixtures so that rho_0,
rho_1, rho_t and the momentum m are all closed form.  Dimension-agnostic
(D = 1, 2, 3, ...), plain numpy, no JAX needed.

Solver convention (python/2d, python/3d):

    d_t rho + div m = eps * Lap rho,   m = rho v,   reference SDE  dX = sqrt(2 eps) dW.

So sigma^2 = 2 eps, and the paper's regularization eps_paper equals 2 eps.

Construction
------------
The optimal plan of the entropic problem with cost |x-y|^2/2 and reg sigma^2 is

    pi*(x, y) = phihat_0(x) * exp(-|x-y|^2 / (2 sigma^2)) * phi_1(y)       (Gibbs form)

for SOME positive functions phihat_0, phi_1 (the Schrodinger factors;
phihat_0 = exp(f*/sigma^2), phi_1 = exp(g*/sigma^2) in potential language).
Conversely, ANY positive pair plugged in gives the optimal plan between its
own two marginals.  That is the whole trick: choose the factors, read off the
marginals.  The paper fixes rho_0 and the potential g* = f*_paper and gets
pi*(y|x) as a Gaussian mixture; rho_1 then needs an integral.  Choosing both
factors as Gaussian mixtures removes the integral:

    phihat_0(x) = sum_i u_i N(x | a_i, P_i),     phi_1(y) = sum_n w_n N(y | b_n, Q_n).

The heat flow with generator eps*Lap adds 2*eps*t to every covariance:

    phi(t, z)    = sum_n w_n N(z | b_n, Q_n + 2 eps (1-t) I)     (backward eq., d_t phi = -eps Lap phi)
    phihat(t, z) = sum_i u_i N(z | a_i, P_i + 2 eps t I)         (forward eq.,  d_t phihat = eps Lap phihat)

and rho_t = phi * phihat / K.  Products of Gaussians are Gaussians, so

    rho_t(z) = sum_{i,n} omega_in N(z | c_in(t), C_in(t)),
    omega_in = u_i w_n N(a_i | b_n, P_i + Q_n + 2 eps I) / K,   K = sum of the numerators,
    C_in(t)  = ( (P_i + 2 eps t I)^-1 + (Q_n + 2 eps (1-t) I)^-1 )^-1,
    c_in(t)  = C_in(t) [ (P_i + 2 eps t I)^-1 a_i + (Q_n + 2 eps (1-t) I)^-1 b_n ].

The branch weights omega_in do not depend on t: omega_in is the mass that
travels from source blob i to target blob n.  rho_0, rho_1 are t = 0, 1.
The optimal drift is v = sigma^2 grad log phi = 2 eps grad log phi(t, z), and
m = rho v.  With one component on each side this reduces exactly to
problems_2d.analytical_sb_gaussian_2d (checked numerically).

Everything lives on R^D.  On the Neumann box [0,1]^D the reference is exact
only up to the mass outside the box: keep every c_in(t) 4-5 standard
deviations of C_in(t) away from the walls.  The code's rho_0/rho_1 are
renormalized on the grid anyway, so the mismatch is the tail mass, not O(1).
"""
from __future__ import annotations

import numpy as np


def gauss(z, mean, cov):
    """N(z | mean, cov) evaluated at z of shape (..., D)."""
    D = z.shape[-1]
    L = np.linalg.cholesky(cov)
    r = np.linalg.solve(L, (z - mean)[..., None])[..., 0]
    q = np.sum(r * r, axis=-1)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return np.exp(-0.5 * q - 0.5 * logdet - 0.5 * D * np.log(2 * np.pi))


class MixtureSB:
    """Closed-form SB from phihat_0 = sum u_i N(a_i,P_i) to phi_1 = sum w_n N(b_n,Q_n)."""

    def __init__(self, eps, u, a, P, w, b, Q):
        self.eps = float(eps)
        self.u, self.a, self.P = (np.asarray(v, dtype=float) for v in (u, a, P))
        self.w, self.b, self.Q = (np.asarray(v, dtype=float) for v in (w, b, Q))
        self.D = self.a.shape[1]
        I = np.eye(self.D)
        num = np.array([[ui * wn * gauss(ai, bn, Pi + Qn + 2 * self.eps * I)
                         for wn, bn, Qn in zip(self.w, self.b, self.Q)]
                        for ui, ai, Pi in zip(self.u, self.a, self.P)])
        self.K = num.sum()
        self.omega = num / self.K  # (I, N) branch masses, sum to 1

    # -- Schrodinger factors -------------------------------------------------
    def phi(self, t, z):
        I = np.eye(self.D)
        return sum(wn * gauss(z, bn, Qn + 2 * self.eps * (1 - t) * I)
                   for wn, bn, Qn in zip(self.w, self.b, self.Q))

    def phihat(self, t, z):
        I = np.eye(self.D)
        return sum(ui * gauss(z, ai, Pi + 2 * self.eps * t * I)
                   for ui, ai, Pi in zip(self.u, self.a, self.P))

    # -- the bridge -----------------------------------------------------------
    def components(self, t):
        """List of (omega_in, c_in(t), C_in(t)) over all branches (i, n)."""
        I = np.eye(self.D)
        out = []
        for i, (ai, Pi) in enumerate(zip(self.a, self.P)):
            Pt = np.linalg.inv(Pi + 2 * self.eps * t * I)
            for n, (bn, Qn) in enumerate(zip(self.b, self.Q)):
                Qt = np.linalg.inv(Qn + 2 * self.eps * (1 - t) * I)
                C = np.linalg.inv(Pt + Qt)
                c = C @ (Pt @ ai + Qt @ bn)
                out.append((self.omega[i, n], c, C))
        return out

    def rho(self, t, z):
        """rho_t at points z of shape (..., D)."""
        return sum(om * gauss(z, c, C) for om, c, C in self.components(t))

    def velocity(self, t, z):
        """Optimal drift v = 2 eps grad log phi(t, z), shape (..., D)."""
        I = np.eye(self.D)
        num = 0.0
        den = 0.0
        for wn, bn, Qn in zip(self.w, self.b, self.Q):
            S = Qn + 2 * self.eps * (1 - t) * I
            g = wn * gauss(z, bn, S)
            num = num + g[..., None] * (-np.linalg.solve(S, (z - bn)[..., None])[..., 0])
            den = den + g
        return 2 * self.eps * num / den[..., None]

    def momentum(self, t, z):
        return self.rho(t, z)[..., None] * self.velocity(t, z)

    # -- the objects the paper talks about -----------------------------------
    def conditional_plan(self, x, y):
        """pi*(y | x) = phi(1, y) N(y | x, 2 eps I) / phi(0, x).

        This is the paper's exp((f*(y) - |x-y|^2/2)/eps_paper) / Z_x: a Gaussian
        mixture in y whose weights depend on x.  'Sampling from the plan'
        means: draw x ~ rho_0, then y from this mixture.
        """
        I = np.eye(self.D)
        return self.phi(1.0, y) * gauss(y, x, 2 * self.eps * I) / self.phi(0.0, x)

    def conditional_mixture(self, x):
        """Weights, means, covariances of pi*(. | x) (Prop. 3.3 of the paper)."""
        I = np.eye(self.D)
        s2 = 2 * self.eps
        ws, ms, Cs = [], [], []
        for wn, bn, Qn in zip(self.w, self.b, self.Q):
            Cn = np.linalg.inv(np.linalg.inv(Qn) + I / s2)          # = eps_p (A_n + I)^-1
            mn = Cn @ (np.linalg.solve(Qn, bn) + x / s2)            # = (A_n + I)^-1 (A_n b_n + x)
            ws.append(wn * gauss(x, bn, Qn + s2 * I))
            ms.append(mn)
            Cs.append(Cn)
        ws = np.array(ws)
        return ws / ws.sum(), np.array(ms), np.array(Cs)

    # -- grid helpers for the solvers ----------------------------------------
    def pdf_func(self, t):
        """Callable f(*coords) -> rho_t on broadcast coordinate arrays (grid_2d/3d style)."""
        def f(*coords):
            z = np.stack(np.broadcast_arrays(*[np.asarray(c, dtype=float) for c in coords]), axis=-1)
            return self.rho(t, z)
        return f

    def staggered_reference(self, nt, ns, L=1.0):
        """Reference (rho, m_1, ..., m_D) on the solvers' staggered grid.

        rho: (nt-1, n_1, ..., n_D) at times k*dt, k = 1..nt-1, cell centres,
             renormalized per slice so that sum * dV = 1 (as the code does).
        m_d: (nt, ..., n_d - 1, ...) at times (k-0.5)*dt, k = 1..nt, on the
             interior d-faces, cell centres in the other directions.
        """
        ns = tuple(ns)
        dt = 1.0 / nt
        h = [L / n for n in ns]
        dV = float(np.prod(h))
        centres = [(np.arange(n) + 0.5) * hd for n, hd in zip(ns, h)]
        faces = [np.arange(1, n) * hd for n, hd in zip(ns, h)]

        def mesh(coords):
            return np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)

        Zc = mesh(centres)
        rho = np.empty((nt - 1,) + ns)
        for k in range(1, nt):
            r = self.rho(k * dt, Zc)
            rho[k - 1] = r / (r.sum() * dV)

        ms = []
        for d in range(self.D):
            coords = list(centres)
            coords[d] = faces[d]
            Zf = mesh(coords)
            shape = list(ns)
            shape[d] -= 1
            m = np.empty((nt,) + tuple(shape))
            for k in range(1, nt + 1):
                t = (k - 0.5) * dt
                r = self.rho(t, Zf)
                r = r / (r.sum() * dV)
                m[k - 1] = r * self.velocity(t, Zf)[..., d]
            ms.append(m)
        return (rho, *ms)


# ---------------------------------------------------------------------------
# Ready-made examples (checked in docs/candidate_3d_problems.tex, Sec. 2.4)
# ---------------------------------------------------------------------------
def example_split_2d(eps=0.01):
    """One blob near (0.34, 0.5) splitting into two blobs near (0.67, 0.31/0.69)."""
    return MixtureSB(
        eps,
        u=[1.0], a=[[0.30, 0.50]], P=[0.05**2 * np.eye(2)],
        w=[0.5, 0.5], b=[[0.70, 0.30], [0.70, 0.70]], Q=[0.04**2 * np.eye(2)] * 2,
    )


def example_split_3d(eps=0.005):
    """Two blobs (y = 0.34 / 0.66) going to three blobs; branch masses ~ .39/.10/.006 per source."""
    return MixtureSB(
        eps,
        u=[0.5, 0.5], a=[[0.30, 0.35, 0.50], [0.30, 0.65, 0.50]], P=[0.05**2 * np.eye(3)] * 2,
        w=[1 / 3] * 3, b=[[0.70, 0.30, 0.35], [0.70, 0.70, 0.35], [0.70, 0.50, 0.70]],
        Q=[0.04**2 * np.eye(3)] * 3,
    )
