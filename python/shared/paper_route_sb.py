"""Gushchin et al.'s route to a constructed Schrodinger-bridge pair, for a
grid solver: FIX rho_0 (anything smooth and positive, not necessarily a Gaussian
mixture), PICK the y-side factor phi_1 as a Gaussian mixture, and read off the
rest.  2D (tensor grids), plain numpy.  Solver convention
d_t rho + div m = eps Lap rho, i.e. sigma^2 = 2 eps.

Steps (Sec. 2.4 of docs/candidate_3d_problems.tex):

  1. phi_1(y) = sum_n w_n N(y | b_n, Q_n)                       chosen
  2. phi(t, y) = sum_n w_n N(y | b_n, Q_n + 2 eps (1-t) I)       closed form (backward heat flow)
  3. phihat_0(x) = rho_0(x) / phi(0, x)                          forced by the x-marginal
  4. phihat(t, x) = int N(x - x' | 0, 2 eps t I) phihat_0(x') dx' forward heat flow: ONE Gaussian
                                                                 convolution, done by midpoint
                                                                 quadrature on a fine grid
  5. rho_t = phi(t) phihat(t)        (K = 1 because rho_0 is normalized)
     rho_1 = phi_1 phihat(1)
  6. v = 2 eps grad log phi(t)       closed form, depends on phi_1 only;  m = rho v.

The static plan is pi*(x,y) = phihat_0(x) N(y | x, 2 eps I) phi_1(y), and the
conditional pi*(y | x) is the paper's Gaussian mixture (Prop. 3.3).
"""
from __future__ import annotations

import numpy as np

from shared.mixture_sb import gauss


class FixedRho0SB:
    def __init__(self, eps, rho0_func, w, b, Q, nfine=512, L=1.0):
        self.eps = float(eps)
        self.w, self.b, self.Q = (np.asarray(v, dtype=float) for v in (w, b, Q))
        self.D = 2
        self.L = L
        # fine quadrature grid for the one convolution
        self.nfine = nfine
        self.hf = L / nfine
        self.xf = (np.arange(nfine) + 0.5) * self.hf
        XF = np.stack(np.meshgrid(self.xf, self.xf, indexing="ij"), -1)
        r0 = np.asarray(rho0_func(XF[..., 0], XF[..., 1]), dtype=float)
        self.rho0_mass_raw = r0.sum() * self.hf**2
        self.rho0_fine = r0 / self.rho0_mass_raw           # normalized on the fine grid
        self._rho0_func = rho0_func
        self.phihat0_fine = self.rho0_fine / self.phi(0.0, XF)

    # -- the chosen factor and its backward heat flow (closed form) ---------
    def phi(self, t, z):
        I = np.eye(2)
        return sum(wn * gauss(z, bn, Qn + 2 * self.eps * (1 - t) * I)
                   for wn, bn, Qn in zip(self.w, self.b, self.Q))

    def grad_log_phi(self, t, z):
        I = np.eye(2)
        num = 0.0
        den = 0.0
        for wn, bn, Qn in zip(self.w, self.b, self.Q):
            S = Qn + 2 * self.eps * (1 - t) * I
            g = wn * gauss(z, bn, S)
            num = num + g[..., None] * (-np.linalg.solve(S, (z - bn)[..., None])[..., 0])
            den = den + g
        return num / den[..., None]

    def velocity(self, t, z):
        return 2 * self.eps * self.grad_log_phi(t, z)

    # -- the forced factor and its forward heat flow (quadrature) ----------
    def rho0(self, xs, ys):
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return np.asarray(self._rho0_func(X, Y), dtype=float) / self.rho0_mass_raw

    def phihat0(self, xs, ys):
        Z = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1)
        return self.rho0(xs, ys) / self.phi(0.0, Z)

    def phihat(self, t, xs, ys):
        """Heat flow of phihat_0 at tensor points xs x ys: K_x F K_y^T."""
        if t == 0.0:
            return self.phihat0(xs, ys)
        var = 2 * self.eps * t
        def kmat(pts):
            d = pts[:, None] - self.xf[None, :]
            return np.exp(-d * d / (2 * var)) / np.sqrt(2 * np.pi * var) * self.hf
        return kmat(np.asarray(xs)) @ self.phihat0_fine @ kmat(np.asarray(ys)).T

    # -- the bridge ---------------------------------------------------------
    def rho(self, t, xs, ys):
        Z = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1)
        return self.phi(t, Z) * self.phihat(t, xs, ys)

    def momentum(self, t, xs, ys):
        Z = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1)
        return self.rho(t, xs, ys)[..., None] * self.velocity(t, Z)

    def conditional_plan(self, x, Y):
        """pi*(y|x) on points Y (..., 2) for one source point x."""
        I = np.eye(2)
        return self.phi(1.0, Y) * gauss(Y, x, 2 * self.eps * I) / self.phi(0.0, np.asarray(x))

    def staggered_reference(self, nt, nx, ny):
        """rho (nt-1, nx, ny) at k dt on cell centres; mx (nt, nx-1, ny), my (nt, nx, ny-1)
        at (k-1/2) dt on interior faces -- the layout of analytical_sb_gaussian_2d."""
        dt = 1.0 / nt
        dx, dy = self.L / nx, self.L / ny
        xc = (np.arange(nx) + 0.5) * dx
        yc = (np.arange(ny) + 0.5) * dy
        xfc = np.arange(1, nx) * dx
        yfc = np.arange(1, ny) * dy
        rho = np.empty((nt - 1, nx, ny))
        for k in range(1, nt):
            r = self.rho(k * dt, xc, yc)
            rho[k - 1] = r / (r.sum() * dx * dy)
        mx = np.empty((nt, nx - 1, ny))
        my = np.empty((nt, nx, ny - 1))
        for k in range(1, nt + 1):
            t = (k - 0.5) * dt
            r = self.rho(t, xfc, yc); r /= r.sum() * dx * dy
            Zx = np.stack(np.meshgrid(xfc, yc, indexing="ij"), -1)
            mx[k - 1] = r * self.velocity(t, Zx)[..., 0]
            r = self.rho(t, xc, yfc); r /= r.sum() * dx * dy
            Zy = np.stack(np.meshgrid(xc, yfc, indexing="ij"), -1)
            my[k - 1] = r * self.velocity(t, Zy)[..., 1]
        return rho, mx, my


def ring_rho0(centre=(0.35, 0.50), radius=0.12, width=0.025):
    """A smooth annulus: rho_0 proportional to exp(-(|x-c| - r)^2 / (2 w^2)). Not a Gaussian mixture."""
    cx, cy = centre
    def f(X, Y):
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        return np.exp(-((r - radius) ** 2) / (2 * width**2))
    return f


def example_ring_to_two_2d(eps=0.01, nfine=512):
    """Ring centred at (0.35, 0.5) -> two blobs near (0.70, 0.30) and (0.70, 0.70)."""
    return FixedRho0SB(eps, ring_rho0(), w=[0.5, 0.5], b=[[0.70, 0.30], [0.70, 0.70]],
                       Q=[0.04**2 * np.eye(2)] * 2, nfine=nfine)


def smoothed_box(X, Y, lo, hi, w):
    """Indicator of the rectangle [lo_x,hi_x]x[lo_y,hi_y] convolved with N(0, w^2 I): a
    product of erf differences. Flat top, straight edges rounded over a width w, strictly
    positive, closed form, and nothing like a Gaussian."""
    from scipy.special import erf
    s = np.sqrt(2.0) * w
    fx = 0.5 * (erf((X - lo[0]) / s) - erf((X - hi[0]) / s))
    fy = 0.5 * (erf((Y - lo[1]) / s) - erf((Y - hi[1]) / s))
    return fx * fy


def cross_rho0(centre=(0.33, 0.50), arm=0.15, thick=0.05, w=0.012):
    """A plus-sign: union of a horizontal and a vertical bar (overlap counted once),
    edges smoothed over w. Not a Gaussian mixture; has a plateau and four concave corners."""
    cx, cy = centre
    def f(X, Y):
        h = smoothed_box(X, Y, (cx - arm, cy - thick), (cx + arm, cy + thick), w)
        v = smoothed_box(X, Y, (cx - thick, cy - arm), (cx + thick, cy + arm), w)
        c = smoothed_box(X, Y, (cx - thick, cy - thick), (cx + thick, cy + thick), w)
        return h + v - c
    return f


def example_cross_to_two_2d(eps=0.01, nfine=512):
    """Plus-sign centred at (0.33, 0.5) -> two blobs near (0.70, 0.30) and (0.70, 0.70)."""
    return FixedRho0SB(eps, cross_rho0(), w=[0.5, 0.5], b=[[0.70, 0.30], [0.70, 0.70]],
                       Q=[0.04**2 * np.eye(2)] * 2, nfine=nfine)
