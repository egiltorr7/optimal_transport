"""Synthetic rho_0 / rho_1 phantoms for the application problems of
candidate_3d_problems.tex Secs. 3.3-3.6.  No data downloads; each phantom has the
geometry, dynamic range and length scales of the real data it stands in for, on
an n^3 cell-centred grid over [0,1]^3 (or n^2 for the 2D cases).  All densities
are nonnegative and normalized to unit mass, as the solver requires.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def grid3(n):
    x = (np.arange(n) + 0.5) / n
    return np.meshgrid(x, x, x, indexing="ij")


def normalize(r, n):
    r = np.maximum(r, 0.0)
    return r / (r.sum() / n ** r.ndim)


def gauss3(X, Y, Z, c, s):
    return np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * s**2))


def ellipsoid(X, Y, Z, c, a, blur_cells=1.0, n=96):
    f = (((X - c[0]) / a[0]) ** 2 + ((Y - c[1]) / a[1]) ** 2 + ((Z - c[2]) / a[2]) ** 2 <= 1).astype(float)
    return gaussian_filter(f, blur_cells) if blur_cells else f


# ---------------------------------------------------------------- 3.3 medical
def glymphatic(n=96):
    """DCE-MRI glymphatic tracer (Chen/Elkin/Tannenbaum rOMT): a brain-shaped
    ellipsoid; rho_0 = tracer shortly after injection at the cisterna magna (base,
    posterior); rho_1 = tracer ~1 h later, spread along perivascular channels on
    the surface and diffusing into the parenchyma.  Returns rho0, rho1, brain mask."""
    X, Y, Z = grid3(n)
    c, a = (0.5, 0.5, 0.5), (0.42, 0.30, 0.27)
    brain = ellipsoid(X, Y, Z, c, a, 1.0, n)
    # injection at the cisterna magna: posterior pole, ventral side
    u0 = -np.pi / 3
    inj = (c[0], c[1] + 0.9 * a[1] * np.sin(u0), c[2] - 0.9 * a[2] * np.cos(u0))
    rho0 = gauss3(X, Y, Z, inj, 0.045) * brain
    # perivascular channels: tubes just under the surface, fanning from the
    # injection site toward the front along the ventral/lateral surface
    chan = np.zeros_like(X)
    for phk in (-1.1, -0.4, 0.4, 1.1):
        for t in np.linspace(0, 1, 70):
            u = u0 + (0.9 * np.pi / 2 - u0) * t
            ph = phk * t
            p = (c[0] + 0.9 * a[0] * np.sin(ph) * np.cos(u),
                 c[1] + 0.9 * a[1] * np.sin(u),
                 c[2] - 0.9 * a[2] * np.cos(ph) * np.cos(u))
            chan += (1 - 0.6 * t) * gauss3(X, Y, Z, p, 0.022)
    chan /= chan.sum()
    paren = gaussian_filter(chan, 5.0)                 # tracer that has entered the parenchyma
    paren /= paren.sum()
    resid = gauss3(X, Y, Z, inj, 0.06) * brain; resid /= resid.sum()
    rho1 = (0.45 * chan + 0.35 * paren + 0.20 * resid) * np.maximum(brain, 0)
    return normalize(rho0, n), normalize(rho1, n), brain


def lungs(n=96):
    """DIR-Lab 4DCT stand-in: two lung lobes, tissue density (1 - air fraction) with a
    gravity gradient.  rho_0 = end-exhale (smaller, denser), rho_1 = end-inhale
    (diaphragm descends: lobes 25% longer in z, less dense).  Tissue mass is the
    conserved quantity, so equal mass after normalization is physically right."""
    X, Y, Z = grid3(n)
    out = []
    for stretch in (1.0, 1.25):
        az = 0.24 * stretch
        cz = 0.56 - (az - 0.24) * 0.9                     # diaphragm (bottom) moves down, apex nearly fixed
        f = ellipsoid(X, Y, Z, (0.34, 0.5, cz), (0.14, 0.19, az), 1.0, n) \
            + ellipsoid(X, Y, Z, (0.66, 0.5, cz), (0.14, 0.19, az), 1.0, n)
        dens = (0.12 + 0.10 * (0.5 - (Y - 0.5)) ) / stretch**0.6          # dependent (posterior) lung denser
        out.append(f * dens)
    return normalize(out[0], n), normalize(out[1], n)


# ---------------------------------------------------------------- 3.4 fluids
def puff(n=96, K=0.004, T=1.0, wind=(0.45, 0.0, 0.12), s0=0.035):
    """Atmospheric dispersion stand-in: a Gaussian puff released at rho_0, advected by
    a uniform wind and spread by an eddy diffusivity K for time T.  rho_1 is the exact
    advection-diffusion solution, so with eps = K the SB between rho_0 and rho_1 IS
    this evolution (zero relative drift)."""
    X, Y, Z = grid3(n)
    c0 = (0.22, 0.5, 0.3)
    c1 = tuple(c0[i] + wind[i] * T for i in range(3))
    rho0 = gauss3(X, Y, Z, c0, s0)
    rho1 = gauss3(X, Y, Z, c1, np.sqrt(s0**2 + 2 * K * T))
    return normalize(rho0, n), normalize(rho1, n), c0, c1


def mixing_scalar(n=96, seed=3):
    """Stand-in for a JHTDB buoyancy-driven mixing cutout: heavy-fluid fraction.
    rho_0 = a perturbed sharp interface (heavy fluid below), rho_1 = the same scalar
    after plumes have developed (random-field-driven stirring, then diffusion).
    Periodic in reality; here only illustrative of the dynamic range."""
    rng = np.random.default_rng(seed)
    X, Y, Z = grid3(n)
    pert = gaussian_filter(rng.standard_normal((n, n)), 6); pert *= 0.02 / pert.std()
    rho0 = 0.5 * (1 - np.tanh((Z - 0.5 - pert[:, :, None]) / 0.02))
    # later: fingers -- displace the interface by a larger-amplitude, smaller-scale field and smooth
    pert1 = gaussian_filter(rng.standard_normal((n, n)), 3); pert1 *= 0.13 / pert1.std()
    rho1 = 0.5 * (1 - np.tanh((Z - 0.5 - pert1[:, :, None]) / 0.05))
    rho1 = gaussian_filter(rho1, 1.5)
    return normalize(rho0, n), normalize(rho1, n)


# ---------------------------------------------------------------- 3.5 shapes
def torus_to_sphere(n=96):
    X, Y, Z = grid3(n)
    R, r = 0.24, 0.085
    tor = ((np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - R) ** 2 + (Z - 0.5) ** 2 <= r**2).astype(float)
    vol = tor.mean()
    rad = (3 * vol / (4 * np.pi)) ** (1 / 3)                 # equal volume sphere
    sph = ((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 <= rad**2).astype(float)
    return normalize(gaussian_filter(tor, 1.0), n), normalize(gaussian_filter(sph, 1.0), n), rad


# ---------------------------------------------------------------- 3.6 cells / control
def single_cell(n=64):
    """KDE of a 3D PCA/PHATE embedding: day-0 cells one compact cluster; day-18 cells
    three fates with unequal weights plus a faint bridge of cells in transit."""
    X, Y, Z = grid3(n)
    rho0 = gauss3(X, Y, Z, (0.25, 0.5, 0.5), 0.05)
    fates = [((0.75, 0.72, 0.62), 0.07, 0.5), ((0.78, 0.30, 0.45), 0.06, 0.3), ((0.62, 0.52, 0.22), 0.05, 0.2)]
    rho1 = sum(w * gauss3(X, Y, Z, c, s) / s**3 for c, s, w in fates)
    for t in np.linspace(0.15, 0.85, 30):                    # cells still on the way to fate 1
        p = ((1 - t) * 0.25 + t * 0.75, (1 - t) * 0.5 + t * 0.72, (1 - t) * 0.5 + t * 0.62)
        rho1 += 0.004 * gauss3(X, Y, Z, p, 0.035) / 0.035**3
    return normalize(rho0, n), normalize(rho1, n)


def swarm_ring_to_cube(n=96):
    """Density control: a swarm of Brownian agents on a ring formation (thin torus in
    the plane z = 0.5) steered into a solid cube.  Reflecting arena walls = our
    Neumann box exactly."""
    X, Y, Z = grid3(n)
    R, r = 0.32, 0.035
    ring = ((np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - R) ** 2 + (Z - 0.5) ** 2 <= r**2).astype(float)
    cube = ((abs(X - 0.5) <= 0.14) & (abs(Y - 0.5) <= 0.14) & (abs(Z - 0.5) <= 0.14)).astype(float)
    return normalize(gaussian_filter(ring, 1.0), n), normalize(gaussian_filter(cube, 1.0), n)


# ---------------------------------------------------------------- 3.2 cryo-EM
def chaperonin(n=96, open_state=False, blur_cells=1.2):
    """Group II chaperonin-like map (EMD-5137 closed / EMD-5139 open stand-in): two
    stacked 8-fold rings of subunits (equatorial + intermediate domains) and eight
    apical 'lid' domains per ring.  Closed: lids fold over the cavity, forming a
    roof.  Open: lids swing outward and up, cavity exposed.  Pseudo-atom Gaussian
    blobs; returns the map normalized to unit mass."""
    X, Y, Z = grid3(n)
    c = np.array([0.5, 0.5, 0.5])
    rho = np.zeros_like(X)
    R_eq, R_lid = 0.20, 0.17
    for ring, zsign in ((0, -1), (1, +1)):
        for k in range(8):
            th = 2 * np.pi * k / 8 + (np.pi / 8 if ring else 0.0)
            e = np.array([np.cos(th), np.sin(th), 0.0])
            # equatorial domain (large), intermediate domain (medium)
            rho += 1.0 * gauss3(X, Y, Z, c + R_eq * e + zsign * np.array([0, 0, 0.055]), 0.040)
            rho += 0.7 * gauss3(X, Y, Z, c + 0.95 * R_eq * e + zsign * np.array([0, 0, 0.125]), 0.032)
            # apical lid domain: closed folds inward over the cavity, open tilts outward
            if open_state:
                p = c + 1.25 * R_lid * e + zsign * np.array([0, 0, 0.215])
            else:
                p = c + 0.45 * R_lid * e + zsign * np.array([0, 0, 0.185])
            rho += 0.55 * gauss3(X, Y, Z, p, 0.030)
            # a thin 'helix' linking intermediate and apical domains
            q = c + (0.95 * R_eq * e + (p - c)) / 2
            rho += 0.3 * gauss3(X, Y, Z, q, 0.025)
    rho = gaussian_filter(rho, blur_cells)
    return normalize(rho, n)
