# Candidate realistic 3D problems for the dynamic OT / Schrödinger-bridge solver

Survey compiled 2026-09-30 (web sources; nothing here has been run). Items the
survey could not verify are flagged. Solver constraints assumed: uniform grid
on [0,1]^3, no-flux walls, equal-mass nonnegative marginals, nt=nx=ny=nz up to
96 in fp64 (112 in mixedc) on one GH200.

## 0. Convention warning (affects every closed-form reference)

The Gaussian closed forms below use the reference SDE dX = sigma dW, i.e. the
static problem min ∫|x-y|^2 dpi + 2 sigma^2 KL(pi | a⊗b) (Janati et al.). Its
Fokker-Planck equation is d_t rho + div(rho v) = (sigma^2/2) Lap rho. If our
diffusion term is written eps Lap rho then sigma^2 = 2 eps; if (eps/2) Lap rho,
sigma^2 = eps. Check before comparing costs or rho_t.

## 1. Verification problems with reference solutions

### 1a. Anisotropic / unequal-covariance Gaussians (closed form) -- top verification pick
- Bunne, Hsieh, Cuturi, Krause, "The Schrödinger Bridge between Gaussian
  Measures has a Closed Form", AISTATS 2023, arXiv:2202.05722, Theorem 3
  (constant sigma, Brownian reference), with Sigma = Sigma_0, Sigma' = Sigma_1:
  - D_s = (4 Sigma^{1/2} Sigma' Sigma^{1/2} + sigma^4 I)^{1/2},
    C_s = 1/2 (Sigma^{1/2} D_s Sigma^{-1/2} - sigma^2 I)   (C_s^2 + sigma^2 C_s = Sigma Sigma', eq. B.18)
  - mu_t = (1-t) mu_0 + t mu_1  (eq. 26)
  - Sigma_t = (1-t)^2 Sigma_0 + t^2 Sigma_1 + t(1-t)(C_s + C_s^T + sigma^2 I)  (eq. 27)
  - optimal SDE dX = f(t,X) dt + sigma dW, f(t,x) = S_t^T Sigma_t^{-1}(x - mu_t) + mu_t'
    (eqs. 28-29; take S_t from the PDF, the ar5iv rendering is garbled).
- Janati, Muzellec, Peyré, Cuturi, "Entropic OT between (unbalanced) Gaussian
  measures has a closed form", NeurIPS 2020, arXiv:2006.02572, Thm 1: plan
  pi* ~ N((a,b), [[A, C_s],[C_s^T, B]]), cost
  OT_s = |a-b|^2 + Tr A + Tr B - Tr D_s + d sigma^2 (1 - log 2 sigma^2) + sigma^2 log det(D_s + sigma^2 I).
  (Our objective may omit the entropy term -- compare the kinetic part via the plan or add the KL term.)
- Mallasto, Gerolin, Minh, "Entropy-regularized 2-Wasserstein distance between
  Gaussian measures", Information Geometry 5 (2022), arXiv:2006.03416 --
  independent derivation, useful to cross-check conventions.
- Momentum reference: the SB current velocity is affine for Gaussian
  marginals, v_t(x) = mu_t' + M_t (x - mu_t) with symmetric M_t solving the
  Lyapunov equation M_t Sigma_t + Sigma_t M_t = Sigma_t' (continuity equation;
  Sigma_t' explicit from eq. 27), m = rho_t v_t. (Agent's derivation -- verify
  against eq. 29 minus (sigma^2/2) grad log rho_t.)
- Fit: perfect if 3-4 sigma of every Sigma_t stays inside the box (Sigma_t
  bulges above the linear interpolant by t(1-t) sigma^2 I). Good sub-cases:
  rotated anisotropic covariance Sigma_1 = R Sigma_0 R^T; Sigma_0 = Sigma_1
  with mu_0 != mu_1; the "self-bridge" rho_0 = rho_1, static classically but
  with Sigma_t = Sigma + t(1-t)(C_s + C_s^T + sigma^2 I - 2 Sigma), a pure-eps
  effect. Effort low. Paper value: essential (reviewers will expect it).

### 1b. Separable (product) problems -- free reference from the 1D code
For quadratic cost the Gibbs kernel factorizes, so the SB between products
rho_0 = ⊗ a_i, rho_1 = ⊗ b_i is the product of the three 1D bridges; this also
holds on the Neumann box (the Neumann heat kernel factorizes). Any non-Gaussian
1D pair solved accurately in 1D (bimodal -> unimodal, box -> bump) gives an
exact 3D rho and m. This is the 2D product-Sinkhorn trick one dimension up.
Effort trivial. Paper value moderate, but a strong code check.

### 1c. Grid-exact Sinkhorn reference on the Neumann box (any pair)
rho_t = (P_t phi)(P_{1-t} psi) with P the Neumann heat semigroup applied by
DCT and phi, psi the Schrödinger potentials from Sinkhorn on OUR grid: the
exact discrete-space SB for the same no-flux BC, for any rho_0, rho_1 (gives
rho_t, not m). A few hundred DCT-Sinkhorn iterations at 96^3 is cheap. Works
for all the application problems below and isolates the LADMM/time-stepping error.

### 1d. Constructed pairs with known SB (Gushchin et al.)
Gushchin, Kolesov, Mokrov et al., "Building the Bridge of Schrödinger: A
Continuous Entropic OT Benchmark", NeurIPS 2023, arXiv:2306.10161,
https://github.com/ngushchin/EntropicOTBenchmark. Given rho_0 and a
log-sum-exp quadratic potential, the conditional pi*(.|x) is an explicit
Gaussian mixture (Prop. 3.3) and the optimal drift is explicit; rho_1 has no
closed form but on a 3D grid it is a quadrature of Gaussian mixtures, i.e.
computable to machine precision. Tested in D in {2,16,64,128}; nothing stops
D=3. Non-Gaussian multimodal exact references. Effort low-moderate. Paper
value good, novel in the grid-solver literature.

### 1e. Reflected SB theory (matches the no-flux walls)
Caluya & Halder, "Reflected Schrödinger Bridge: Density Control with Path
Constraints", ACC 2021, arXiv:2003.13895: forward-backward PDE system with a
Neumann condition on the forward factor and an oblique Neumann condition on the
backward factor. Cite as the continuous problem the discretization targets;
justifies tests where mass touches the walls (uniform -> bump).

## 2. Application problems

### 2a. Cosmology: Lagrangian reconstruction (MAK) as dynamic OT -- strongest story, boundary mismatch
- Literature: Frisch, Matarrese, Mohayaee, Sobolevski, Nature 417:260 (2002);
  Brenier, Frisch, Hénon, Loeper, Matarrese, Mohayaee, Sobolevskii, MNRAS
  346:501 (2003) "Reconstruction of the early Universe as a convex
  optimization problem" (MAK); Lévy, Mohayaee, von Hausegger, MNRAS 506:1165
  (2021) semi-discrete OT; von Hausegger, Lévy, Mohayaee, PRL 128:201302
  (2022) BAO; Nikakhtar, Sheth, Lévy, Mohayaee, PRL 129 (2022);
  Nikakhtar et al. arXiv:2307.03671 (redshift space). Displacement interpolation
  from a uniform initial density to the z=0 density (Zel'dovich = quadratic-cost OT).
- Schrödinger-bridge version: the agent found NO paper doing an entropic/SB
  cosmological reconstruction. A 3D SB reconstruction with eps read as a
  velocity-dispersion / multistreaming smoothing would be new; the physical
  meaning is heuristic -- say so rather than claim it.
- Boundary conditions: all simulation tests use a PERIODIC box. Observational
  work (Lavaux et al., MNRAS 2008, 2MRS) pads with a homogeneous particle
  distribution and only trusts the inner region. Recipe for our box: take a
  sub-cube, pad by a margin >= the typical displacement (~10 h^-1 Mpc at z=0),
  evaluate only the interior; no-flux is then a controlled approximation.
- Data: CAMELS Multifield Dataset (CMD), Villaescusa-Navarro et al., ApJS
  259:61 (2022): 3D grids Mtot/Mcdm/Mgas at 128^3/256^3/512^3 over
  (25 h^-1 Mpc)^3, z = 0, 0.5, 1, 1.5, 2; float32 .npy, 8 MB per 128^3 grid;
  https://camels-multifield-dataset.readthedocs.io. Quijote: 3D density fields
  at 64-512^3 over (1000 h^-1 Mpc)^3,
  https://quijote-simulations.readthedocs.io/en/latest/df.html. A 128^3 CMD
  z=1 -> z=0 pair downsampled to 96^3 is ready-made; uniform -> z=0 is the MAK
  problem proper.
- Sanity check: OT displacement vs true Lagrangian displacements (particle IDs
  from raw snapshots) or vs the MAK / semi-discrete solution (Lévy's code is
  public). Dynamic range at z=0 is huge (voids ~0.1, filaments >100x mean):
  expect stiffness; consider smoothing as Nikakhtar et al. do.
- Effort moderate. Paper value high.

### 2b. Cryo-EM density maps (EMDB) -- best real-data fit to the box
- Fit: maps are on uniform Cartesian grids with zero density at the box edges,
  nonnegative after thresholding, mass renormalizable; conformational change =
  physically meaningful displacement interpolation.
- Literature doing this with STATIC entropic OT: Ecoffet, Poitevin, Dao Duc,
  "MorphOT: transport-based interpolation between EM maps with UCSF ChimeraX",
  Bioinformatics 36:5528 (2020), bioRxiv 10.1101/2020.09.08.286302; Ecoffet et
  al., AIMS Mathematics 7(1) (2022), PMC9377683 -- entropic W2 barycenters via
  Gaussian convolution at 60^3/120^3/240^3, default 60^3, gamma=1, <=1500 its;
  assume equal mass. Alignment: Riahi et al., "AlignOT", IEEE TCBB 2023
  (arXiv:2210.09361); "EMPOT", arXiv:2311.00850; PRX Life 3:023003 (2025).
  Zelesko, Moscovich, Kileel, Singer, ISBI 2020 (simulated ATP synthase;
  availability not verified).
- Concrete pair (used by MorphOT): Mm-cpn group II chaperonin EMD-5137
  (wild-type closed, 4.3 A) and EMD-5139 (wild-type open, 8 A), Zhang et al.,
  Nature 463:379 (2010); lidless EMD-5138/5140. Same particle, same lab, same
  box: align rigidly (AlignOT/ChimeraX), resample to 64-96^3. Also E. coli ATP
  synthase rotational states EMD-20006/EMD-20008 (Sobti et al., eLife 2019).
- Reference: MorphOT's Sinkhorn barycenters at the same gamma (R^3 Gaussian
  kernel, not Neumann; negligible for zero-padded maps) plus our DCT-Sinkhorn
  (1c). Our dynamic solver adds the momentum/velocity field (residue motions).
- Effort low-moderate (https://www.ebi.ac.uk/emdb/; threshold, align,
  resample). Paper value high; "static barycenter vs dynamic bridge on
  EMD-5137 -> 5139" is a clean figure.
- Smaller alternative: molecular electron densities as Gaussian .cube files
  (EDBench, arXiv:2505.09262; QM9-derived densities). A chemRxiv 2025 preprint
  "Electron Density Transport During Chemical Reactions" uses POT/Sinkhorn on
  cube densities along reaction paths -- reactant/product densities have equal
  electron count, so equal mass automatically. Effort low. Paper value moderate/novel.

### 2c. Medical imaging
- Haker, Zhu, Tannenbaum, Angenent, IJCV 60:225 (2004), 3D brain MRI
  Monge-Kantorovich warping. Chen, Elkin, Tannenbaum et al., "Visualizing
  fluid flows via regularized optimal mass transport with applications to
  neuroscience", J. Sci. Comput. (2023) and "Unbalanced regularized OMT with
  applications to fluid flows in the brain", Sci. Rep. (2024,
  https://www.nature.com/articles/s41598-023-50874-y) -- Benamou-Brenier +
  diffusion, i.e. essentially our PDE, on cell-centred grids of 50^3
  (synthetic) and 56x106x51 (rat brain DCE-MRI, glymphatic tracer); there the
  diffusion coefficient IS physical (tracer diffusivity). Rat data not public
  as far as found; the closest existing 3D application of our formulation, to cite.
- Datasets: DIR-Lab 4DCT (https://www.dir-lab.com/): 10 cases x 10 phases,
  ~1x1x2.5 mm, 300 landmark pairs EE<->EI (sanity check on the map). Learn2Reg
  OASIS brain MRI (https://learn2reg.grand-challenge.org/): 416 skull-stripped
  1 mm T1 at 160x224x192 with label maps. BrainWeb phantoms
  (https://brainweb.bic.mni.mcgill.ca/): 181x217x181 at 1 mm with ground-truth
  tissue classes. ACDC cardiac cine: slices 5-8 mm, poor fit for a cubic grid.
- Caveats: intensities are not conserved mass (lung density changes with
  inflation; MRI intensity only roughly proportional to tissue mass); must
  renormalize and say so. DIR-Lab lungs are the most mass-like (HU ~ tissue
  density) and have real motion. Skulls/table touch the box edges unless masked.
  Effort moderate. Paper value moderate -- reviewers know unbalanced OT is the
  honest model (cite the 2024 unbalanced rOMT).

### 2d. Fluids / geoscience
- JHTDB (https://turbulence.pha.jhu.edu/) "Homogeneous buoyancy driven
  turbulence": 1024^3 periodic, 1015 frames, stores density (Atwood 0.05,
  Sc 1); cutouts via web/SciServer API. Two density frames on a 96^3 sub-cube
  = "transport a mixing scalar"; periodic, and the physics is advection-diffusion
  by a turbulent velocity, so the check is qualitative (OT velocity vs DNS
  velocity at large scales). eps physical (kappa = nu/Sc) only if the domain matches.
- ERA5 pressure levels (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels):
  specific humidity / ozone on 37 levels, 0.25 deg; lon-lat-pressure cube is
  physically non-uniform (air-mass weighting, log-pressure vertical) and
  horizontally periodic; workable for a regional box with weak boundary
  fluxes; metric not Euclidean. Fit weak, illustrative.
- OT data assimilation: Feyeux, Vidard, Nodet, NPG 25:55 (2018) (2D shallow
  water, https://npg.copernicus.org/articles/25/55/2018/); Bocquet et al.,
  NPG 31:335 (2024) (https://npg.copernicus.org/articles/31/335/2024/);
  Farchi et al., Tellus B 68:31682 (2016) Fukushima plumes (solver
  http://cerea.enpc.fr/opttrans/). All 2D; a 3D plume needs a model run.
  Effort high. Paper value moderate.

### 2e. Shape interpolation (graphics)
- Solomon, de Goes, Peyré, Cuturi, Butscher, Nguyen, Du, Guibas,
  "Convolutional Wasserstein distances", SIGGRAPH 2015: 3D indicator
  interpolation/barycenters on 60^3 and 100^3 grids
  (https://github.com/gpeyre/2015-SIGGRAPH-convolutional-ot). Papadakis,
  Peyré, Oudet, SIIMS 7:212 (2014) proximal-splitting Benamou-Brenier with
  obstacles (MATLAB, 2D+time 32^3, https://github.com/gpeyre/2013-SIIMS-ot-splitting).
  Lévy, ESAIM M2AN 49:1693 (2015) semi-discrete 3D. Haber & Horesh, NMTMA 8:97
  (2015) multilevel time-dependent OT (3D sizes not confirmed). Stanford
  bunny/armadillo/dragon (Stanford 3D Scanning Repository) voxelize trivially.
- Fit: uniform indicator densities are discontinuous (sharp fronts stress
  positivity; eps smooths them), zero at walls, equal mass after normalization.
  Reference: Solomon's barycenters at matched blur + our DCT-Sinkhorn. Effort
  low. Paper value moderate (familiar to OT reviewers, good figures, shows
  eps-smoothing of shocks); 100^3 is a natural resolution comparison point.

### 2f. Single-cell / trajectory inference and density control
- Schiebinger et al., Cell 176:928 (2019) Waddington-OT (236k cells, 39 time
  points; https://broadinstitute.github.io/wot/); Lavenant, Zhang, Kim,
  Schiebinger, "Towards a mathematical theory of trajectory inference" (Ann.
  Appl. Probab. 2024): SB with the reference diffusion as the model, eps = cell
  diffusivity. Gridding a 3D PCA/PHATE embedding (KDE at 64^3) of day-0 vs
  day-18 cells gives an SB with physically motivated eps, but growth
  (unbalanced) issues; no published 3D-gridded version found. Paper value
  moderate/novel; reviewers will ask why not the intermediate time points.
- Density control: Chen, Georgiou, Pavon, IEEE TAC 2016/2018 and Annual Review
  of Control 2021 "Optimal transport in systems and control": a swarm of
  Brownian agents steered between two 3D formations with reflecting arena
  walls IS the Caluya-Halder reflected SB; our solver gives the feedback field
  v(t,x). Synthetic, no data. Paper value moderate (controls audience).

## Ranked shortlist

1. Anisotropic / unequal-covariance Gaussians (Bunne 2023 / Janati 2020) incl.
   the self-bridge rho_0 = rho_1: mandatory verification; cost, rho_t, m all
   closed form. [verification]
2. Gushchin-style LSE mixture pairs in 3D (multimodal, exact drift; rho_1 by
   quadrature) plus product-of-1D bridges; DCT-Sinkhorn as the grid-exact
   reference for everything else. [verification]
3. Cryo-EM: EMD-5137 -> EMD-5139 (Mm-cpn closed -> open) at 64-96^3, compared
   with MorphOT's entropic barycenters. [application; best box/no-flux fit]
4. Cosmology: CAMELS CMD 128^3 Mtot grids, z=1 (or uniform) -> z=0 on a padded
   sub-cube, displacement checks vs Lagrangian truth or Lévy's semi-discrete
   MAK; discuss periodic vs no-flux and eps as heuristic velocity-dispersion
   smoothing. [application; highest narrative value]
5. Lung 4DCT (DIR-Lab, EE -> EI) with landmark checks, framed as approximately
   balanced; cite rOMT (Chen/Tannenbaum 2023-24). [application]
6. 3D shape interpolation (Stanford bunny -> armadillo at 64-100^3) replicating
   Solomon et al. 2015 resolutions, showing eps-smoothing of fronts.
   [application/benchmark]

Where eps is physical: glymphatic/tracer rOMT (diffusivity), buoyancy /
passive-scalar turbulence (molecular diffusivity, only with matching BCs),
single-cell SB (cell-state diffusivity), swarm steering (agent noise). In
cosmology and cryo-EM it is a regularization/blur (MorphOT's gamma), not a
physical constant.
