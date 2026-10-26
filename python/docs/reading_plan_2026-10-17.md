# One-day reading plan (17 Oct 2026): examples for the 3D paper

Companion: `paper_assessment_3d.tex` has per-paper notes; `candidate_3d_problems.tex` Secs. 2.4 and 3.x have the
phantoms and figures. Convention reminder: our constraint is d_t rho + div m = eps Lap rho, so sigma^2 = 2 eps and the
EOT literature's eps is twice ours.

## Morning: verification + cosmology + cryo-EM/shapes
1. Gushchin et al., NeurIPS 2023, arXiv:2306.10161 (45 min). Sec. 2 conventions; Sec. 3: Thm 3.2, eq. 15, Prop. 3.3,
   Cor. 3.5 / eq. 16. Skip Secs. 4-6. Goal: explain "choose the factors, read off the marginals".
2. Brenier et al., MNRAS 346:501 (2003), astro-ph/0304214 (75 min). Sec. 2 setup; App. D + eq. 51 (Benamou-Brenier
   form); pp. 505-507 (periodic box, Caffarelli-Li patch argument); Figs. 8-11 (scoring). Goal: decide Zel'dovich toy
   at D=0.3-0.5 + Quijote field.
3. Levy, Mohayaee, von Hausegger, MNRAS 506:1165 (2021), arXiv:2012.09074 (25 min skim). Eq. 9, data section, Fig. 11.
4. Ecoffet, Poitevin, Dao Duc, MorphOT, Bioinformatics 36:5528 (2020) (30 min). Methods (threshold, 60^3, barycenter by
   Gaussian convolution, gamma); results on EMD-5137/5139.
5. Solomon et al., Convolutional Wasserstein distances, SIGGRAPH 2015 (40 min). Sec. 3 (heat-kernel = entropic kernel,
   gamma <-> eps), Sec. 4 + Alg. 2 (barycenters); skim 3D applications.

## Afternoon: PDE-identical application, single cell, extensions
6. Chen, Elkin, Tannenbaum et al., J. Sci. Comput. 2023 (rOMT) + Sci. Rep. 2024 (unbalanced rOMT) (45 min). Model
   section (constraint with sigma^2 diffusion), numerics (grids, pathlines/speed maps); 2024: formulation with source.
7. Papadakis, Peyre, Oudet, SIIMS 7:212 (2014) (45 min). Staggered-grid discretization section; proximal operators
   (KE prox closed form, projection via Poisson solve). Ancestor of our scheme; their 2D code is a cross-check.
8. Lavenant, Zhang, Kim, Schiebinger, AAP 2024, arXiv:2102.09204 (45 min). Intro + model (SDE with drift and sigma),
   estimator = Schrodinger problem with soft snapshot fits, main theorem statement. eps = cell diffusivity.
9. Chen, Georgiou, Pavon, Annual Review of Control 2021 (40 min). SB as stochastic control (1/2 E int |u|^2), density
   steering + linear-Gaussian Riccati closed form, references with drift. + Caluya & Halder arXiv:2003.13895 (15 min):
   the reflected-SB PDE system and boundary conditions.
10. Chizat, Peyre, Schmitzer, Vialard, JFA 2018, arXiv:1508.05216 (40 min). Intro, dynamic formulation with source,
    theorem equating it to the static problem with the cos cost (eps=0 benchmark route). + Janati et al., NeurIPS 2020,
    arXiv:2006.02572 (15 min): unbalanced-Gaussian closed form (soft-marginal variant).

## If time remains
- Chen, Georgiou, Pavon, JOTA 2016, arXiv:1412.4430: fluid-dynamic SB with the Fisher-information term (drift -> cost).
- Haasler, Ringh, Chen, Karlsson, IEEE TIT 2021, Secs. 2-3: tree-structured multi-marginal = pairwise bridges.
- Frisch et al., Nature 417:260 (2002): the one-paragraph physical story.
