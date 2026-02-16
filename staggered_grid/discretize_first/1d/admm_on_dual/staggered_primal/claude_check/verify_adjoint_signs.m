%% Verify Adjoint Signs Using Integration by Parts
% This checks the fundamental relationship between derivatives and their adjoints

clear;

fprintf('==============================================\n');
fprintf('VERIFYING ADJOINT SIGNS VIA INTEGRATION BY PARTS\n');
fprintf('==============================================\n\n');

%% Mathematical background
fprintf('MATHEMATICAL BACKGROUND\n');
fprintf('-----------------------\n\n');

fprintf('For operator D = d/dt, the adjoint D* satisfies:\n');
fprintf('  <Du, v> = <u, D*v>\n\n');

fprintf('Integration by parts:\n');
fprintf('  ∫ (du/dt)·v dt = [u·v] - ∫ u·(dv/dt) dt\n\n');

fprintf('With ZERO boundary conditions (homogeneous Dirichlet: u(0)=u(1)=0):\n');
fprintf('  ∫ (du/dt)·v dt = -∫ u·(dv/dt) dt\n');
fprintf('  So: D* = -D  (adjoint is negative derivative)\n\n');

fprintf('With NATURAL boundary conditions (Neumann: du/dt has no BC):\n');
fprintf('  The boundary terms don''t vanish in general\n');
fprintf('  The adjoint relationship is more complex\n\n');

fprintf('KEY INSIGHT for staggered grids:\n');
fprintf('  - deriv_t_at_phi maps from faces to centers\n');
fprintf('  - deriv_t_at_rho maps from centers to faces\n');
fprintf('  - These live on different grids!\n');
fprintf('  - The adjoint relationship must account for grid differences\n\n');

%% Test adjoint numerically
fprintf('NUMERICAL ADJOINT TEST\n');
fprintf('----------------------\n\n');

nt = 63; ntm = nt-1;
dt = 1/nt;

% Build operators
c = [1 -1 zeros(1,ntm-1)];
r = [1 zeros(1,ntm-1)];
D_t_phi = toeplitz(c,r)/dt;  % (nt, ntm) - faces to centers

c = [-1 zeros(1,ntm-1)];
r = [-1 1 zeros(1,ntm-1)];
D_t_rho = toeplitz(c,r)/dt;  % (ntm, nt) - centers to faces

fprintf('D_t_phi: %d × %d (faces → centers)\n', size(D_t_phi,1), size(D_t_phi,2));
fprintf('D_t_rho: %d × %d (centers → faces)\n\n', size(D_t_rho,1), size(D_t_rho,2));

% Test: is D_t_rho = -D_t_phi^T?
fprintf('Testing if D_t_rho = -D_t_phi^T:\n');
fprintf('  ||D_t_rho + D_t_phi^T|| = %.2e\n', norm(D_t_rho + D_t_phi', 'fro'));

if norm(D_t_rho + D_t_phi', 'fro') < 1e-10
    fprintf('  ✓ YES! D_t_rho = -D_t_phi^T\n\n');
    fprintf('  This means deriv_t_at_rho is the NEGATIVE ADJOINT of deriv_t_at_phi\n\n');
else
    fprintf('  ✗ NO, they are not negative adjoints\n\n');
end

%% Implication for projection formula
fprintf('IMPLICATION FOR PROJECTION\n');
fprintf('--------------------------\n\n');

fprintf('Let A = [∂/∂t, ∂/∂x] be the divergence operator.\n');
fprintf('For projection onto ker(A), we have:\n');
fprintf('  proj(μ,ψ) = (μ,ψ) - A^T(AA^T)^{-1}A(μ,ψ)\n\n');

fprintf('If ∂/∂t has adjoint -∂/∂t (negative!), then:\n');
fprintf('  A^T = [-∂/∂t, -∂/∂x]\n\n');

fprintf('And AA^T:\n');
fprintf('  AA^T φ = A([-∂φ/∂t, -∂φ/∂x])\n');
fprintf('         = -∂²φ/∂t² - ∂²φ/∂x²\n');
fprintf('         = -Δφ\n\n');

fprintf('So we solve:\n');
fprintf('  -Δφ = A(μ,ψ) = ∂μ/∂t + ∂ψ/∂x\n\n');

fprintf('And the projection is:\n');
fprintf('  (ρ,m) = (μ,ψ) - A^T φ\n');
fprintf('        = (μ,ψ) - [-∂φ/∂t, -∂φ/∂x]\n');
fprintf('        = (μ,ψ) + [∂φ/∂t, ∂φ/∂x]\n');
fprintf('        = [μ + ∂φ/∂t, ψ + ∂φ/∂x]\n\n');

fprintf('CONCLUSION:\n');
fprintf('  ρ = μ + ∂φ/∂t  (PLUS sign)\n');
fprintf('  m = ψ + ∂φ/∂x  (PLUS sign)\n\n');

fprintf('The current code is CORRECT!\n\n');

%% Why did I think it should be minus?
fprintf('WHY THE CONFUSION?\n');
fprintf('------------------\n\n');

fprintf('The standard projection formula in textbooks often writes:\n');
fprintf('  proj(v) = v - A^T(AA^T)^{-1}Av\n\n');

fprintf('But this assumes A^T is the "positive" adjoint.\n\n');

fprintf('With staggered grids and integration by parts:\n');
fprintf('  The adjoint of ∂/∂t is -∂/∂t (NEGATIVE!)\n');
fprintf('  So A^T = [-∂/∂t, -∂/∂x]\n\n');

fprintf('When we compute proj(μ,ψ) - A^T φ:\n');
fprintf('  We get (μ,ψ) - [-∂φ/∂t, -∂φ/∂x] = (μ,ψ) + [∂φ/∂t, ∂φ/∂x]\n');
fprintf('  Hence the PLUS signs!\n\n');

%% Verify with explicit adjoint test
fprintf('EXPLICIT ADJOINT VERIFICATION\n');
fprintf('-----------------------------\n\n');

rng(42);
u = randn(ntm, 1);  % On faces
v = randn(nt, 1);   % On centers

% Compute <D_t_phi u, v> where D_t_phi: faces → centers
Du = D_t_phi * u;
inner1 = dot(Du, v) * dt;

% Compute <u, D_t_phi^T v>
DTv = D_t_phi' * v;
inner2 = dot(u, DTv) * dt;

fprintf('Testing <D_t_phi u, v> = <u, D_t_phi^T v>:\n');
fprintf('  <D_t_phi u, v> = %.10f\n', inner1);
fprintf('  <u, D_t_phi^T v> = %.10f\n', inner2);
fprintf('  Difference: %.2e\n\n', abs(inner1 - inner2));

% Now test with negative
DTv_neg = -D_t_rho * v;
inner3 = dot(u, DTv_neg) * dt;

fprintf('Testing <D_t_phi u, v> = <u, -D_t_rho v>:\n');
fprintf('  <D_t_phi u, v> = %.10f\n', inner1);
fprintf('  <u, -D_t_rho v> = %.10f\n', inner3);
fprintf('  Difference: %.2e\n\n', abs(inner1 - inner3));

if abs(inner1 - inner3) < 1e-10
    fprintf('  ✓ CONFIRMED: D_t_rho is the NEGATIVE adjoint of D_t_phi\n\n');
end

%% Final verdict
fprintf('==============================\n');
fprintf('FINAL VERDICT\n');
fprintf('==============================\n\n');

fprintf('The projection operator in ot1d_admm.m lines 262-263:\n\n');
fprintf('  rho_out = mu + deriv_t_at_rho(phi_temp);\n');
fprintf('  m_out   = psi + deriv_x_at_m(phi_temp);\n\n');

fprintf('is MATHEMATICALLY CORRECT!\n\n');

fprintf('The PLUS signs are correct because:\n');
fprintf('  1. deriv_t_at_rho is the NEGATIVE adjoint of deriv_t_at_phi\n');
fprintf('  2. A^T = [-∂/∂t, -∂/∂x] (negative derivatives)\n');
fprintf('  3. proj = (μ,ψ) - A^T φ = (μ,ψ) + [∂φ/∂t, ∂φ/∂x]\n\n');

fprintf('My initial analysis was WRONG because I did not account for\n');
fprintf('the negative adjoint relationship from integration by parts.\n\n');

fprintf('Sorry for the confusion!\n');
