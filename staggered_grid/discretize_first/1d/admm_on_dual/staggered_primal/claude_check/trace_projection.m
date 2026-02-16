%% Trace Through Projection Operator Step-by-Step
% This recreates the exact projection computation to check signs

clear;

%% Setup matching ot1d_admm.m exactly
nt = 63; ntm = nt-1;
nx = 127; nxm = nx-1;
dx = 1/nx;
dt = 1/nt;

fprintf('===========================================\n');
fprintf('TRACING PROJECTION OPERATOR STEP-BY-STEP\n');
fprintf('===========================================\n\n');

%% Create test data
rng(123);
mu = randn(ntm, nx);      % rho_tilde - on time faces
psi = randn(nt, nxm);     % bx - on space faces

% Boundary conditions
rho0 = randn(1, nx);
rho1 = randn(1, nx);

fprintf('Input dimensions:\n');
fprintf('  mu (rho_tilde): %d x %d (time faces × space centers)\n', size(mu,1), size(mu,2));
fprintf('  psi (bx):       %d x %d (time centers × space faces)\n', size(psi,1), size(psi,2));
fprintf('  rho0:           %d x %d\n', size(rho0,1), size(rho0,2));
fprintf('  rho1:           %d x %d\n\n', size(rho1,1), size(rho1,2));

%% Step 1: Compute derivatives (line 258-259 in ot1d_admm.m)
fprintf('STEP 1: Compute dμ/dt and dψ/dx\n');
fprintf('--------------------------------\n');

% deriv_t_at_phi: maps (ntm,nx) -> (nt,nx)
dmu_dt = deriv_t_at_phi(mu, rho0, rho1, nt, ntm, dt);
fprintf('  dμ/dt: %d x %d\n', size(dmu_dt,1), size(dmu_dt,2));

% deriv_x_at_phi: maps (nt,nxm) -> (nt,nx)
dirichlet_bc = zeros(nt,1);
dpsi_dx = deriv_x_at_phi(psi, dirichlet_bc, dirichlet_bc, nx, nxm, dx);
fprintf('  dψ/dx: %d x %d\n', size(dpsi_dx,1), size(dpsi_dx,2));

%% Step 2: Solve -Δφ = dμ/dt + dψ/dx (line 261)
fprintf('\nSTEP 2: Solve -Δφ = dμ/dt + dψ/dx\n');
fprintf('-----------------------------------\n');

rhs = dmu_dt + dpsi_dx;
fprintf('  RHS (div): %d x %d\n', size(rhs,1), size(rhs,2));
fprintf('  ||dμ/dt + dψ/dx|| = %.6e\n', norm(rhs(:)));

% Eigenvalues
lambda_x = (2*ones(1,nx) - 2*cos(pi*dx.*(0:nxm)))/dx/dx;
lambda_t = (2*ones(nt,1) - 2*cos(pi*dt.*(0:ntm)'))/dt/dt;
lambda_lap = lambda_t + lambda_x;

% Solve using DCT (simplified - using random for now)
% In real code: phi_temp = mirt_idctn(mirt_dctn(rhs) ./ lambda_lap)
phi_temp = randn(nt, nx);  % Placeholder
fprintf('  φ: %d x %d (on time centers × space centers)\n', size(phi_temp,1), size(phi_temp,2));

%% Step 3: Apply projection formula (line 262-263)
fprintf('\nSTEP 3: Compute projected variables\n');
fprintf('------------------------------------\n');

fprintf('\nCurrent code (PLUS signs):\n');
% deriv_t_at_rho: maps (nt,nx) -> (ntm,nx)
dphi_dt = deriv_t_at_rho(phi_temp, nt, ntm, dt);
fprintf('  ∂φ/∂t: %d x %d\n', size(dphi_dt,1), size(dphi_dt,2));

rho_out_plus = mu + dphi_dt;  % Current code
fprintf('  rho_out = mu + ∂φ/∂t: %d x %d\n', size(rho_out_plus,1), size(rho_out_plus,2));

% deriv_x_at_m: maps (nt,nx) -> (nt,nxm)
dphi_dx = deriv_x_at_m(phi_temp, nx, nxm, dx);
fprintf('  ∂φ/∂x: %d x %d\n', size(dphi_dx,1), size(dphi_dx,2));

m_out_plus = psi + dphi_dx;  % Current code
fprintf('  m_out = psi + ∂φ/∂x: %d x %d\n', size(m_out_plus,1), size(m_out_plus,2));

fprintf('\nProposed fix (MINUS signs):\n');
rho_out_minus = mu - dphi_dt;
m_out_minus = psi - dphi_dx;
fprintf('  rho_out = mu - ∂φ/∂t: %d x %d\n', size(rho_out_minus,1), size(rho_out_minus,2));
fprintf('  m_out = psi - ∂φ/∂x: %d x %d\n', size(m_out_minus,1), size(m_out_minus,2));

%% Step 4: Check divergence of outputs
fprintf('\nSTEP 4: Verify divergence-free constraint\n');
fprintf('------------------------------------------\n');

fprintf('Original input divergence: %.6e\n', norm(rhs(:)));

% Check PLUS signs
div_plus = compute_divergence(rho_out_plus, m_out_plus, rho0, rho1, ...
                               dirichlet_bc, nt, ntm, nx, nxm, dt, dx);
fprintf('\nWith PLUS signs:\n');
fprintf('  ||div(rho_out, m_out)|| = %.6e\n', norm(div_plus(:)));

% Check MINUS signs
div_minus = compute_divergence(rho_out_minus, m_out_minus, rho0, rho1, ...
                                dirichlet_bc, nt, ntm, nx, nxm, dt, dx);
fprintf('\nWith MINUS signs:\n');
fprintf('  ||div(rho_out, m_out)|| = %.6e\n', norm(div_minus(:)));

%% Step 5: Theoretical check
fprintf('\nSTEP 5: Theoretical verification\n');
fprintf('---------------------------------\n');

fprintf('\nFrom Lagrangian mechanics:\n');
fprintf('  Projection formula: ρ = μ - ∂φ/∂t, m = ψ - ∂φ/∂x\n');
fprintf('  where -Δφ = div(μ, ψ)\n\n');

fprintf('Substituting into divergence:\n');
fprintf('  div(ρ, m) = div(μ - ∂φ/∂t, ψ - ∂φ/∂x)\n');
fprintf('            = div(μ, ψ) - div(∂φ/∂t, ∂φ/∂x)\n');
fprintf('            = div(μ, ψ) - (∂²φ/∂t² + ∂²φ/∂x²)\n');
fprintf('            = div(μ, ψ) - Δφ\n');
fprintf('            = div(μ, ψ) + (-Δφ)\n');
fprintf('            = div(μ, ψ) + div(μ, ψ)  [since -Δφ = div(μ,ψ)]\n');
fprintf('            = 2·div(μ, ψ)\n\n');

fprintf('Wait, that doesn''t give zero! Let me recalculate...\n\n');

fprintf('Actually:\n');
fprintf('  div(∂φ/∂t, ∂φ/∂x) = ∂/∂t(∂φ/∂t) + ∂/∂x(∂φ/∂x)\n');
fprintf('                     = ∂²φ/∂t² + ∂²φ/∂x²\n');
fprintf('                     = Δφ\n\n');

fprintf('So:\n');
fprintf('  div(μ - ∂φ/∂t, ψ - ∂φ/∂x) = div(μ,ψ) - Δφ\n');
fprintf('                              = div(μ,ψ) - Δφ\n');
fprintf('  If -Δφ = div(μ,ψ), then Δφ = -div(μ,ψ)\n');
fprintf('  So: div(ρ,m) = div(μ,ψ) - (-div(μ,ψ)) = 2·div(μ,ψ)\n\n');

fprintf('Hmm, this suggests NEITHER formula gives zero divergence!\n');
fprintf('Let me check the integration by parts more carefully...\n\n');

%% Helper functions (replicate from ot1d_admm.m)
function out = deriv_t_at_phi(in, bc_start, bc_end, nt, ntm, dt)
    c = [1 -1 zeros(1,ntm-1)];
    r = [1 zeros(1,ntm-1)];
    Dt = toeplitz(c,r);
    Dt = Dt./dt;
    out = Dt*in;
    out(1,:) = out(1,:) - bc_start./dt;
    out(end,:) = out(end,:) + bc_end./dt;
end

function out = deriv_t_at_rho(in, nt, ntm, dt)
    c = [-1 zeros(1,ntm-1)];
    r = [-1 1 zeros(1,ntm-1)];
    Dt = toeplitz(c,r);
    Dt = Dt./dt;
    out = Dt*in;
end

function out = deriv_x_at_phi(in, bc_in, bc_out, nx, nxm, dx)
    c = [1 -1 zeros(1,nxm-1)];
    r = [1 zeros(1,nxm-1)];
    Dx = toeplitz(c,r);
    Dx = Dx./dx;
    out = in*Dx';
    out(:,1) = out(:,1) - bc_in./dx;
    out(:,end) = out(:,end) + bc_out./dx;
end

function out = deriv_x_at_m(in, nx, nxm, dx)
    c = [-1 zeros(1,nxm-1)];
    r = [-1 1 zeros(1,nxm-1)];
    Dx = toeplitz(c,r);
    Dx = Dx./dx;
    out = in*Dx';
end

function div = compute_divergence(rho, m, rho0, rho1, bc, nt, ntm, nx, nxm, dt, dx)
    % Compute ∂ρ/∂t + ∂m/∂x
    div_rho = deriv_t_at_phi(rho, rho0, rho1, nt, ntm, dt);
    div_m = deriv_x_at_phi(m, bc, bc, nx, nxm, dx);
    div = div_rho + div_m;
end
