% Verify the projection operator sign convention
clear;

nt = 63; ntm = nt-1;
nx = 127; nxm = nx-1;
dx = 1/nx;
dt = 1/nt;

fprintf('=== Checking Projection Operator Signs ===\n\n');

% Create test data with known divergence
rng(42);
mu_test = randn(ntm, nx);  % On rho_tilde grid (faces)
psi_test = randn(nt, nxm);  % On bx grid (x-faces)

% Boundary conditions
rho0 = randn(1, nx);
rho1 = randn(1, nx);

% Derivative operators
c_dt_phi = [1 -1 zeros(1,ntm-1)];
r_dt_phi = [1 zeros(1,ntm-1)];
D_t_phi = toeplitz(c_dt_phi, r_dt_phi)/dt;

c_dx_phi = [1 -1 zeros(1,nxm-1)];
r_dx_phi = [1 zeros(1,nxm-1)];
D_x_phi = toeplitz(c_dx_phi, r_dx_phi)/dx;

c_dt_rho = [-1 zeros(1,ntm-1)];
r_dt_rho = [-1 1 zeros(1,ntm-1)];
D_t_rho = toeplitz(c_dt_rho, r_dt_rho)/dt;

c_dx_m = [-1 zeros(1,nxm-1)];
r_dx_m = [-1 1 zeros(1,nxm-1)];
D_x_m = toeplitz(c_dx_m, r_dx_m)/dx;

% Compute divergence of input
div_mu = D_t_phi * mu_test;
div_mu(1,:) = div_mu(1,:) - rho0/dt;
div_mu(end,:) = div_mu(end,:) + rho1/dt;

div_psi = psi_test * D_x_phi';
div_psi(:,1) = div_psi(:,1);  % Zero BC
div_psi(:,end) = div_psi(:,end);  % Zero BC

total_div_input = div_mu + div_psi;
fprintf('Input divergence: ||∂μ/∂t + ∂ψ/∂x|| = %.6e\n\n', norm(total_div_input(:)));

% Now test both sign conventions for projection
% Method 1: Current code (PLUS signs)
% Method 2: Mathematical derivation (MINUS signs)

% Compute Laplacian eigenvalues
lambda_x = (2*ones(1,nx) - 2*cos(pi*dx.*(0:nxm)))/dx/dx;
lambda_t = (2*ones(nt,1) - 2*cos(pi*dt.*(0:ntm)'))/dt/dt;
lambda_lap = lambda_t + lambda_x;

% Solve -Δφ = div(mu, psi)
rhs = total_div_input;

% Note: For DCT, we need to extend to the correct size
% Let's use a simple approach - assume periodic wrapping for now
% Actually, the mirt_dctn should handle Neumann BC

% For simplicity, let's manually construct phi using the formula
fprintf('Testing projection with PLUS signs (current code):\n');
% This requires calling the actual DCT functions, skip for now

fprintf('Testing projection with MINUS signs (mathematical formula):\n');
% Same issue

fprintf('\n=== Checking Derivative Sign Convention ===\n');
test_vec = [1; 2; 4; 7; 11];  % Length 5
deriv_forward = D_t_rho(1:4,1:5) * test_vec;
fprintf('Test vector: [1, 2, 4, 7, 11]\n');
fprintf('Forward difference (deriv_t_at_rho): [%.1f, %.1f, %.1f, %.1f]\n', deriv_forward);
fprintf('Expected (x[i+1] - x[i])/dt: [1, 2, 3, 4] * %g = [%.1f, %.1f, %.1f, %.1f]\n', 1/dt, 1/dt, 2/dt, 3/dt, 4/dt);

% Check if deriv_t_at_rho computes +∂/∂t or -∂/∂t
expected_forward = ([2-1, 4-2, 7-4, 11-7] / dt)';
fprintf('\nIs deriv_t_at_rho computing +∂/∂t? Difference: %.2e\n', norm(deriv_forward - expected_forward));

fprintf('\n=== Theoretical Sign Check ===\n');
fprintf('For projection onto div-free constraint:\n');
fprintf('  Standard formula: ρ = μ - ∂φ/∂t, m = ψ - ∂φ/∂x\n');
fprintf('  Current code has: ρ = μ + ∂φ/∂t, m = ψ + ∂φ/∂x\n');
fprintf('  This suggests a SIGN ERROR unless there is a compensating convention.\n\n');

fprintf('Possible explanations:\n');
fprintf('  1. Sign error in projection (most likely)\n');
fprintf('  2. Different sign convention in deriv operators\n');
fprintf('  3. Laplacian solve has opposite sign\n');
fprintf('  4. Integration by parts convention differs\n');
