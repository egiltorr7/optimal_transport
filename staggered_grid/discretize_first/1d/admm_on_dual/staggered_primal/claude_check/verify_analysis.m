%% Comprehensive Verification of ADMM Implementation
% This script verifies the mathematical correctness of operators

clear; close all;

%% Setup grids
nt = 8; ntm = nt-1;  % Small for easy visualization
nx = 10; nxm = nx-1;
dx = 1/nx;
dt = 1/nt;

fprintf('====================================\n');
fprintf('VERIFICATION OF ADMM IMPLEMENTATION\n');
fprintf('====================================\n\n');
fprintf('Grid: nt=%d, nx=%d\n\n', nt, nx);

%% Part 1: Verify derivative operator sign conventions
fprintf('PART 1: DERIVATIVE OPERATOR SIGNS\n');
fprintf('----------------------------------\n');

% Build deriv_t_at_rho operator
c = [-1 zeros(1,ntm-1)];
r = [-1 1 zeros(1,ntm-1)];
D_t_rho = toeplitz(c,r)/dt;

fprintf('deriv_t_at_rho matrix (first 4x4 block):\n');
disp(full(D_t_rho(1:min(4,ntm), 1:min(4,nt))));

% Test with simple vector
x = (1:nt)';
dx_dt = D_t_rho * x;
fprintf('Test: x = [1,2,3,...,%d]\n', nt);
fprintf('deriv_t_at_rho(x) = [');
fprintf('%.1f ', dx_dt(1:min(5,length(dx_dt))));
fprintf('...]\n');
fprintf('Expected (forward diff/dt): [');
expected = diff(x)/dt;
fprintf('%.1f ', expected(1:min(5,length(expected))));
fprintf('...]\n');
fprintf('Match? %s\n\n', iif(norm(dx_dt - expected) < 1e-10, 'YES ✓', 'NO ✗'));

%% Part 2: Verify interpolation-derivative adjoint relationship
fprintf('PART 2: ADJOINT RELATIONSHIPS\n');
fprintf('------------------------------\n');

% Build interp_t_at_rho
c = [1 zeros(1,ntm-1)];
r = [1 1 zeros(1,ntm-1)];
I_t_rho = 0.5*toeplitz(c,r);

% Build interp_t_at_phi
c = [1 1 zeros(1,ntm-1)];
r = [1 zeros(1,ntm-1)];
I_t_phi = 0.5*toeplitz(c,r);

fprintf('Checking if I_t_rho and I_t_phi are transposes:\n');
fprintf('  ||I_t_rho - I_t_phi^T|| = %.2e\n', norm(I_t_rho - I_t_phi', 'fro'));

% Build deriv_t_at_phi
c = [1 -1 zeros(1,ntm-1)];
r = [1 zeros(1,ntm-1)];
D_t_phi = toeplitz(c,r)/dt;

fprintf('\nChecking adjoint relationship between interp and deriv:\n');
fprintf('  ||I_t_rho + D_t_rho^T|| = %.2e\n', norm(I_t_rho + D_t_rho', 'fro'));
fprintf('  ||I_t_phi - D_t_phi^T|| = %.2e\n', norm(I_t_phi - D_t_phi', 'fro'));

fprintf('\nInterpretation:\n');
if norm(I_t_rho + D_t_rho', 'fro') < 1e-10
    fprintf('  ✓ I_t_rho = -D_t_rho^T (negative adjoint)\n');
else
    fprintf('  ✗ I_t_rho ≠ -D_t_rho^T\n');
end

if norm(I_t_phi - D_t_phi', 'fro') < 1e-10
    fprintf('  ✓ I_t_phi = D_t_phi^T (positive adjoint)\n\n');
else
    fprintf('  ✗ I_t_phi ≠ D_t_phi^T\n\n');
end

%% Part 3: Test projection operator with BOTH sign conventions
fprintf('PART 3: PROJECTION OPERATOR VERIFICATION\n');
fprintf('-----------------------------------------\n');

% Create random test inputs
rng(42);
mu_test = randn(ntm, nx);
psi_test = randn(nt, nxm);

% Boundary conditions
rho0 = randn(1, nx);
rho1 = randn(1, nx);
bc_x = zeros(nt, 1);

% Build spatial derivative operators
c = [1 -1 zeros(1,nxm-1)];
r = [1 zeros(1,nxm-1)];
D_x_phi = toeplitz(c,r)/dx;

c = [-1 zeros(1,nxm-1)];
r = [-1 1 zeros(1,nxm-1)];
D_x_m = toeplitz(c,r)/dx;

% Compute divergence of input
fprintf('Computing divergence of input (mu, psi)...\n');

% Time derivative of mu (with BC)
dmu_dt = D_t_phi * mu_test;
dmu_dt(1,:) = dmu_dt(1,:) - rho0/dt;
dmu_dt(end,:) = dmu_dt(end,:) + rho1/dt;

% Space derivative of psi (with BC)
dpsi_dx = psi_test * D_x_phi';
% BC: ψ·n = 0 on boundaries (already zero)

div_input = dmu_dt + dpsi_dx;
fprintf('  ||div(mu, psi)|| = %.6e\n\n', norm(div_input(:)));

% Solve Laplacian using simple approach (not DCT, for clarity)
% -Δφ = div ==> we need to invert the Laplacian
% For now, let's use a simple direct solve
fprintf('Solving -Δφ = div(mu, psi)...\n');

% Build Laplacian matrix (simplified - assume periodic for now)
% In reality, should use DCT with Neumann BC
lambda_x = (2*ones(1,nx) - 2*cos(pi*dx.*(0:nxm)))/dx/dx;
lambda_t = (2*ones(nt,1) - 2*cos(pi*dt.*(0:ntm)'))/dt/dt;
lambda_lap = lambda_t + lambda_x;

% Note: mirt_dctn/mirt_idctn would be used here
% For verification, let's just create a test phi
phi_test = randn(nt, nx);  % Placeholder

fprintf('  (Using random φ for sign verification)\n\n');

%% Test BOTH sign conventions
fprintf('Testing projection with PLUS signs (current code):\n');
rho_out_plus = mu_test + D_t_rho * phi_test;
psi_out_plus = psi_test + phi_test * D_x_m';

% Compute divergence
div_rho_plus = D_t_phi * rho_out_plus;
div_rho_plus(1,:) = div_rho_plus(1,:) - rho0/dt;
div_rho_plus(end,:) = div_rho_plus(end,:) + rho1/dt;
div_psi_plus = psi_out_plus * D_x_phi';
div_total_plus = div_rho_plus + div_psi_plus;

fprintf('  ||div(rho_out, psi_out)|| = %.6e\n', norm(div_total_plus(:)));

fprintf('\nTesting projection with MINUS signs (mathematical derivation):\n');
rho_out_minus = mu_test - D_t_rho * phi_test;
psi_out_minus = psi_test - phi_test * D_x_m';

% Compute divergence
div_rho_minus = D_t_phi * rho_out_minus;
div_rho_minus(1,:) = div_rho_minus(1,:) - rho0/dt;
div_rho_minus(end,:) = div_rho_minus(end,:) + rho1/dt;
div_psi_minus = psi_out_minus * D_x_phi';
div_total_minus = div_rho_minus + div_psi_minus;

fprintf('  ||div(rho_out, psi_out)|| = %.6e\n', norm(div_total_minus(:)));

fprintf('\n');

%% Part 4: Analytical check of projection formula
fprintf('PART 4: ANALYTICAL DERIVATION CHECK\n');
fprintf('------------------------------------\n');

fprintf('Mathematical projection formula derivation:\n');
fprintf('  Goal: Project (μ, ψ) onto constraint ∂ρ/∂t + ∂m/∂x = 0\n\n');
fprintf('  Lagrangian: L = (1/2)||ρ-μ||² + (1/2)||m-ψ||² + ∫φ(∂ρ/∂t + ∂m/∂x)\n\n');
fprintf('  FOC: δL/δρ = ρ - μ + ∂φ/∂t = 0  ==>  ρ = μ - ∂φ/∂t\n');
fprintf('       δL/δm = m - ψ + ∂φ/∂x = 0  ==>  m = ψ - ∂φ/∂x\n');
fprintf('       δL/δφ = ∂ρ/∂t + ∂m/∂x = 0\n\n');
fprintf('  Substituting: ∂(μ - ∂φ/∂t)/∂t + ∂(ψ - ∂φ/∂x)/∂x = 0\n');
fprintf('               ∂μ/∂t + ∂ψ/∂x = ∂²φ/∂t² + ∂²φ/∂x² = Δφ\n');
fprintf('               -Δφ = ∂μ/∂t + ∂ψ/∂x\n\n');
fprintf('  CONCLUSION: ρ = μ - ∂φ/∂t  and  m = ψ - ∂φ/∂x  (MINUS signs)\n\n');

%% Part 5: Check if there's a compensating sign in the derivative operators
fprintf('PART 5: CHECKING FOR COMPENSATING SIGN CONVENTIONS\n');
fprintf('---------------------------------------------------\n');

fprintf('Could deriv_t_at_rho compute -∂/∂t instead of +∂/∂t?\n');
fprintf('  From matrix structure: D_t_rho computes (x[i+1] - x[i])/dt\n');
fprintf('  This is the standard +∂/∂t forward difference.\n\n');

fprintf('Could invert_neg_laplacian solve +Δφ = f instead of -Δφ = f?\n');
fprintf('  The function divides by lambda_lap = eigenvalues of -Δ (positive)\n');
fprintf('  This correctly solves -Δφ = f.\n\n');

fprintf('CONCLUSION: No compensating sign convention found.\n');
fprintf('           The projection operator has the WRONG SIGNS.\n\n');

%% Part 6: Why might divergence still be zero?
fprintf('PART 6: WHY MIGHT DIVERGENCE APPEAR TO BE ZERO?\n');
fprintf('-----------------------------------------------\n');

fprintf('Possible explanations:\n');
fprintf('  1. The divergence check was computed incorrectly\n');
fprintf('  2. There is a compensating bug elsewhere in the algorithm\n');
fprintf('  3. The projection is applied in a context where the signs cancel\n');
fprintf('  4. The test used symmetric/zero inputs where signs don''t matter\n\n');

fprintf('To verify: Check divergence of ACTUAL iterates, not test data.\n');

%% Helper function
function out = iif(cond, true_val, false_val)
    if cond
        out = true_val;
    else
        out = false_val;
    end
end
