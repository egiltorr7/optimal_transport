% Test if operators are proper adjoints
% This is CRITICAL for the algorithm to work!

clear;
nt = 63; ntm = nt-1;
nx = 127; nxm = nx-1;
dx = 1/nx;
dt = 1/nt;

fprintf('=== Testing Adjoint Properties ===\n\n');

%% Test 1: interp_t_at_rho vs interp_t_at_phi (should they be adjoints?)
% Actually, these are both interpolation operators, so let's check what they are

c1 = [1 zeros(1,ntm-1)];
r1 = [1 1 zeros(1,ntm-1)];
I_t_rho = 0.5*toeplitz(c1,r1);  % (62, 63)

c2 = [1 1 zeros(1,ntm-1)];
r2 = [1 zeros(1,ntm-1)];
I_t_phi = 0.5*toeplitz(c2,r2);  % (63, 62)

fprintf('interp_t_at_rho: %d x %d\n', size(I_t_rho,1), size(I_t_rho,2));
fprintf('interp_t_at_phi: %d x %d\n', size(I_t_phi,1), size(I_t_phi,2));

% Check if they are transposes
error_transpose = norm(I_t_rho - I_t_phi', 'fro');
fprintf('||I_t_rho - I_t_phi''|| = %.2e ', error_transpose);
if error_transpose < 1e-10
    fprintf('✓ They are transposes!\n');
else
    fprintf('✗ NOT transposes\n');
end

%% Test 2: interp_x_at_m vs interp_x_at_phi
c1 = [1 zeros(1,nxm-1)];
r1 = [1 1 zeros(1,nxm-1)];
I_x_m = 0.5*toeplitz(c1,r1);  % (126, 127)

c2 = [1 1 zeros(1,nxm-1)];
r2 = [1 zeros(1,nxm-1)];
I_x_phi = 0.5*toeplitz(c2,r2);  % (127, 126)

fprintf('\ninterp_x_at_m: %d x %d\n', size(I_x_m,1), size(I_x_m,2));
fprintf('interp_x_at_phi: %d x %d\n', size(I_x_phi,1), size(I_x_phi,2));

error_transpose = norm(I_x_m - I_x_phi', 'fro');
fprintf('||I_x_m - I_x_phi''|| = %.2e ', error_transpose);
if error_transpose < 1e-10
    fprintf('✓ They are transposes!\n');
else
    fprintf('✗ NOT transposes\n');
end

%% Test 3: Check what the algorithm assumes
% In lines 61-62, the code uses:
% tmp_rho = rho - (gamma/tau)*interp_t_at_phi(tmp_rho, ...)
% This suggests interp_t_at_phi is the adjoint of interp_t_at_rho

fprintf('\n=== Algorithm Structure ===\n');
fprintf('Line 57-58: A*x = (interp_t_at_rho, interp_x_at_m)\n');
fprintf('Line 61-62: Uses (interp_t_at_phi, interp_x_at_phi) as A^T\n');
fprintf('For algorithm to work: A^T should be adjoint of A\n\n');

%% Test 4: Are the derivative operators consistent?
c_dt_phi = [1 -1 zeros(1,ntm-1)];
r_dt_phi = [1 zeros(1,ntm-1)];
D_t_phi = toeplitz(c_dt_phi, r_dt_phi)/dt;  % (63, 62)

c_dt_rho = [-1 zeros(1,ntm-1)];
r_dt_rho = [-1 1 zeros(1,ntm-1)];
D_t_rho = toeplitz(c_dt_rho, r_dt_rho)/dt;  % (62, 63)

fprintf('deriv_t_at_phi: %d x %d\n', size(D_t_phi,1), size(D_t_phi,2));
fprintf('deriv_t_at_rho: %d x %d\n', size(D_t_rho,1), size(D_t_rho,2));

% Check relationship between derivative and interpolation
fprintf('\nChecking: D_t_phi vs I_t_phi\n');
fprintf('||D_t_phi/dt + I_t_phi|| = %.2e\n', norm(D_t_phi*dt + I_t_phi, 'fro'));

fprintf('\nChecking: D_t_rho vs I_t_rho\n');
fprintf('||D_t_rho/dt + I_t_rho|| = %.2e\n', norm(D_t_rho*dt + I_t_rho, 'fro'));

%% Test 5: Check if derivatives are negative adjoints of interpolations
% For integration by parts: <Dx, y> = -<x, D^T y> (with appropriate BC)
% So we might expect D^T ≈ -I or similar

fprintf('\n=== Checking if deriv = -interp^T ===\n');
fprintf('||D_t_rho + I_t_rho^T|| = %.2e\n', norm(D_t_rho + I_t_rho', 'fro'));
fprintf('||D_t_phi + I_t_phi^T|| = %.2e\n', norm(D_t_phi + I_t_phi', 'fro'));

fprintf('\n');
