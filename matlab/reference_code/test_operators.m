% Test script to verify operator properties
clear;

nt = 63; ntp = nt+1; ntm = nt-1;
dt = 1/nt;
nx = 127; nxp = nx+1; nxm = nx-1;
dx = 1/nx;

fprintf('=== Testing Operator Dimensions ===\n');
fprintf('nt = %d, nx = %d\n', nt, nx);

% Test spatial operators
c_xm = [1 zeros(1,nxm-1)];
r_xm = [1 1 zeros(1,nxm-1)];
Ix_m = 0.5*toeplitz(c_xm,r_xm);
fprintf('interp_x_at_m: %d x %d (cell centers -> faces)\n', size(Ix_m,1), size(Ix_m,2));

c_xphi = [1 1 zeros(1,nxm-1)];
r_xphi = [1 zeros(1,nxm-1)];
Ix_phi = 0.5*toeplitz(c_xphi,r_xphi);
fprintf('interp_x_at_phi: %d x %d (faces -> cell centers)\n', size(Ix_phi,1), size(Ix_phi,2));

c_dxphi = [1 -1 zeros(1,nxm-1)];
r_dxphi = [1 zeros(1,nxm-1)];
Dx_phi = toeplitz(c_dxphi,r_dxphi)/dx;
fprintf('deriv_x_at_phi: %d x %d (cell centers -> faces)\n', size(Dx_phi,1), size(Dx_phi,2));

c_dxm = [-1 zeros(1,nxm-1)];
r_dxm = [-1 1 zeros(1,nxm-1)];
Dx_m = toeplitz(c_dxm,r_dxm)/dx;
fprintf('deriv_x_at_m: %d x %d (faces -> cell centers)\n', size(Dx_m,1), size(Dx_m,2));

% Test if they are adjoints
fprintf('\n=== Testing Adjoint Relationships ===\n');
fprintf('Should be adjoint pairs:\n');
fprintf('  interp_x_at_m and deriv_x_at_m: ||Ix_m + Dx_m''|| = %.2e (should be ~0)\n', ...
    norm(Ix_m + Dx_m', 'fro'));
fprintf('  interp_x_at_phi and deriv_x_at_phi: ||Ix_phi - Dx_phi''|| = %.2e (should be ~0)\n', ...
    norm(Ix_phi - Dx_phi', 'fro'));

% Test temporal operators
c_tm = [1 zeros(1,ntm-1)];
r_tm = [1 1 zeros(1,ntm-1)];
It_m = 0.5*toeplitz(c_tm,r_tm);
fprintf('\ninterp_t_at_rho: %d x %d (cell centers -> time faces)\n', size(It_m,1), size(It_m,2));

c_tphi = [1 1 zeros(1,ntm-1)];
r_tphi = [1 zeros(1,ntm-1)];
It_phi = 0.5*toeplitz(c_tphi,r_tphi);
fprintf('interp_t_at_phi: %d x %d (time faces -> cell centers)\n', size(It_phi,1), size(It_phi,2));

c_dtphi = [1 -1 zeros(1,ntm-1)];
r_dtphi = [1 zeros(1,ntm-1)];
Dt_phi = toeplitz(c_dtphi,r_dtphi)/dt;
fprintf('deriv_t_at_phi: %d x %d (time faces -> cell centers)\n', size(Dt_phi,1), size(Dt_phi,2));

c_dtm = [-1 zeros(1,ntm-1)];
r_dtm = [-1 1 zeros(1,ntm-1)];
Dt_m = toeplitz(c_dtm,r_dtm)/dt;
fprintf('deriv_t_at_rho: %d x %d (cell centers -> time faces)\n', size(Dt_m,1), size(Dt_m,2));

fprintf('\n=== Testing Adjoint Relationships ===\n');
fprintf('  interp_t_at_rho and deriv_t_at_rho: ||It_m + Dt_m''|| = %.2e (should be ~0)\n', ...
    norm(It_m + Dt_m', 'fro'));
fprintf('  interp_t_at_phi and deriv_t_at_phi: ||It_phi - Dt_phi''|| = %.2e (should be ~0)\n', ...
    norm(It_phi - Dt_phi', 'fro'));

% Test eigenvalues
fprintf('\n=== Testing Eigenvalue Formula ===\n');
lambda_x_current = (2*ones(1,nx) - 2*cos(pi*dx.*(0:nxm)))/dx/dx;
lambda_x_correct = 4/(dx^2) * sin(pi*dx/2*(0:nxm)).^2;
fprintf('Current eigenvalue formula: lambda_x(1) = %.4f, lambda_x(end) = %.4f\n', ...
    lambda_x_current(1), lambda_x_current(end));
fprintf('Correct eigenvalue formula: lambda_x(1) = %.4f, lambda_x(end) = %.4f\n', ...
    lambda_x_correct(1), lambda_x_correct(end));
fprintf('Difference: ||current - correct|| = %.2e\n', norm(lambda_x_current - lambda_x_correct));

% For Neumann BC with DCT, check which formula matches DCT eigenvalues
test_vec = randn(nx, 1);
% The DCT-II basis corresponds to eigenvalue 2/h^2 * (1 - cos(pi*k/n))
lambda_dct = 2/(dx^2) * (1 - cos(pi*(0:nxm)/nx));
fprintf('DCT-based eigenvalue: lambda_x(1) = %.4f, lambda_x(end) = %.4f\n', ...
    lambda_dct(1), lambda_dct(end));
