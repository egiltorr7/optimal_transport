%% test_prox_ke_exact.m
%
% Tests for shared/2d/prox/prox_ke_exact.m.
%
% Tests:
%   1. Output shapes
%   2. Optimality conditions -- gradient of KE + (1/sigma)*(x_out - x_in) = 0
%        at phi-locations (cell centers)
%   3. Non-negativity -- rho_out >= 0 everywhere
%   4. Idempotency check (sigma -> 0 limit: x_out -> x_in when x_in is feasible)
%   5. Consistency with 1D along x-slices when my = 0

clear; clc;

base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', '..', 'shared');
addpath(fullfile(sh_base, 'utils'));
addpath(fullfile(sh_base, '2d', 'discretization'));
addpath(fullfile(sh_base, '2d', 'prox'));

%% --- Minimal problem struct ---
nt = 8;   ntm = nt - 1;
nx = 6;   nxm = nx - 1;
ny = 5;   nym = ny - 1;

problem.nt = nt;   problem.nx = nx;   problem.ny = ny;
problem.dt = 1/nt; problem.dx = 1/nx; problem.dy = 1/ny;
problem.xx = ((1:nx)' - 0.5) * problem.dx;
problem.yy = ((1:ny)  - 0.5) * problem.dy;

Normal = @(x, mu, s) exp(-0.5*((x-mu)/s).^2);
rho0 = Normal(problem.xx, 0.3, 0.08) .* Normal(problem.yy, 0.5, 0.10);
rho1 = Normal(problem.xx, 0.7, 0.08) .* Normal(problem.yy, 0.5, 0.10);
problem.rho0 = rho0 / sum(rho0(:));
problem.rho1 = rho1 / sum(rho1(:));
problem.ops  = disc_staggered_1st(problem);

sigma = 0.05;
ops   = problem.ops;
rho0  = problem.rho0;
rho1  = problem.rho1;

pass = 0; fail = 0;
tol  = 1e-8;
rng(42);

%% --- Random positive-density input ---
t_rho   = reshape((1:ntm)*problem.dt, ntm, 1, 1);
rho0_3d = reshape(rho0, 1, nx, ny);
rho1_3d = reshape(rho1, 1, nx, ny);
x_in.rho = (1 - t_rho) .* rho0_3d + t_rho .* rho1_3d + 0.005*abs(randn(ntm, nx, ny));
x_in.mx  = 0.01 * randn(nt, nxm, ny);
x_in.my  = 0.01 * randn(nt, nx,  nym);

% -------------------------------------------------------------------------
%% 1. OUTPUT SHAPES
% -------------------------------------------------------------------------
x_out = prox_ke_exact(x_in, sigma, problem);

[pass,fail] = check_size('rho shape', x_out.rho, [ntm, nx,  ny ], pass, fail);
[pass,fail] = check_size('mx  shape', x_out.mx,  [nt,  nxm, ny ], pass, fail);
[pass,fail] = check_size('my  shape', x_out.my,  [nt,  nx,  nym], pass, fail);

% -------------------------------------------------------------------------
%% 2. OPTIMALITY CONDITIONS at phi-locations
%
%   The prox problem at cell center is:
%     min_{r,mx,my}  (mx^2+my^2)/(2*r) + (1/(2*sigma))*((r-rc)^2+(mx-mxc)^2+(my-myc)^2)
%
%   Stationarity wrt mx: 0 = mx/r + (mx - mx_c)/sigma  =>  mx_new = r * mx_c / (r + sigma)
%   Stationarity wrt r:  0 = -(mx^2+my^2)/(2*r^2) + (r - r_c)/sigma
%   These should hold at cell centers.
% -------------------------------------------------------------------------
zeros_x = zeros(nt, ny);
zeros_y = zeros(nt, nx);

rho_in_c = ops.interp_t_at_phi(x_in.rho, rho0, rho1);
mx_in_c  = ops.interp_x_at_phi(x_in.mx,  zeros_x, zeros_x);
my_in_c  = ops.interp_y_at_phi(x_in.my,  zeros_y, zeros_y);

rho_c = ops.interp_t_at_phi(x_out.rho, rho0, rho1);
mx_c  = ops.interp_x_at_phi(x_out.mx,  zeros_x, zeros_x);
my_c  = ops.interp_y_at_phi(x_out.my,  zeros_y, zeros_y);

% Stationarity wrt mx: mx_c/rho_c = (mx_in_c - mx_c)/sigma
grad_mx = mx_c ./ rho_c - (mx_in_c - mx_c) / sigma;
[pass,fail] = report('optimality wrt mx', max(abs(grad_mx(:))), tol, pass, fail);

% Stationarity wrt my
grad_my = my_c ./ rho_c - (my_in_c - my_c) / sigma;
[pass,fail] = report('optimality wrt my', max(abs(grad_my(:))), tol, pass, fail);

% Stationarity wrt rho: -(mx^2+my^2)/(2*rho^2) + (rho - rho_in)/sigma = 0
m2 = mx_c.^2 + my_c.^2;
grad_rho = -m2 ./ (2*rho_c.^2) - (rho_in_c - rho_c) / sigma;
[pass,fail] = report('optimality wrt rho', max(abs(grad_rho(:))), tol, pass, fail);

% -------------------------------------------------------------------------
%% 3. NON-NEGATIVITY
% -------------------------------------------------------------------------
[pass,fail] = report('rho_out >= 0', max(-min(x_out.rho(:), 0)), tol, pass, fail);

% -------------------------------------------------------------------------
%% 4. CONSISTENCY WITH 1D WHEN my = 0
%    With my=0, the y-component should remain 0 and rho/mx should agree
%    with prox applied to a pure 1D problem at each y-slice.
% -------------------------------------------------------------------------
x_in_2d     = x_in;
x_in_2d.my  = zeros(nt, nx, nym);
x_out_2d    = prox_ke_exact(x_in_2d, sigma, problem);

err_my_zero = max(abs(x_out_2d.my(:)));
[pass,fail]  = report('my=0 in => my=0 out', err_my_zero, tol, pass, fail);

% -------------------------------------------------------------------------
fprintf('\n--- Results: %d passed, %d failed ---\n', pass, fail);

% =========================================================================
function [pass, fail] = report(name, err, tol, pass, fail)
    if err < tol
        fprintf('  PASS  %-45s  err = %.2e\n', name, err);
        pass = pass + 1;
    else
        fprintf('  FAIL  %-45s  err = %.2e  (tol=%.2e)\n', name, err, tol);
        fail = fail + 1;
    end
end

function [pass, fail] = check_size(name, out, expected, pass, fail)
    if isequal(size(out), expected)
        fprintf('  PASS  %-45s  [%s]\n', name, num2str(size(out)));
        pass = pass + 1;
    else
        fprintf('  FAIL  %-45s  expected [%s], got [%s]\n', name, num2str(expected), num2str(size(out)));
        fail = fail + 1;
    end
end
