%% test_compute_objective.m
%
% Tests for shared/2d/utils/compute_objective.m.
%
% Tests:
%   1. Output is a non-negative scalar
%   2. Zero momentum -> zero objective
%   3. Known value: constant rho=c, mx=a, my=b -> (a^2+b^2)/c * dt*dx*dy * nt*nx*ny cells

clear; clc;

base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', '..', 'shared');
sh2d    = fullfile(sh_base, '2d');
addpath(fullfile(sh_base, 'utils'));
addpath(fullfile(sh2d, 'discretization'));
addpath(fullfile(sh2d, 'utils'));

nt = 6;  ntm = nt-1;
nx = 8;  nxm = nx-1;
ny = 7;  nym = ny-1;

problem.nt = nt;  problem.nx = nx;  problem.ny = ny;
problem.dt = 1/nt; problem.dx = 1/nx; problem.dy = 1/ny;
problem.xx = ((1:nx)' - 0.5) * problem.dx;
problem.yy = ((1:ny)  - 0.5) * problem.dy;
problem.rho0 = ones(nx, ny) / (nx*ny);
problem.rho1 = ones(nx, ny) / (nx*ny);
problem.ops  = disc_staggered_1st(problem);

pass = 0; fail = 0;
tol  = 1e-10;

% -------------------------------------------------------------------------
%% 1. OUTPUT IS NON-NEGATIVE SCALAR
% -------------------------------------------------------------------------
rho = abs(randn(ntm, nx, ny)) + 0.1;
mx  = 0.1 * randn(nt, nxm, ny);
my  = 0.1 * randn(nt, nx,  nym);

obj = compute_objective(rho, mx, my, problem);

[pass,fail] = report('output is scalar', double(~isscalar(obj)), tol, pass, fail);
[pass,fail] = report('output is non-negative', max(-obj, 0), tol, pass, fail);

% -------------------------------------------------------------------------
%% 2. ZERO MOMENTUM -> ZERO OBJECTIVE
% -------------------------------------------------------------------------
obj_zero = compute_objective(rho, zeros(nt,nxm,ny), zeros(nt,nx,nym), problem);
[pass,fail] = report('zero momentum -> zero objective', obj_zero, tol, pass, fail);

% -------------------------------------------------------------------------
%% 3. KNOWN VALUE: constant fields
%    rho = c everywhere (BCs match, so rho_c = c exactly at all phi-points).
%    mx = a on staggered x-faces with zero wall BCs:
%      interior x-cells get a, boundary x-cells get 0.5*a.
%    my = b on staggered y-faces with zero wall BCs:
%      interior y-cells get b, boundary y-cells get 0.5*b.
% -------------------------------------------------------------------------
c = 2.0;  a = 0.3;  b = 0.5;

problem.rho0 = c * ones(nx, ny);
problem.rho1 = c * ones(nx, ny);

rho_const = c * ones(ntm, nx, ny);
mx_const  = a * ones(nt,  nxm, ny);
my_const  = b * ones(nt,  nx,  nym);

obj_const = compute_objective(rho_const, mx_const, my_const, problem);

sum_mx2 = nt * ny * ((nx-2)*a^2 + 2*(0.5*a)^2);
sum_my2 = nt * nx * ((ny-2)*b^2 + 2*(0.5*b)^2);
obj_expected = (sum_mx2 + sum_my2) / c * problem.dt * problem.dx * problem.dy;

[pass,fail] = report('known value (constant fields)', abs(obj_const - obj_expected), tol, pass, fail);

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
