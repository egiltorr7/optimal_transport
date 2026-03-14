%% test_setup_problem.m
%
% Tests for setup_problem.m — verifies grid, boundary conditions,
% and eigenvalue formulas used in the spectral solver.
%
% Checks:
%   1. lambda_x and lambda_t: correct size, non-negative
%   2. Poisson round-trip: apply eigenvalues, solve, recover original phi
%   3. Biharmonic round-trip: same with the full operator used in proj_fokker_planck
%   4. rho0/rho1 normalization
%   5. Grid spacing consistency

clear; clc;

% Add paths (run from tests/ or from 1d/)
base = fileparts(mfilename('fullpath'));
addpath(fullfile(base, '..'));               % setup_problem.m
addpath(fullfile(base, '..', 'config'));
addpath(fullfile(base, '..', 'problems'));
addpath(fullfile(base, '..', 'discretization'));
addpath(fullfile(base, '..', 'utils'));

%% Build problem via setup_problem
cfg      = cfg_staggered_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

nt = problem.nt;
nx = problem.nx;

pass = 0; fail = 0;
tol = 1e-10;

% -------------------------------------------------------------------------
%% 1. EIGENVALUE SIZES AND SIGN
% -------------------------------------------------------------------------
% lambda_x should be (1 x nx), lambda_t should be (nt x 1).
% Both must be non-negative (they are eigenvalues of a positive operator).
% lambda_x(1) = lambda_t(1) = 0 (zero mode, DC component).

err = double(~isequal(size(problem.lambda_x), [1, nx]));
[pass, fail] = report('lambda_x size is (1 x nx)', err, 0.5, pass, fail);

err = double(~isequal(size(problem.lambda_t), [nt, 1]));
[pass, fail] = report('lambda_t size is (nt x 1)', err, 0.5, pass, fail);

err = double(any(problem.lambda_x(:) < 0));
[pass, fail] = report('lambda_x non-negative', err, 0.5, pass, fail);

err = double(any(problem.lambda_t(:) < 0));
[pass, fail] = report('lambda_t non-negative', err, 0.5, pass, fail);

err = abs(problem.lambda_x(1));
[pass, fail] = report('lambda_x(1) == 0 (DC mode)', err, tol, pass, fail);

err = abs(problem.lambda_t(1));
[pass, fail] = report('lambda_t(1) == 0 (DC mode)', err, tol, pass, fail);

% -------------------------------------------------------------------------
%% 2. POISSON ROUND-TRIP
% -------------------------------------------------------------------------
% Strategy: pick a single DCT mode phi, compute f = (-Delta)phi via
% eigenvalues, feed f into the spectral solver, check we recover phi.
%
% This tests that lambda_x + lambda_t are the correct eigenvalues for
% the 2D Neumann Laplacian on the (nt x nx) phi-grid.

% Single DCT mode as test phi (avoids DC component issue by picking k>0)
phi_hat_in        = zeros(nt, nx);
phi_hat_in(2, 3)  = 1.0;
phi_true          = mirt_idctn(phi_hat_in);

% Compute f = (-Delta) phi using eigenvalues as ground truth
lambda    = problem.lambda_x + problem.lambda_t;   % (nt x nx) by broadcasting
f_hat     = lambda .* phi_hat_in;
f         = mirt_idctn(f_hat);

% Solve Poisson: (-Delta)^{-1} f
phi_hat_rec      = mirt_dctn(f) ./ lambda;
phi_hat_rec(1,1) = 0;                              % fix gauge (zero mean)
phi_rec          = mirt_idctn(phi_hat_rec);

% Remove mean from phi_true to match gauge
phi_true_zm = phi_true - mean(phi_true(:));
err = max(abs(phi_rec(:) - phi_true_zm(:)));
[pass, fail] = report('Poisson round-trip', err, tol, pass, fail);

% -------------------------------------------------------------------------
%% 3. BIHARMONIC ROUND-TRIP
% -------------------------------------------------------------------------
% Tests the full operator used in proj_fokker_planck:
%   (-Delta + eps^2 * Delta^2) phi = f
% i.e., eigenvalues:  lambda_x + lambda_t - eps^2 * lambda_x^2

vareps      = cfg.vareps;
lambda_bih  = lambda - vareps^2 .* problem.lambda_x.^2;

% Check operator is positive definite (all eigenvalues > 0 except DC)
lambda_bih_interior = lambda_bih;
lambda_bih_interior(1,1) = 1;   % ignore DC
err = double(any(lambda_bih_interior(:) <= 0));
[pass, fail] = report('Biharmonic operator positive definite', err, 0.5, pass, fail);

% Round-trip
f_hat_bih        = lambda_bih .* phi_hat_in;
f_bih            = mirt_idctn(f_hat_bih);

phi_hat_rec_bih      = mirt_dctn(f_bih) ./ lambda_bih;
phi_hat_rec_bih(1,1) = 0;
phi_rec_bih          = mirt_idctn(phi_hat_rec_bih);

err = max(abs(phi_rec_bih(:) - phi_true_zm(:)));
[pass, fail] = report('Biharmonic round-trip', err, tol, pass, fail);

% -------------------------------------------------------------------------
%% 4. BOUNDARY CONDITIONS
% -------------------------------------------------------------------------

err = abs(sum(problem.rho0(:)) - 1.0);
[pass, fail] = report('rho0 sums to 1', err, tol, pass, fail);

err = abs(sum(problem.rho1(:)) - 1.0);
[pass, fail] = report('rho1 sums to 1', err, tol, pass, fail);

err = double(any(problem.rho0(:) < 0));
[pass, fail] = report('rho0 non-negative', err, 0.5, pass, fail);

err = double(any(problem.rho1(:) < 0));
[pass, fail] = report('rho1 non-negative', err, 0.5, pass, fail);

% -------------------------------------------------------------------------
%% 5. GRID CONSISTENCY
% -------------------------------------------------------------------------

err = abs(problem.dt - 1/problem.nt);
[pass, fail] = report('dt == 1/nt', err, tol, pass, fail);

err = abs(problem.dx - 1/problem.nx);
[pass, fail] = report('dx == 1/nx', err, tol, pass, fail);

% Cell centers should span (dx/2, 1-dx/2)
err_lo = abs(problem.xx(1)   - problem.dx/2);
err_hi = abs(problem.xx(end) - (1 - problem.dx/2));
[pass, fail] = report('xx first cell center == dx/2',   err_lo, tol, pass, fail);
[pass, fail] = report('xx last  cell center == 1-dx/2', err_hi, tol, pass, fail);

% -------------------------------------------------------------------------
%% Summary
% -------------------------------------------------------------------------
fprintf('\n--- Results: %d passed, %d failed ---\n', pass, fail);

% -------------------------------------------------------------------------
%% Helper
% -------------------------------------------------------------------------
function [pass, fail] = report(name, err, tol, pass, fail)
    if err < tol
        fprintf('  PASS  %-40s  err = %.2e\n', name, err);
        pass = pass + 1;
    else
        fprintf('  FAIL  %-40s  err = %.2e  (tol=%.2e)\n', name, err, tol);
        fail = fail + 1;
    end
end
