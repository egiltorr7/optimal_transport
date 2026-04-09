%% test_pipeline_smoke.m
%
% End-to-end smoke test for the 2D discretize_then_optimize pipeline.
%
% Runs the full ADMM loop on a tiny grid and checks:
%   1. Result struct has correct field shapes
%   2. FP residual on the staggered output x is small (constraint satisfied)
%   3. ADMM residual is decreasing (solver is making progress)

clear; clc;

% Add paths relative to this test file
base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', '..', 'shared');
sh2d    = fullfile(sh_base, '2d');

addpath(fullfile(sh_base, 'utils'));
addpath(sh2d);
addpath(fullfile(sh2d, 'utils'));
addpath(fullfile(sh2d, 'problems'));
addpath(fullfile(sh2d, 'discretization'));
addpath(fullfile(sh2d, 'prox'));
addpath(fullfile(sh2d, 'projection'));
addpath(fullfile(sh2d, 'pipelines'));
addpath(fullfile(base, '..', 'prox'));     % prox_ke_cc
addpath(fullfile(base, '..', 'config'));   % cfg_ladmm_gaussian (unused here)

pass = 0; fail = 0;
tol_fp  = 1e-4;   % FP constraint tolerance (loose; small grid, few iters)

%% --- Tiny grid config ---
cfg.name       = 'smoke';
cfg.disc       = @disc_staggered_1st;
cfg.prox_ke    = @prox_ke_cc;
cfg.projection = @proj_fokker_planck_banded;
cfg.nt         = 6;
cfg.nx         = 8;
cfg.ny         = 7;
cfg.gamma      = 100;
cfg.tau        = 101;
cfg.alpha      = 1.0;
cfg.vareps     = 0.5;
cfg.max_iter   = 200;
cfg.tol        = 1e-6;

%% --- Problem ---
prob_def = prob_gaussian();

problem = setup_problem(cfg, prob_def);

nt = problem.nt;  ntm = nt - 1;
nx = problem.nx;  nxm = nx - 1;
ny = problem.ny;  nym = ny - 1;

%% --- Run pipeline ---
result = discretize_then_optimize(cfg, problem);

% -------------------------------------------------------------------------
%% 1. RESULT FIELD SHAPES
% -------------------------------------------------------------------------
[pass,fail] = check_size('rho_stag', result.rho_stag, [ntm, nx,  ny ], pass, fail);
[pass,fail] = check_size('mx_stag',  result.mx_stag,  [nt,  nxm, ny ], pass, fail);
[pass,fail] = check_size('my_stag',  result.my_stag,  [nt,  nx,  nym], pass, fail);
[pass,fail] = check_size('rho_cc',   result.rho_cc,   [nt,  nx,  ny ], pass, fail);
[pass,fail] = check_size('mx_cc',    result.mx_cc,    [nt,  nx,  ny ], pass, fail);
[pass,fail] = check_size('my_cc',    result.my_cc,    [nt,  nx,  ny ], pass, fail);

% -------------------------------------------------------------------------
%% 2. FP CONSTRAINT ON STAGGERED OUTPUT
% -------------------------------------------------------------------------
ops     = problem.ops;
rho0    = problem.rho0;
rho1    = problem.rho1;
zeros_x = zeros(nt, ny);
zeros_y = zeros(nt, nx);

x.rho = result.rho_stag;
x.mx  = result.mx_stag;
x.my  = result.my_stag;

rho_phi   = ops.interp_t_at_phi(x.rho, rho0, rho1);
nabla_rho = ops.deriv_x_at_phi(ops.deriv_x_at_m(rho_phi), zeros_x, zeros_x) ...
          + ops.deriv_y_at_phi(ops.deriv_y_at_m(rho_phi), zeros_y, zeros_y);
fp_res = ops.deriv_t_at_phi(x.rho, rho0, rho1) ...
       + ops.deriv_x_at_phi(x.mx, zeros_x, zeros_x) ...
       + ops.deriv_y_at_phi(x.my, zeros_y, zeros_y) ...
       - cfg.vareps * nabla_rho;

err_fp = max(abs(fp_res(:)));
[pass,fail] = report('FP constraint on staggered output', err_fp, tol_fp, pass, fail);

% -------------------------------------------------------------------------
%% 3. ADMM RESIDUAL DECREASING
% -------------------------------------------------------------------------
res = result.residual;
decreasing = res(end) < res(1);
if decreasing
    fprintf('  PASS  %-45s  final=%.2e  initial=%.2e\n', 'residual decreasing', res(end), res(1));
    pass = pass + 1;
else
    fprintf('  FAIL  %-45s  final=%.2e  initial=%.2e\n', 'residual decreasing', res(end), res(1));
    fail = fail + 1;
end

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
