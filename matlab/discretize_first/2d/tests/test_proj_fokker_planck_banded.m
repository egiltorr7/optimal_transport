%% test_proj_fokker_planck_banded.m
%
% Tests for shared/2d/projection/proj_fokker_planck_banded.m
% and     shared/2d/utils/precomp_banded_proj.m.
%
% Tests:
%   1. Output shapes
%   2. FP constraint satisfied (eps > 0)   -- main correctness check
%   3. Pure continuity equation (eps = 0)
%   4. Idempotency
%   5. Agreement with spectral solver at eps = 0 (both should be identical)
%   6. Temporal BCs -- FP residual at t=0 and t=1 slices

clear; clc;

base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', '..', 'shared');
addpath(fullfile(sh_base, 'utils'));
addpath(fullfile(sh_base, '2d', 'discretization'));
addpath(fullfile(sh_base, '2d', 'projection'));
addpath(fullfile(sh_base, '2d', 'utils'));

%% --- Build a minimal problem struct ---
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

problem.lambda_t = (2 - 2*cos(pi*problem.dt*(0:ntm)' )) / problem.dt^2;
problem.lambda_x = (2 - 2*cos(pi*problem.dx*(0:nxm)  )) / problem.dx^2;
problem.lambda_y = (2 - 2*cos(pi*problem.dy*(0:nym)   )) / problem.dy^2;

problem.ops = disc_staggered_1st(problem);

cfg.vareps = 0.1;
cfg.gamma  = 100;

pass = 0; fail = 0;
tol  = 1e-8;
rng(42);

%% --- Precompute banded projection ---
problem.banded_proj = precomp_banded_proj(problem, cfg.vareps);

%% --- Random input near a linear interpolation ---
t_rho    = reshape((1:ntm)*problem.dt, ntm, 1, 1);
rho0_3d  = reshape(problem.rho0, 1, nx, ny);
rho1_3d  = reshape(problem.rho1, 1, nx, ny);
x_in.rho = (1 - t_rho) .* rho0_3d + t_rho .* rho1_3d + 0.01*randn(ntm, nx, ny);
x_in.mx  = 0.01 * randn(nt, nxm, ny);
x_in.my  = 0.01 * randn(nt, nx,  nym);

% -------------------------------------------------------------------------
%% 1. OUTPUT SHAPES
% -------------------------------------------------------------------------
x_out = proj_fokker_planck_banded(x_in, problem, cfg);

[pass,fail] = check_size('rho shape', x_out.rho, [ntm, nx,  ny ], pass, fail);
[pass,fail] = check_size('mx  shape', x_out.mx,  [nt,  nxm, ny ], pass, fail);
[pass,fail] = check_size('my  shape', x_out.my,  [nt,  nx,  nym], pass, fail);

% -------------------------------------------------------------------------
%% 2. FP CONSTRAINT SATISFIED (eps > 0)   -- key test
% -------------------------------------------------------------------------
res = fp_residual(x_out, problem, cfg.vareps);
err = max(abs(res(:)));
[pass,fail] = report('FP constraint satisfied (eps>0)', err, tol, pass, fail);

% -------------------------------------------------------------------------
%% 3. PURE CONTINUITY EQUATION (eps = 0)
% -------------------------------------------------------------------------
cfg0        = cfg;
cfg0.vareps = 0.0;
problem0    = problem;
problem0.banded_proj = precomp_banded_proj(problem0, 0.0);

x_out0 = proj_fokker_planck_banded(x_in, problem0, cfg0);
res0   = fp_residual(x_out0, problem0, 0.0);
err    = max(abs(res0(:)));
[pass,fail] = report('continuity equation satisfied (eps=0)', err, tol, pass, fail);

% -------------------------------------------------------------------------
%% 4. IDEMPOTENCY
% -------------------------------------------------------------------------
x_out2 = proj_fokker_planck_banded(x_out, problem, cfg);

[pass,fail] = report('idempotency: rho', max(abs(x_out2.rho(:) - x_out.rho(:))), tol, pass, fail);
[pass,fail] = report('idempotency: mx',  max(abs(x_out2.mx(:)  - x_out.mx(:))),  tol, pass, fail);
[pass,fail] = report('idempotency: my',  max(abs(x_out2.my(:)  - x_out.my(:))),  tol, pass, fail);

% -------------------------------------------------------------------------
%% 5. AGREEMENT WITH SPECTRAL SOLVER AT eps = 0
% -------------------------------------------------------------------------
x_spec = proj_fokker_planck(x_in, problem0, cfg0);

[pass,fail] = report('banded==spectral (eps=0): rho', max(abs(x_out0.rho(:) - x_spec.rho(:))), tol, pass, fail);
[pass,fail] = report('banded==spectral (eps=0): mx',  max(abs(x_out0.mx(:)  - x_spec.mx(:))),  tol, pass, fail);
[pass,fail] = report('banded==spectral (eps=0): my',  max(abs(x_out0.my(:)  - x_spec.my(:))),  tol, pass, fail);

% -------------------------------------------------------------------------
%% 6. TEMPORAL BCs
% -------------------------------------------------------------------------
[pass,fail] = report('FP residual at t=0 slice', max(abs(res(1,:,:))),   tol, pass, fail);
[pass,fail] = report('FP residual at t=1 slice', max(abs(res(end,:,:))), tol, pass, fail);

% -------------------------------------------------------------------------
fprintf('\n--- Results: %d passed, %d failed ---\n', pass, fail);

% =========================================================================
%% Local functions
% =========================================================================

function res = fp_residual(x, problem, vareps)
    ops     = problem.ops;
    rho0    = problem.rho0;
    rho1    = problem.rho1;
    nt      = problem.nt;
    nx      = problem.nx;
    ny      = problem.ny;
    zeros_x = zeros(nt, ny);
    zeros_y = zeros(nt, nx);

    dt_rho  = ops.deriv_t_at_phi(x.rho, rho0, rho1);
    dx_mx   = ops.deriv_x_at_phi(x.mx,  zeros_x, zeros_x);
    dy_my   = ops.deriv_y_at_phi(x.my,  zeros_y, zeros_y);

    rho_phi   = ops.interp_t_at_phi(x.rho, rho0, rho1);
    nabla_rho = ops.deriv_x_at_phi(ops.deriv_x_at_m(rho_phi), zeros_x, zeros_x) ...
              + ops.deriv_y_at_phi(ops.deriv_y_at_m(rho_phi), zeros_y, zeros_y);

    res = dt_rho + dx_mx + dy_my - vareps .* nabla_rho;
end

function [pass, fail] = report(name, err, tol, pass, fail)
    if err < tol
        fprintf('  PASS  %-55s  err = %.2e\n', name, err);
        pass = pass + 1;
    else
        fprintf('  FAIL  %-55s  err = %.2e  (tol=%.2e)\n', name, err, tol);
        fail = fail + 1;
    end
end

function [pass, fail] = check_size(name, out, expected, pass, fail)
    if isequal(size(out), expected)
        fprintf('  PASS  %-55s  [%s]\n', name, num2str(size(out)));
        pass = pass + 1;
    else
        fprintf('  FAIL  %-55s  expected [%s], got [%s]\n', name, num2str(expected), num2str(size(out)));
        fail = fail + 1;
    end
end
