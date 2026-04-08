%% test_proj_fokker_planck.m
%
% Tests for projection/proj_fokker_planck.m.
%
% The projection enforces:  d_t rho + d_x m = eps * d_xx rho
% with BCs: rho(0,.) = rho0, rho(1,.) = rho1, m = 0 at x=0,1.
%
% Tests:
%   1. Constraint satisfaction  -- output satisfies FP equation to near machine precision
%   2. eps=0 case               -- reduces to continuity equation d_t rho + d_x m = 0
%   3. Idempotency              -- projecting an already-projected field changes nothing
%   4. Temporal BCs             -- FP residual at t=0 and t=1 slices is small

clear; clc;

base = fileparts(mfilename('fullpath'));
run(fullfile(base, '..', 'setup_paths.m'));

%% Build problem
cfg      = cfg_staggered_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

nt  = problem.nt;   ntm = nt - 1;
nx  = problem.nx;   nxm = nx - 1;
rho0 = problem.rho0;
rho1 = problem.rho1;

pass = 0; fail = 0;
tol = 1e-8;

%% Build a test input (random, not satisfying constraint)
rng(42);
t_rho = (1:ntm)' * problem.dt;

x_in.rho = (1 - t_rho) .* rho0 + t_rho .* rho1 + 0.01*randn(ntm, nx);
x_in.mx  = 0.01 * randn(nt, nxm);

% -------------------------------------------------------------------------
%% 1. CONSTRAINT SATISFACTION (eps > 0)
% -------------------------------------------------------------------------
x_out = proj_fokker_planck(x_in, problem, cfg);

res = fp_residual(x_out.rho, x_out.mx, problem, cfg.vareps);
err = max(abs(res(:)));
[pass, fail] = report('FP constraint satisfied after projection (eps>0)', err, tol, pass, fail);

% -------------------------------------------------------------------------
%% 2. eps = 0 (pure continuity equation)
% -------------------------------------------------------------------------
cfg0        = cfg;
cfg0.vareps = 0.0;

x_out0 = proj_fokker_planck(x_in, problem, cfg0);

res0 = fp_residual(x_out0.rho, x_out0.mx, problem, 0.0);
err  = max(abs(res0(:)));
[pass, fail] = report('continuity equation satisfied after projection (eps=0)', err, tol, pass, fail);

% -------------------------------------------------------------------------
%% 3. IDEMPOTENCY
% -------------------------------------------------------------------------
% x_out already satisfies the constraint. Projecting it again should return
% the same point — this is the true test of idempotency.

x_out2 = proj_fokker_planck(x_out, problem, cfg);

err_rho = max(abs(x_out2.rho(:) - x_out.rho(:)));
err_mx  = max(abs(x_out2.mx(:)  - x_out.mx(:)));
[pass, fail] = report('idempotency: rho unchanged on re-projection', err_rho, tol, pass, fail);
[pass, fail] = report('idempotency: mx unchanged on re-projection',  err_mx,  tol, pass, fail);

% -------------------------------------------------------------------------
%% 4. TEMPORAL BOUNDARY CONDITIONS
% -------------------------------------------------------------------------
res_bc = fp_residual(x_out.rho, x_out.mx, problem, cfg.vareps);
err_t0 = max(abs(res_bc(1,:)));
err_t1 = max(abs(res_bc(end,:)));
[pass, fail] = report('FP residual at t=0 slice', err_t0, tol, pass, fail);
[pass, fail] = report('FP residual at t=1 slice', err_t1, tol, pass, fail);

% -------------------------------------------------------------------------
%% Summary
% -------------------------------------------------------------------------
fprintf('\n--- Results: %d passed, %d failed ---\n', pass, fail);

% =========================================================================
%% Local functions
% =========================================================================

function res = fp_residual(rho_in, mx_in, problem, vareps)
    ops     = problem.ops;
    rho0    = problem.rho0;
    rho1    = problem.rho1;
    nt      = problem.nt;
    zeros_x = zeros(nt, 1);

    dt_rho   = ops.deriv_t_at_phi(rho_in, rho0, rho1);
    dx_mx    = ops.deriv_x_at_phi(mx_in, zeros_x, zeros_x);
    rho_phi  = ops.interp_t_at_phi(rho_in, rho0, rho1);
    d_xx_rho = ops.deriv_x_at_phi(ops.deriv_x_at_m(rho_phi), zeros_x, zeros_x);

    res = dt_rho + dx_mx - vareps .* d_xx_rho;
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
