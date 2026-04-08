%% test_prox_ke_exact.m
%
% Tests for prox/prox_ke_exact.m.
%
% The prox takes a proximal point x_in = (rho, mx) at staggered locations,
% interpolates to cell centers, applies the exact cubic solve, interpolates back.
%
% Tests:
%   1. Non-negativity    -- rho output >= 0 always
%   2. Zero momentum     -- x_in.mx = 0 => x_out.mx = 0 exactly
%   3. Cubic residual    -- solve_cubic output satisfies the cubic equation
%   4. Momentum formula  -- at cell centers: mx_new = rho_new*mx_in/(rho_new+sigma)
%   5. Thresholding      -- at near-zero rho: output is 0; cubic holds elsewhere

clear; clc;

base = fileparts(mfilename('fullpath'));
run(fullfile(base, '..', 'setup_paths.m'));

%% Build problem
cfg      = cfg_staggered_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

nt  = problem.nt;   ntm = nt - 1;
nx  = problem.nx;   nxm = nx - 1;
ops = problem.ops;
rho0 = problem.rho0;
rho1 = problem.rho1;
sigma = 1 / cfg.gamma;

pass = 0; fail = 0;
tol_exact  = 1e-12;
tol_interp = 1e-4;

zeros_x = zeros(nt, 1);

%% Build a smooth proximal point x_in
t_rho = (1:ntm)' * problem.dt;
t_phi = ((1:nt)'  - 0.5) * problem.dt;
x_m   = (1:nxm)   * problem.dx;

x_in.rho = (1 - t_rho) .* rho0 + t_rho .* rho1;          % (ntm x nx)
x_in.mx  = 0.1 * sin(pi * t_phi) .* sin(pi * x_m);        % (nt  x nxm)

% -------------------------------------------------------------------------
%% 1. NON-NEGATIVITY
% -------------------------------------------------------------------------
x_out = prox_ke_exact(x_in, sigma, problem);

err = double(any(x_out.rho(:) < 0));
[pass, fail] = report('rho output non-negative', err, 0.5, pass, fail);

% -------------------------------------------------------------------------
%% 2. ZERO MOMENTUM
% -------------------------------------------------------------------------
% If x_in.mx = 0, the prox decouples: mx_new = 0 exactly.

x_in_zm     = x_in;
x_in_zm.mx  = zeros(nt, nxm);
x_out_zm    = prox_ke_exact(x_in_zm, sigma, problem);

err = max(abs(x_out_zm.mx(:)));
[pass, fail] = report('zero momentum in => zero momentum out', err, tol_exact, pass, fail);

% -------------------------------------------------------------------------
%% 3. CUBIC RESIDUAL AT CELL CENTERS
% -------------------------------------------------------------------------
% Interpolate x_in to cell centers (same as inside prox_ke_exact),
% call solve_cubic independently, check the algebraic residual.

rho_c = ops.interp_t_at_phi(x_in.rho, rho0, rho1);
mx_c  = ops.interp_x_at_phi(x_in.mx, zeros_x, zeros_x);

rho_ref   = solve_cubic(1, 2*sigma - rho_c, sigma^2 - 2*sigma*rho_c, ...
                        -sigma*(sigma*rho_c + 0.5*mx_c.^2));
cubic_res = rho_ref.^3 + (2*sigma - rho_c).*rho_ref.^2 ...
          + (sigma^2 - 2*sigma*rho_c).*rho_ref ...
          - sigma*(sigma*rho_c + 0.5*mx_c.^2);

err = max(abs(cubic_res(:)));
[pass, fail] = report('solve_cubic algebraic residual', err, tol_exact, pass, fail);

% Compare rho_ref with round-trip of prox output (staggered -> cell center)
rho_out_c = ops.interp_t_at_phi(x_out.rho, rho0, rho1);
err = max(abs(rho_out_c(:) - rho_ref(:)));
[pass, fail] = report('cubic rho matches prox output (round-trip)', err, tol_interp, pass, fail);

% -------------------------------------------------------------------------
%% 4. MOMENTUM FORMULA AT CELL CENTERS
% -------------------------------------------------------------------------
mx_expected_c = rho_ref .* mx_c ./ (rho_ref + sigma);
mx_out_c      = ops.interp_x_at_phi(x_out.mx, zeros_x, zeros_x);

err = max(abs(mx_out_c(:) - mx_expected_c(:)));
[pass, fail] = report('momentum formula at cell centers (round-trip)', err, tol_interp, pass, fail);

% -------------------------------------------------------------------------
%% 5. THRESHOLDING BEHAVIOR
% -------------------------------------------------------------------------
x_in_t     = x_in;
x_in_t.rho(:, 1:floor(nx/2)) = 1e-14;   % near-zero in left half

rho_c_t = ops.interp_t_at_phi(x_in_t.rho, rho0, rho1);
mx_c_t  = ops.interp_x_at_phi(x_in_t.mx,  zeros_x, zeros_x);

rho_ref_t  = solve_cubic(1, 2*sigma - rho_c_t, sigma^2 - 2*sigma*rho_c_t, ...
                         -sigma*(sigma*rho_c_t + 0.5*mx_c_t.^2));
thresh_idx = rho_ref_t <= 1e-12;
active_idx = ~thresh_idx;

x_out_t   = prox_ke_exact(x_in_t, sigma, problem);
rho_out_tc = ops.interp_t_at_phi(x_out_t.rho, rho0, rho1);
mx_out_tc  = ops.interp_x_at_phi(x_out_t.mx,  zeros_x, zeros_x);

if any(thresh_idx(:))
    err = max(abs(rho_out_tc(thresh_idx)));
    [pass, fail] = report('thresholded rho == 0', err, tol_interp, pass, fail);
    err = max(abs(mx_out_tc(thresh_idx)));
    [pass, fail] = report('thresholded mx == 0',  err, tol_interp, pass, fail);
else
    fprintf('  SKIP  thresholding tests (no points triggered threshold)\n');
end

if any(active_idx(:))
    cubic_res_t = rho_ref_t.^3 + (2*sigma - rho_c_t).*rho_ref_t.^2 ...
                + (sigma^2 - 2*sigma*rho_c_t).*rho_ref_t ...
                - sigma*(sigma*rho_c_t + 0.5*mx_c_t.^2);
    err = max(abs(cubic_res_t(active_idx)));
    [pass, fail] = report('cubic residual at active points', err, tol_exact, pass, fail);
end

% -------------------------------------------------------------------------
%% Summary
% -------------------------------------------------------------------------
fprintf('\n--- Results: %d passed, %d failed ---\n', pass, fail);

% -------------------------------------------------------------------------
function [pass, fail] = report(name, err, tol, pass, fail)
    if err < tol
        fprintf('  PASS  %-45s  err = %.2e\n', name, err);
        pass = pass + 1;
    else
        fprintf('  FAIL  %-45s  err = %.2e  (tol=%.2e)\n', name, err, tol);
        fail = fail + 1;
    end
end
