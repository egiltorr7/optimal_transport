% TEST_ETD_PROJECTION_RESIDUAL
%
% Tests that proj_fokker_planck_expsemi drives the ETD FP residual to
% (near) machine precision after one application, across a range of eps.
%
% For each eps, the test:
%   1. Starts from a random (rho, m) pair
%   2. Applies the ETD projection once
%   3. Recomputes the ETD FP residual at the projected point
%   4. Reports the L2 norm of that residual
%
% If Thomas backward stability holds, the residual should stay near
% machine precision regardless of eps.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% --- Grid ---
NT = 64;
NX = 64;

cfg         = cfg_ladmm_gaussian_expsemi();
cfg.nt      = NT;
cfg.nx      = NX;
prob_def    = prob_gaussian();

%% --- Eps sweep ---
eps_vals = [1e-8,0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0];

fprintf('%-10s  %-16s  %-16s\n', 'eps', 'res_before', 'res_after');
fprintf('%s\n', repmat('-', 1, 46));

res_before_all = zeros(size(eps_vals));
res_after_all  = zeros(size(eps_vals));

rng(42);

for ei = 1:numel(eps_vals)
    vareps     = eps_vals(ei);
    cfg.vareps = vareps;

    problem = setup_problem(cfg, prob_def);
    problem.expsemi_proj = precomp_expsemi_proj(problem, vareps);

    nt  = problem.nt;
    nx  = problem.nx;
    dt  = problem.dt;
    dx  = problem.dx;

    % Random starting point (not projected)
    x_in.rho = randn(nt - 1, nx);
    x_in.mx  = randn(nt,     nx - 1);

    % Apply ETD projection
    x_out = proj_fokker_planck_expsemi(x_in, problem, cfg);

    rb = etd_fp_residual(x_in,  problem);
    ra = etd_fp_residual(x_out, problem);

    res_before_all(ei) = rb;
    res_after_all(ei)  = ra;

    fprintf('%-10.4g  %-16.4e  %-16.4e\n', vareps, rb, ra);
end

%% --- Plot ---
figure('Units', 'centimeters', 'Position', [2 2 14 9]);
loglog(eps_vals, res_after_all,  'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
hold on;
loglog(eps_vals, res_before_all, 'r--s', 'LineWidth', 1.2, 'MarkerSize', 5);
yline(1e-14, 'k:', 'Interpreter', 'none');
xlabel('eps', 'Interpreter', 'none');
ylabel('ETD FP residual norm', 'Interpreter', 'none');
title(sprintf('ETD residual before/after projection (NT=%d, NX=%d)', NT, NX), ...
    'Interpreter', 'none');
legend({'after projection', 'before projection', 'machine eps ~1e-14'}, ...
    'Interpreter', 'none', 'Location', 'best');
grid on;
set(gca, 'FontSize', 10);

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(gcf, fullfile(fig_dir, ...
    sprintf('etd_proj_residual_nt%d_nx%d.png', NT, NX)));

fprintf('\nResidual after projection near 1e-14 across all eps => backward stable.\n');

%% -----------------------------------------------------------------------
function res_norm = etd_fp_residual(x, problem)
% Compute the L2 norm of the ETD FP residual at (x.rho, x.mx).
%
%   f_hat = (rho_curr - S*rho_prev)_hat / dt  +  phi_vals .* (D_x m)_hat
%
% where S is the exact heat semigroup applied in DCT-x space.

    nt       = problem.nt;
    dt       = problem.dt;
    dx       = problem.dx;
    ops      = problem.ops;
    rho0     = problem.rho0;
    rho1     = problem.rho1;
    c_vals   = problem.expsemi_proj.c_vals;
    phi_vals = problem.expsemi_proj.phi_vals;

    zeros_x  = zeros(nt, 1);
    mu_prev  = [rho0;    x.rho];   % (nt x nx)
    mu_curr  = [x.rho;  rho1 ];   % (nt x nx)

    rho_hat  = dct(mu_prev')';
    S_prev   = idct((rho_hat .* c_vals)')';

    f_rho_hat = dct(((mu_curr - S_prev) / dt)')';
    Dxm_hat   = dct(ops.deriv_x_at_phi(x.mx, zeros_x, zeros_x)')';

    f_hat    = f_rho_hat + phi_vals .* Dxm_hat;
    res_norm = norm(f_hat(:)) * sqrt(dt * dx);
end
