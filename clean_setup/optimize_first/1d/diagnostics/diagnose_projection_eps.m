%% diagnose_projection_eps.m
%
% Demonstrates that proj_fokker_planck is NOT a true (idempotent) projection
% for eps > 0, and quantifies how the violation grows with eps.
%
% ROOT CAUSE:
%   The projection is derived by inverting the *continuous* Fokker-Planck
%   operator using a DCT spectral solve. The DCT Laplacian is NOT the same
%   as the staggered-grid discrete Laplacian. For eps=0 the two agree (the
%   continuity equation involves only first-order operators which are
%   exactly skew-adjoint on the staggered grid). For eps>0 the mismatch in
%   the Laplacian term causes the projection to be inconsistent with the
%   discrete PDE, and the violation grows with eps.
%
% Plots generated:
%   1. Discrete FP residual norm  vs eps   (shows constraint violation)
%   2. Idempotency violation norm vs eps   (shows it is not a true projection)

clear; clc;

base = fileparts(mfilename('fullpath'));
addpath(fullfile(base, '..'));
addpath(fullfile(base, '..', 'config'));
addpath(fullfile(base, '..', 'problems'));
addpath(fullfile(base, '..', 'discretization'));
addpath(fullfile(base, '..', 'projection'));
addpath(fullfile(base, '..', 'utils'));

%% Setup
cfg      = cfg_staggered_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

nt  = problem.nt;   ntm = nt - 1;
nx  = problem.nx;   nxm = nx - 1;
ops = problem.ops;
rho0 = problem.rho0;  rho1 = problem.rho1;
zeros_x = zeros(nt, 1);

%% Build a fixed test input (same for all eps)
rng(42);
t_rho = (1:ntm)' * problem.dt;
x_in.rho = (1 - t_rho) .* rho0 + t_rho .* rho1 + 0.01*randn(ntm, nx);
x_in.mx  = 0.01 * randn(nt, nxm);

%% Sweep over eps values
eps_values = [0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1];
n_eps = length(eps_values);

fp_residual_norm   = zeros(n_eps, 1);
idempotency_norm   = zeros(n_eps, 1);

for k = 1:n_eps
    cfg_k        = cfg;
    cfg_k.vareps = eps_values(k);

    % First projection
    x_out  = proj_fokker_planck(x_in,  problem, cfg_k);

    % Second projection (idempotency check)
    x_out2 = proj_fokker_planck(x_out, problem, cfg_k);

    % Discrete FP residual of x_out using staggered-grid operators
    res = discrete_fp_residual(x_out.rho, x_out.mx, problem, eps_values(k));
    fp_residual_norm(k) = norm(res(:)) * sqrt(problem.dt * problem.dx);

    % Idempotency violation: how much does the second projection move x_out?
    drho = x_out2.rho - x_out.rho;
    dmx  = x_out2.mx  - x_out.mx;
    idempotency_norm(k) = norm([drho(:); dmx(:)]) * sqrt(problem.dt * problem.dx);
end

%% --- Plot 1: FP residual norm vs eps ---
figure('Name', 'PDE Residual vs eps');
loglog(eps_values(2:end), fp_residual_norm(2:end), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
hold on;
e = eps_values(2:end);
loglog(e, fp_residual_norm(2)*e/e(1),      'k--', 'LineWidth', 1.5);
xlabel('$\epsilon$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\|\partial_t\rho + \partial_x m - \epsilon \Delta_{\mathrm{disc}}\rho\|$', ...
       'Interpreter', 'latex', 'FontSize', 14);
title({'Discrete FP constraint violation after projection', ...
       '($\epsilon=0$ satisfies it exactly; violation grows with $\epsilon$)'}, ...
       'Interpreter', 'latex', 'FontSize', 13);
legend('FP residual norm', '$\mathcal{O}(\epsilon)$ reference', ...
       'Interpreter', 'latex', 'Location', 'northwest');
grid on;

%% --- Plot 2: Idempotency violation vs eps ---
figure('Name', 'Idempotency Violation vs eps');
semilogy(eps_values, idempotency_norm, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogy(eps_values(1), idempotency_norm(1), 'go', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('$\epsilon$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\|P_C(P_C(\rho,m)) - P_C(\rho,m)\|$', 'Interpreter', 'latex', 'FontSize', 14);
title({'Idempotency violation: $\|P_C(P_C(\rho,m)) - P_C(\rho,m)\|$', ...
       '(zero = true projection; grows with $\epsilon$)'}, ...
       'Interpreter', 'latex', 'FontSize', 13);
legend('violation', '$\epsilon=0$ (machine precision)', ...
       'Interpreter', 'latex', 'Location', 'southeast');
grid on;


%% --- Save figures ---
fig_dir = fullfile(base, '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

figHandles = findall(0, 'Type', 'figure');
for f = 1:length(figHandles)
    fname = strrep(figHandles(f).Name, ' ', '_');
    fname = strrep(fname, '(', '');
    fname = strrep(fname, ')', '');
    fname = strrep(fname, '=', '');
    fname = strrep(fname, '.', 'p');
    saveas(figHandles(f), fullfile(fig_dir, [fname '.png']));
end
fprintf('\nFigures saved to: %s\n', fig_dir);

% =========================================================================
%% Local functions
% =========================================================================

function res = discrete_fp_residual(rho_in, mx_in, problem, vareps)
% DISCRETE_FP_RESIDUAL  Evaluate the discrete Fokker-Planck residual:
%   d_t rho + d_x m - eps * d_xx rho
% using the staggered-grid operators throughout (consistent discretization).
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
