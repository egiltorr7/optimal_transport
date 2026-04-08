%% test_proj_fokker_planck_banded.m
%
% Compares proj_fokker_planck (2D DCT, eps=0 only) against
% proj_fokker_planck_banded (DCT-x + tridiagonal-t, exact for all eps).
%
% Tests:
%   1. Both projections produce the same output for eps=0 (where DCT is exact)
%   2. Banded projection reduces the FP residual to machine precision for eps>0
%   3. Both are idempotent (projecting twice = projecting once)
%   4. Timing comparison
%
% Figures saved to: results/figures/compare_proj/

clear; clc;

base = fileparts(mfilename('fullpath'));
addpath(fullfile(base, '..'));
addpath(fullfile(base, '..', 'config'));
addpath(fullfile(base, '..', 'problems'));
addpath(fullfile(base, '..', 'discretization'));
addpath(fullfile(base, '..', 'projection'));
addpath(fullfile(base, '..', 'prox'));
addpath(fullfile(base, '..', 'pipelines'));
addpath(fullfile(base, '..', 'utils'));

fig_dir = fullfile(base, '..', 'results', 'figures', 'compare_proj');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% Setup: medium grid
cfg      = cfg_staggered_gaussian();
cfg.nx   = 64;
cfg.nt   = 128;
prob_def = prob_gaussian();

eps_vals = [0, 1e-4, 1e-3, 1e-2];
n_eps    = length(eps_vals);

err_vs_dct   = zeros(n_eps, 1);   % ||banded - DCT|| (only valid for eps=0)
res_dct      = zeros(n_eps, 1);   % FP residual after DCT projection
res_banded   = zeros(n_eps, 1);   % FP residual after banded projection
idem_dct     = zeros(n_eps, 1);   % idempotency error for DCT
idem_banded  = zeros(n_eps, 1);   % idempotency error for banded
time_dct     = zeros(n_eps, 1);
time_banded  = zeros(n_eps, 1);

n_apply = 20;   % repeat projections n_apply times for reliable timing

for ei = 1:n_eps
    vareps = eps_vals(ei);
    cfg.vareps = vareps;

    prob = setup_problem(cfg, prob_def);
    prob.banded_proj = precomp_banded_proj(prob, vareps);

    nt  = prob.nt;   ntm = nt - 1;
    nx  = prob.nx;   nxm = nx - 1;
    dt  = prob.dt;   dx  = prob.dx;

    % Random feasibility-violating input (not on the FP constraint set)
    rng(42);
    x_in.rho = rand(ntm, nx);
    x_in.mx  = rand(nt,  nxm);

    % --- DCT projection ---
    t0 = tic;
    for rep = 1:n_apply
        x_dct = proj_fokker_planck(x_in, prob, cfg);
    end
    time_dct(ei) = toc(t0) / n_apply;

    % --- Banded projection ---
    t0 = tic;
    for rep = 1:n_apply
        x_ban = proj_fokker_planck_banded(x_in, prob, cfg);
    end
    time_banded(ei) = toc(t0) / n_apply;

    % --- Agreement between methods (eps=0: both should be exact) ---
    err_vs_dct(ei) = sqrt(dt*dx) * norm([x_dct.rho(:) - x_ban.rho(:); ...
                                          x_dct.mx(:)  - x_ban.mx(:)]);

    % --- FP residual after projection ---
    res_dct(ei)    = fp_residual(x_dct, prob, cfg);
    res_banded(ei) = fp_residual(x_ban, prob, cfg);

    % --- Idempotency: apply each projection twice ---
    x_dct2 = proj_fokker_planck(x_dct, prob, cfg);
    x_ban2 = proj_fokker_planck_banded(x_ban, prob, cfg);

    idem_dct(ei)    = sqrt(dt*dx) * norm([x_dct2.rho(:) - x_dct.rho(:); ...
                                           x_dct2.mx(:)  - x_dct.mx(:)]);
    idem_banded(ei) = sqrt(dt*dx) * norm([x_ban2.rho(:) - x_ban.rho(:); ...
                                           x_ban2.mx(:)  - x_ban.mx(:)]);

    fprintf('eps=%.0e  |banded-DCT|=%.2e  res_dct=%.2e  res_banded=%.2e  idem_dct=%.2e  idem_banded=%.2e\n', ...
        vareps, err_vs_dct(ei), res_dct(ei), res_banded(ei), idem_dct(ei), idem_banded(ei));
end

%% -----------------------------------------------------------------------
%% Figure 1: FP residual after projection vs eps
%% -----------------------------------------------------------------------
figure('Name', 'FP residual after projection');
semilogy(1:n_eps, res_dct,    'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
semilogy(1:n_eps, res_banded, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
set(gca, 'XTick', 1:n_eps, 'XTickLabel', arrayfun(@(e) sprintf('%.0e',e), eps_vals, 'UniformOutput', false));
xlabel('$\varepsilon$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\|A(\rho,m)\|$', 'Interpreter', 'latex', 'FontSize', 14);
title('FP residual after projection', 'Interpreter', 'latex', 'FontSize', 13);
legend('DCT (2D separable)', 'Banded (DCT-x + tridiag-t)', ...
       'Interpreter', 'latex', 'Location', 'northwest');
grid on;

%% -----------------------------------------------------------------------
%% Figure 2: Idempotency error vs eps
%% -----------------------------------------------------------------------
figure('Name', 'Idempotency error');
semilogy(1:n_eps, idem_dct,    'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
semilogy(1:n_eps, idem_banded, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
set(gca, 'XTick', 1:n_eps, 'XTickLabel', arrayfun(@(e) sprintf('%.0e',e), eps_vals, 'UniformOutput', false));
xlabel('$\varepsilon$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\|P^2 u - Pu\|$', 'Interpreter', 'latex', 'FontSize', 14);
title('Idempotency: $\|P(Pu) - Pu\|$', 'Interpreter', 'latex', 'FontSize', 13);
legend('DCT (2D separable)', 'Banded (DCT-x + tridiag-t)', ...
       'Interpreter', 'latex', 'Location', 'northwest');
grid on;

%% -----------------------------------------------------------------------
%% Figure 3: Timing comparison
%% -----------------------------------------------------------------------
figure('Name', 'Timing comparison');
bar(1:n_eps, [time_dct, time_banded]*1e3);
set(gca, 'XTick', 1:n_eps, 'XTickLabel', arrayfun(@(e) sprintf('%.0e',e), eps_vals, 'UniformOutput', false));
xlabel('$\varepsilon$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Time per call (ms)', 'Interpreter', 'latex', 'FontSize', 14);
title(sprintf('Projection timing  ($n_t=%d,\\, n_x=%d$, avg over %d calls)', ...
    cfg.nt, cfg.nx, n_apply), 'Interpreter', 'latex', 'FontSize', 13);
legend('DCT (2D separable)', 'Banded (DCT-x + tridiag-t)', ...
       'Interpreter', 'latex', 'Location', 'northwest');
grid on;

%% Save
figHandles = findall(0, 'Type', 'figure');
for f = 1:length(figHandles)
    fname = strrep(figHandles(f).Name, ' ', '_');
    saveas(figHandles(f), fullfile(fig_dir, [fname '.png']));
end
fprintf('\nFigures saved to: %s\n', fig_dir);

%% -----------------------------------------------------------------------
%% Helper: FP residual norm
%% -----------------------------------------------------------------------
function res = fp_residual(x, problem, cfg)
    ops     = problem.ops;
    rho0    = problem.rho0;
    rho1    = problem.rho1;
    vareps  = cfg.vareps;
    nt      = problem.nt;
    zeros_x = zeros(nt, 1);

    lap_rho = ops.deriv_x_at_phi( ...
                  ops.deriv_x_at_m( ...
                      ops.interp_t_at_phi(x.rho, rho0, rho1)), ...
                  zeros_x, zeros_x);

    f = ops.deriv_t_at_phi(x.rho, rho0, rho1) ...
      + ops.deriv_x_at_phi(x.mx, zeros_x, zeros_x) ...
      - vareps * lap_rho;

    res = sqrt(problem.dt * problem.dx) * norm(f(:));
end
