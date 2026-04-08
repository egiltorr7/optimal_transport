% SWEEP_GRID  Grid-refinement study (spectral ADMM).
%
% Sweeps (nt, nx) on a fixed eps and gamma, plotting L2 error in rho
% vs analytical SB to assess convergence order.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% --- Parameters ---
grid_vals = [16, 32, 64, 128];    % nt and nx values (equal refinement)
n_grid    = numel(grid_vals);

cfg_base = cfg_admm_spectral_fe();
prob_def = prob_gaussian();

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
ftag = sprintf('sweep_grid_gam%g_eps%g', cfg_base.gamma, cfg_base.vareps);

%% --- Run ---
results   = cell(n_grid, 1);
l2_errors = zeros(n_grid, 1);
dts       = zeros(n_grid, 1);

for k = 1:n_grid
    cfg_k    = cfg_base;
    cfg_k.nt = grid_vals(k);
    cfg_k.nx = grid_vals(k);
    problem_k = setup_problem_spectral(cfg_k, prob_def);

    fprintf('[%d/%d]  nt=nx=%d ... ', k, n_grid, grid_vals(k));
    r = cfg_k.pipeline(cfg_k, problem_k);
    fprintf('iters=%d  error=%.2e  time=%.1fs\n', r.iters, r.error, r.walltime);

    [rho_a, ~] = analytical_sb_gaussian_spectral(problem_k, cfg_k.vareps);
    l2_errors(k) = sqrt(problem_k.dx * mean( ...
        sum((r.rho - rho_a).^2, 2)));    % time-averaged L2 error in rho
    dts(k)       = problem_k.dt;
    results{k}   = r;
end

%% --- Plot 1: L2 error vs grid spacing ---
figure('Position', [100 100 600 400]);
loglog(dts, l2_errors, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;

% Reference slopes
h_ref = dts([1 end]);
loglog(h_ref, l2_errors(1) * (h_ref/dts(1)).^1, 'k--', 'DisplayName', 'slope 1');
loglog(h_ref, l2_errors(1) * (h_ref/dts(1)).^2, 'k:',  'DisplayName', 'slope 2');

xlabel('$\Delta t = \Delta x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$\|\rho - \rho_\mathrm{ana}\|_{L^2}$', 'Interpreter', 'latex', 'FontSize', 13);
title(sprintf('Grid convergence   $\\gamma=%g,\\; \\varepsilon=%g$', ...
    cfg_base.gamma, cfg_base.vareps), 'Interpreter', 'latex');
legend('Spectral ADMM', 'slope 1', 'slope 2', 'Interpreter', 'latex', 'Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('convergence_%s.png', ftag)));

%% --- Summary ---
fprintf('\n%-8s  %-8s  %-12s\n', 'nt=nx', 'iters', 'L2 error');
fprintf('%s\n', repmat('-', 1, 32));
for k = 1:n_grid
    fprintf('%-8d  %-8d  %-12.4e\n', grid_vals(k), results{k}.iters, l2_errors(k));
end
