% SWEEP_EPS  Epsilon sweep for Gaussian-to-Gaussian SB (spectral ADMM).
%
% Sweeps vareps = [0, 1e-3, 1e-2, 1e-1, 1] on a fixed grid.
%
% Plots:
%   1. Density evolution at selected times for each eps
%   2. Kinetic energy vs eps vs analytical SB
%   3. ADMM residual history for each eps

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% --- Parameters ---
eps_vals = [0, 1e-3, 1e-2, 1e-1, 1];
n_eps    = numel(eps_vals);

cfg_base = cfg_admm_spectral_fe();
prob_def = prob_gaussian();

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
ftag = sprintf('sweep_eps_nt%d_nx%d_gam%g', cfg_base.nt, cfg_base.nx, cfg_base.gamma);

%% --- Run (rebuild problem per eps: precomp changes with vareps) ---
results  = cell(n_eps, 1);
rho_ana  = cell(n_eps, 1);
ke_admm  = zeros(n_eps, 1);
ke_ana   = zeros(n_eps, 1);

for k = 1:n_eps
    cfg_k        = cfg_base;
    cfg_k.vareps = eps_vals(k);
    problem_k    = setup_problem_spectral(cfg_k, prob_def);

    fprintf('[%d/%d]  eps=%.1e ... ', k, n_eps, eps_vals(k));
    r = cfg_k.pipeline(cfg_k, problem_k);
    fprintf('iters=%d  error=%.2e  time=%.1fs\n', r.iters, r.error, r.walltime);

    [rho_a, mx_a] = analytical_sb_gaussian_spectral(problem_k, eps_vals(k));
    ke_admm(k) = compute_objective_spectral(r.rho, r.mx, problem_k) / problem_k.dx;
    ke_ana(k)  = compute_objective_spectral(rho_a, mx_a, problem_k) / problem_k.dx;

    results{k} = r;
    rho_ana{k} = rho_a;
end

% Use the last problem struct for grid info (same grid for all eps)
problem = problem_k;

%% --- Plot 1: density evolution per eps ---
t_fracs  = [0.1, 0.25, 0.5, 0.75, 0.9];
colors_t = lines(numel(t_fracs));
nt = problem.nt;  nx = problem.nx;  xx = problem.xx;  dt = problem.dt;
stride = max(1, floor(nx / 40));  idx = 1:stride:nx;

ncols = ceil((n_eps + 1) / 2);
figure('Position', [100 100 300*ncols 500]);
for k = 1:n_eps
    subplot(2, ncols, k);
    hold on;
    for j = 1:numel(t_fracs)
        ti = max(1, round(t_fracs(j) * nt) + 1);   % edge index
        plot(xx, rho_ana{k}(ti,:), '-', 'Color', colors_t(j,:), 'LineWidth', 1.5);
        plot(xx(idx), results{k}.rho(ti,idx), 'o', 'Color', colors_t(j,:), ...
             'MarkerSize', 5, 'MarkerFaceColor', colors_t(j,:));
    end
    title(sprintf('$\\varepsilon = %.1g$', eps_vals(k)), 'Interpreter', 'latex');
    xlabel('$x$', 'Interpreter', 'latex');
    if mod(k-1, ncols) == 0, ylabel('$\rho$', 'Interpreter', 'latex'); end
    grid on;
end

ax_leg = subplot(2, ncols, n_eps + 1);
hold on; axis off;
leg = {};
for j = 1:numel(t_fracs)
    plot(nan, nan, '-',  'Color', colors_t(j,:), 'LineWidth', 1.5);
    plot(nan, nan, 'o',  'Color', colors_t(j,:), 'MarkerFaceColor', colors_t(j,:), 'MarkerSize', 5);
    leg{end+1} = sprintf('Ana $t=%.2f$',      t_fracs(j)); %#ok<AGROW>
    leg{end+1} = sprintf('Spectral $t=%.2f$', t_fracs(j)); %#ok<AGROW>
end
legend(leg, 'Interpreter', 'latex', 'FontSize', 8, 'Location', 'best', 'NumColumns', 2);
saveas(gcf, fullfile(fig_dir, sprintf('density_%s.png', ftag)));

%% --- Plot 2: KE vs eps ---
eps_plot = max(eps_vals, 1e-6);
figure('Position', [100 600 600 350]);
semilogx(eps_plot, ke_admm, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
semilogx(eps_plot, ke_ana,  'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('$\varepsilon$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$\iint m^2/\rho\,dt\,dx$', 'Interpreter', 'latex', 'FontSize', 13);
title(sprintf('Kinetic energy vs $\\varepsilon$   $n_t=%d,\\, n_x=%d,\\, \\gamma=%g$', ...
    cfg_base.nt, cfg_base.nx, cfg_base.gamma), 'Interpreter', 'latex');
legend('Spectral ADMM', 'Analytical SB', 'Interpreter', 'latex', 'Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ke_%s.png', ftag)));

%% --- Plot 3: residual history ---
colors_e = lines(n_eps);
figure('Position', [720 600 600 350]);
hold on;
for k = 1:n_eps
    semilogy(results{k}.residual, '-', 'Color', colors_e(k,:), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('$\\varepsilon=%.1g$', eps_vals(k)));
end
yline(cfg_base.tol, 'k--', sprintf('tol=%.0e', cfg_base.tol));
xlabel('Iteration');
ylabel('$\|y^{k+1} - y^k\|$', 'Interpreter', 'latex');
title('ADMM residual history');
legend('Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 9);
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('residual_%s.png', ftag)));

%% --- Summary ---
fprintf('\n%-10s  %-8s  %-12s  %-12s\n', 'eps', 'iters', 'KE (spec)', 'KE (ana)');
fprintf('%s\n', repmat('-', 1, 48));
for k = 1:n_eps
    fprintf('%-10.1e  %-8d  %-12.6f  %-12.6f\n', ...
        eps_vals(k), results{k}.iters, ke_admm(k), ke_ana(k));
end
