% SWEEP_GAMMA  Sweep over penalty parameter gamma for Gaussian-to-Gaussian SB.
%
% For each gamma, sets tau = gamma + 1 (minimal safe margin above convergence threshold).
%
% Plots:
%   1. Density evolution at selected times for each gamma
%   2. Kinetic energy vs gamma vs analytical SB
%   3. ADMM residual history for each gamma

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% --- Parameters ---
gamma_vals = [1, 100, 500, 1000];
n_gamma    = numel(gamma_vals);

cfg_base = cfg_ladmm_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg_base, prob_def);   % fixed grid for all gamma

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
ftag = sprintf('sweep_gamma_nt%d_nx%d_eps%g', cfg_base.nt, cfg_base.nx, cfg_base.vareps);

%% --- Analytical solution (same for all gamma) ---
[rho_a, mx_a] = analytical_sb_gaussian(problem, cfg_base.vareps);
ke_ana_val    = compute_objective(rho_a, mx_a, problem) / problem.dx;

%% --- Run ---
results = cell(n_gamma, 1);
ke_admm = zeros(n_gamma, 1);

for k = 1:n_gamma
    cfg_k       = cfg_base;
    cfg_k.gamma = gamma_vals(k);
    cfg_k.tau   = gamma_vals(k) + 1;

    fprintf('[%d/%d]  gamma=%g  tau=%g ... ', k, n_gamma, cfg_k.gamma, cfg_k.tau);
    r = cfg_k.pipeline(cfg_k, problem);
    fprintf('iters=%d  error=%.2e  time=%.1fs\n', r.iters, r.error, r.walltime);

    ke_admm(k) = compute_objective(r.rho_stag, r.mx_stag, problem) / problem.dx;
    results{k} = r;
end

%% -----------------------------------------------------------------------
%% Plot 1: Density evolution per gamma value
%% -----------------------------------------------------------------------
t_fracs  = [0.1, 0.25, 0.5, 0.75, 0.9];
colors_t = lines(numel(t_fracs));
ntm      = problem.nt - 1;
nx       = problem.nx;
xx       = problem.xx;
dt       = problem.dt;
stride   = max(1, floor(nx / 40));
idx      = 1:stride:nx;

ncols = ceil((n_gamma + 1) / 2);
figure('Position', [100 100 300*ncols 500]);
for k = 1:n_gamma
    subplot(2, ncols, k);
    hold on;
    for j = 1:numel(t_fracs)
        ti = max(1, min(ntm, round(t_fracs(j) / dt)));
        plot(xx, rho_a(ti,:), '-', 'Color', colors_t(j,:), 'LineWidth', 1.5);
        plot(xx(idx), results{k}.rho_stag(ti,idx), 'o', 'Color', colors_t(j,:), ...
             'MarkerSize', 5, 'MarkerFaceColor', colors_t(j,:));
    end
    title(sprintf('$\\gamma=%g,\\; \\tau=%g$', gamma_vals(k), gamma_vals(k)+1), ...
          'Interpreter', 'latex');
    xlabel('$x$', 'Interpreter', 'latex');
    if mod(k-1, ncols) == 0
        ylabel('$\rho$', 'Interpreter', 'latex');
    end
    grid on;
end

ax_leg = subplot(2, ncols, n_gamma + 1);
hold on; axis off;
leg = {};
for j = 1:numel(t_fracs)
    plot(nan, nan, '-',  'Color', colors_t(j,:), 'LineWidth', 1.5);
    plot(nan, nan, 'o',  'Color', colors_t(j,:), 'MarkerFaceColor', colors_t(j,:), 'MarkerSize', 5);
    leg{end+1} = sprintf('Ana $t=%.2f$',   t_fracs(j)); %#ok<AGROW>
    leg{end+1} = sprintf('LADMM $t=%.2f$', t_fracs(j)); %#ok<AGROW>
end
legend(leg, 'Interpreter', 'latex', 'FontSize', 8, 'Location', 'best', 'NumColumns', 2);
saveas(gcf, fullfile(fig_dir, sprintf('density_%s.png', ftag)));

%% -----------------------------------------------------------------------
%% Plot 2: Kinetic energy vs gamma
%% -----------------------------------------------------------------------
figure('Position', [100 600 600 350]);
semilogx(gamma_vals, ke_admm, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
yline(ke_ana_val, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Analytical SB');
xlabel('$\gamma$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$\iint m^2/\rho\,dt\,dx$', 'Interpreter', 'latex', 'FontSize', 13);
title(sprintf('Kinetic energy vs $\\gamma$   $n_t=%d,\\, n_x=%d,\\, \\varepsilon=%g$', ...
    cfg_base.nt, cfg_base.nx, cfg_base.vareps), 'Interpreter', 'latex');
legend('LADMM  ($\tau = \gamma+1$)', 'Analytical SB', 'Interpreter', 'latex', 'Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ke_%s.png', ftag)));

%% -----------------------------------------------------------------------
%% Plot 3: ADMM residual history per gamma
%% -----------------------------------------------------------------------
colors_g = lines(n_gamma);

figure('Position', [720 600 600 350]);
hold on;
for k = 1:n_gamma
    semilogy(results{k}.residual, '-', 'Color', colors_g(k,:), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('$\\gamma=%g$', gamma_vals(k)));
end
yline(cfg_base.tol, 'k--', sprintf('tol=%.0e', cfg_base.tol));
xlabel('Iteration');
ylabel('$\|y^{k+1} - y^k\|$', 'Interpreter', 'latex');
title(sprintf('ADMM residual   $\\varepsilon=%g,\\; \\tau=\\gamma+1$', cfg_base.vareps), ...
    'Interpreter', 'latex');
legend('Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 9);
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('residual_%s.png', ftag)));

%% --- Summary table ---
fprintf('\n%-10s  %-10s  %-8s  %-12s  %-12s\n', 'gamma', 'tau', 'iters', 'KE (LADMM)', 'KE (ana)');
fprintf('%s\n', repmat('-', 1, 58));
for k = 1:n_gamma
    fprintf('%-10g  %-10g  %-8d  %-12.6f  %-12.6f\n', ...
        gamma_vals(k), gamma_vals(k)+1, results{k}.iters, ke_admm(k), ke_ana_val);
end
