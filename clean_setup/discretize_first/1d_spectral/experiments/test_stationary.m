% TEST_STATIONARY  Stationary problem: rho0 = rho1 (spectral ADMM).
%
%   When the marginals are identical the optimal solution is stationary
%   (rho constant in t, mx = 0).  Verifies that the solver recovers this.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
cfg      = cfg_admm_spectral_fe();
prob_def = prob_stationary();
problem  = setup_problem_spectral(cfg, prob_def);

%% --- Run ---
fprintf('Running %s  (nt=%d, nx=%d, eps=%.4g)...\n', cfg.name, cfg.nt, cfg.nx, cfg.vareps);
result = cfg.pipeline(cfg, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

ftag = sprintf('nt%d_nx%d_gam%g_eps%g', cfg.nt, cfg.nx, cfg.gamma, cfg.vareps);
nt = problem.nt;  nx = problem.nx;  xx = problem.xx;

%% --- Check: rho should be constant in t, mx should be ~0 ---
rho_std_t = std(result.rho, 0, 1);   % std over time for each x
mx_max    = max(abs(result.mx(:)));
fprintf('  max std(rho over t) = %.2e  (should be ~0)\n', max(rho_std_t));
fprintf('  max |mx|            = %.2e  (should be ~0)\n', mx_max);

%% --- Plot rho(t) at several x locations ---
x_idx = round(linspace(1, nx, 5));
t_cc  = ((1:nt)' - 0.5) * problem.dt;

figure('Position', [100 100 600 400]);
hold on;
colors = lines(numel(x_idx));
for p = 1:numel(x_idx)
    plot(t_cc, result.rho(:, x_idx(p)), '-', 'Color', colors(p,:), ...
         'LineWidth', 1.5, 'DisplayName', sprintf('x=%.2f', xx(x_idx(p))));
end
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\rho(t,x)$', 'Interpreter', 'latex');
title('Stationary test: $\rho$ vs $t$ at fixed $x$ (should be flat)', 'Interpreter', 'latex');
legend('Location', 'best', 'FontSize', 9);
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('stationary_%s.png', ftag)));

%% --- Residual ---
figure('Position', [720 100 600 300]);
semilogy(result.residual, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol=%.1e', cfg.tol));
xlabel('Iteration');  ylabel('Residual');
title(sprintf('ADMM residual   iters=%d', result.iters));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('residual_stationary_%s.png', ftag)));
