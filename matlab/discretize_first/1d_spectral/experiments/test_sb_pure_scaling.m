% TEST_SB_PURE_SCALING  SB with pure variance scaling (spectral ADMM).
%
%   rho0 and rho1 are Gaussians with the same mean but different variances.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
cfg      = cfg_admm_spectral_fe();
prob_def = prob_gaussian_pure_scaling();
problem  = setup_problem_spectral(cfg, prob_def);

%% --- Run ---
fprintf('Running %s  (nt=%d, nx=%d, eps=%.4g)...\n', cfg.name, cfg.nt, cfg.nx, cfg.vareps);
result = cfg.pipeline(cfg, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

ftag = sprintf('nt%d_nx%d_gam%g_eps%g', cfg.nt, cfg.nx, cfg.gamma, cfg.vareps);

%% --- Density evolution ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
nt = problem.nt;  nx = problem.nx;  xx = problem.xx;  dt = problem.dt;
n_t = numel(t_fracs);  colors = parula(n_t);

figure('Name', 'SB pure scaling', 'Position', [100 100 700 400]);
hold on;
for p = 1:n_t
    k      = max(1, round(t_fracs(p) * nt));
    stride = max(1, floor(nx / 60));  idx = 1:stride:nx;
    plot(xx(idx), result.rho(k,idx), 'o-', 'Color', colors(p,:), ...
        'MarkerSize', 4, 'LineWidth', 1.2, ...
        'DisplayName', sprintf('t=%.2f', (k-0.5)*dt));
end
legend('Location', 'best', 'FontSize', 8);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('SB pure scaling   $\\varepsilon=%.4g$', cfg.vareps), 'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('density_sb_pure_scaling_%s.png', ftag)));

%% --- Residual ---
figure('Position', [720 100 600 300]);
semilogy(result.residual, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol=%.1e', cfg.tol));
xlabel('Iteration');  ylabel('Residual');
title(sprintf('ADMM residual   iters=%d', result.iters));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('residual_sb_pure_scaling_%s.png', ftag)));
