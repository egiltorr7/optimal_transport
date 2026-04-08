% TEST_SB_GAUSSIAN  Schrödinger bridge between two Gaussians (spectral ADMM).
%
%   Runs discretize_then_optimize with the spectral FE config, compares
%   density evolution and momentum against the analytical SB solution,
%   and plots the ADMM residual history.
%
%   Swap cfg_admm_spectral_fe() for cfg_admm_spectral_be() to test the
%   backward-Euler variant.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
% cfg      = cfg_admm_spectral_fe();
cfg      = cfg_admm_spectral_be();
prob_def = prob_gaussian();
problem  = setup_problem_spectral(cfg, prob_def);

%% --- Run ---
fprintf('Running %s  (nt=%d, nx=%d, gamma=%.4g, eps=%.4g)...\n', ...
    cfg.name, cfg.nt, cfg.nx, cfg.gamma, cfg.vareps);

result = cfg.pipeline(cfg, problem);

fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

ftag = sprintf('nt%d_nx%d_gam%g_eps%g', cfg.nt, cfg.nx, cfg.gamma, cfg.vareps);

%% --- Analytical solution (edge grid) ---
[rho_ana, ~] = analytical_sb_gaussian_spectral(problem, cfg.vareps);

rho_num = result.rho;

%% --- Figure 1: density evolution ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
nt      = problem.nt;
nx      = problem.nx;
xx      = problem.xx;
dt      = problem.dt;

n_t    = numel(t_fracs);
colors = parula(n_t);

figure('Name', sprintf('SB Gaussian density  gamma=%.4g eps=%.4g', cfg.gamma, cfg.vareps), ...
       'Position', [100 100 700 400]);
hold on;

for p = 1:n_t
    k      = max(1, round(t_fracs(p) * nt) + 1);   % +1: edge index, t = (k-1)*dt
    stride = max(1, floor(nx / 60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana(k,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), rho_num(k,idx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end

leg_str = cell(2*n_t, 1);
for p = 1:n_t
    k = max(1, round(t_fracs(p) * nt) + 1);
    leg_str{2*p-1} = sprintf('Ana      t=%.2f', (k-1)*dt);
    leg_str{2*p}   = sprintf('Spectral t=%.2f', (k-1)*dt);
end
legend(leg_str, 'Location', 'best', 'FontSize', 7);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('Density evolution   $\\gamma$=%.4g,  $\\varepsilon$=%.4g', ...
    cfg.gamma, cfg.vareps), 'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('density_%s.png', ftag)));

%% --- Figure 2: ADMM residual ---
figure('Position', [100 550 600 300]);
semilogy(result.residual, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol = %.1e', cfg.tol));
xlabel('Iteration');
ylabel('$\|y^{k+1} - y^k\|$', 'Interpreter', 'latex');
title(sprintf('ADMM residual   iters=%d,  converged=%d', result.iters, result.converged));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('residual_%s.png', ftag)));

%% --- Figure 3: L2 error in rho ---
dx      = problem.dx;
err     = sqrt(dx * sum((rho_num - rho_ana).^2, 2));   % (nt+1 x 1)
t_edges = (0:nt)' * dt;

figure('Position', [720 550 600 300]);
plot(t_edges, err, 'b-', 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\|\rho_\mathrm{spec} - \rho_\mathrm{ana}\|_{L^2(x)}$', 'Interpreter', 'latex');
title('$L^2$ error in $\rho$ vs analytical SB', 'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('l2error_%s.png', ftag)));
