% TEST_SB_GAUSSIAN  Quick test: Schrödinger bridge between two Gaussians.
%
%   Runs discretize_then_optimize on the Gaussian problem, compares
%   density evolution and momentum against the analytical SB solution,
%   and plots the ADMM residual history.
%
%   Run setup_paths first, or call this script from the repo root after
%   adding the required folders to the MATLAB path.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

% Output directory for figures
fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
cfg      = cfg_ladmm_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

%% --- Run ---
fprintf('Running %s  (nt=%d, nx=%d, gamma=%.4g, tau=%.4g, eps=%.4g)...\n', ...
    cfg.name, cfg.nt, cfg.nx, cfg.gamma, cfg.tau, cfg.vareps);

result = cfg.pipeline(cfg, problem);

fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

% Shared filename tag encoding all key parameters
ftag = sprintf('nt%d_nx%d_gam%g_tau%g_eps%g', ...
    cfg.nt, cfg.nx, cfg.gamma, cfg.tau, cfg.vareps);

%% --- Analytical solution ---
[rho_ana_s, ~] = analytical_sb_gaussian(problem, cfg.vareps);

ops  = problem.ops;
rho0 = problem.rho0;
rho1 = problem.rho1;

% Analytical on cell-centres (for comparison with y = result.rho_cc)
rho_ana_cc = ops.interp_t_at_phi(rho_ana_s, rho0, rho1);   % (nt x nx)

% Numerical: use cell-centre result (y) for comparison, staggered (x) is FP-feasible
rho_num_cc = result.rho_cc;

%% --- Figure 1: density evolution (all times in one plot) ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
nt      = problem.nt;
nx      = problem.nx;
xx      = problem.xx;
dt      = problem.dt;

n_t    = numel(t_fracs);
colors = parula(n_t);   % one colour per time snapshot

figure('Name', sprintf('SB Gaussian density  gamma=%.4g tau=%.4g eps=%.4g', ...
    cfg.gamma, cfg.tau, cfg.vareps), 'Position', [100 100 700 400]);
hold on;

for p = 1:n_t
    k  = max(1, round(t_fracs(p) * nt));
    t_val = (k - 0.5) * dt;
    % Subsample markers so at most ~30 are shown regardless of grid size
    stride = max(1, floor(nx / 60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana_cc(k,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), rho_num_cc(k,idx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end

% Legend: one entry per time (solid = analytical, dashed = LADMM)
leg_str = cell(2*n_t, 1);
for p = 1:n_t
    k = max(1, round(t_fracs(p) * nt));
    t_val = (k - 0.5) * dt;
    leg_str{2*p-1} = sprintf('Ana   t=%.2f', t_val);
    leg_str{2*p}   = sprintf('LADMM t=%.2f', t_val);
end
legend(leg_str, 'Location', 'best', 'FontSize', 7);

xlabel('$x$',    'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('Density evolution   $\\gamma$=%.4g,  $\\tau$=%.4g,  $\\varepsilon$=%.4g', ...
    cfg.gamma, cfg.tau, cfg.vareps), 'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('density_%s.png', ftag)));

%% --- Figure 2: ADMM residual history ---
figure('Name', sprintf('Residual  gamma=%.4g tau=%.4g eps=%.4g', ...
    cfg.gamma, cfg.tau, cfg.vareps), 'Position', [100 550 600 300]);

semilogy(result.residual, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol = %.1e', cfg.tol));
xlabel('Iteration');
ylabel('$\|y^{k+1} - y^k\|$', 'Interpreter', 'latex');
title(sprintf('ADMM residual   iters=%d,  converged=%d', ...
    result.iters, result.converged));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('residual_%s.png', ftag)));

%% --- Figure 3: L2 error in rho vs analytical ---
% Compute per-time error on cell-centre grid
dx   = problem.dx;
err  = sqrt(dx * sum((rho_num_cc - rho_ana_cc).^2, 2));   % (nt x 1)
t_cc = ((1:nt)' - 0.5) * dt;

figure('Name', sprintf('L2 error  gamma=%.4g tau=%.4g eps=%.4g', ...
    cfg.gamma, cfg.tau, cfg.vareps), 'Position', [720 550 600 300]);

plot(t_cc, err, 'b-', 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\|\rho_\mathrm{LADMM} - \rho_\mathrm{ana}\|_{L^2(x)}$', 'Interpreter', 'latex');
title('$L^2$ error in $\rho$ vs analytical SB', 'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('l2error_%s.png', ftag)));
