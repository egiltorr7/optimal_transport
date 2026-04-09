% TEST_SB_GAUSSIAN  Schrödinger bridge between two 2D Gaussians.
%
%   Runs discretize_then_optimize on the 2D Gaussian problem, compares
%   the density slices and x-momentum against the analytical SB solution,
%   and plots the ADMM residual history.
%
%   Run setup_paths first, or call this script from experiments/.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
cfg      = cfg_ladmm_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

nt = problem.nt;  ntm = nt - 1;
nx = problem.nx;  dt  = problem.dt;
ny = problem.ny;  dx  = problem.dx;  dy = problem.dy;
xx = problem.xx;  yy  = problem.yy;

%% --- Run ---
fprintf('Running %s  (nt=%d, nx=%d, ny=%d, eps=%.4g)...\n', ...
    cfg.name, cfg.nt, cfg.nx, cfg.ny, cfg.vareps);

result = cfg.pipeline(cfg, problem);

fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

ftag = sprintf('nt%d_nx%d_ny%d_gam%g_tau%g_eps%g', ...
    cfg.nt, cfg.nx, cfg.ny, cfg.gamma, cfg.tau, cfg.vareps);

%% --- Analytical solution ---
[rho_ana, mx_ana, ~] = analytical_sb_gaussian(problem, cfg.vareps);

ops  = problem.ops;
rho0 = problem.rho0;
rho1 = problem.rho1;

% Interpolate analytical rho to cell-centres for comparison with y = result.rho_cc
rho_ana_cc = ops.interp_t_at_phi(rho_ana, rho0, rho1);   % (nt x nx x ny)
rho_num_cc = result.rho_cc;                                % (nt x nx x ny)

%% --- Figure 1: density heatmaps at selected times ---
t_fracs = [0.1, 0.3, 0.5, 0.7, 0.9];
n_t     = numel(t_fracs);

figure('Name', 'SB Gaussian density snapshots', 'Position', [50 50 1200 500]);
for p = 1:n_t
    k = max(1, round(t_fracs(p) * nt));

    % Analytical (x-slice averaged over y since solution is uniform in y)
    subplot(2, n_t, p);
    imagesc(xx, yy, squeeze(rho_ana_cc(k,:,:))');
    axis xy; colorbar; clim([0, max(rho_ana_cc(:))]);
    title(sprintf('Ana  t=%.2f', (k-0.5)*dt));
    xlabel('x'); ylabel('y');

    % Numerical
    subplot(2, n_t, n_t + p);
    imagesc(xx, yy, squeeze(rho_num_cc(k,:,:))');
    axis xy; colorbar; clim([0, max(rho_ana_cc(:))]);
    title(sprintf('LADMM  t=%.2f', (k-0.5)*dt));
    xlabel('x'); ylabel('y');
end
sgtitle(sprintf('Density  gamma=%.4g  tau=%.4g  eps=%.4g', ...
    cfg.gamma, cfg.tau, cfg.vareps));
saveas(gcf, fullfile(fig_dir, sprintf('density_%s.png', ftag)));

%% --- Figure 2: x-slice of rho at y = 0.5 ---
% Compare 1D cross-section through the middle
[~, iy] = min(abs(yy - 0.5));

t_fracs2 = [0.1, 0.25, 0.5, 0.75, 0.9];
n_t2     = numel(t_fracs2);
colors   = parula(n_t2);

figure('Name', 'SB Gaussian x-slice at y=0.5', 'Position', [50 600 700 400]);
hold on;
leg_handles = gobjects(n_t2, 2);
for p = 1:n_t2
    k = max(1, round(t_fracs2(p) * nt));
    t_val = (k - 0.5) * dt;
    leg_handles(p,1) = plot(xx, rho_ana_cc(k,:,iy), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    leg_handles(p,2) = plot(xx, rho_num_cc(k,:,iy), 'o--','Color', colors(p,:), 'MarkerSize', 4);
end

% One legend entry per time: solid = analytical, dashed+markers = LADMM
leg_str = cell(n_t2, 2);
for p = 1:n_t2
    k = max(1, round(t_fracs2(p) * nt));
    t_val = (k - 0.5) * dt;
    leg_str{p,1} = sprintf('Ana   t=%.2f', t_val);
    leg_str{p,2} = sprintf('LADMM t=%.2f', t_val);
end
legend(leg_handles(:), leg_str(:), 'Location', 'best', 'FontSize', 7);

xlabel('x'); ylabel('\rho(t,x,y=0.5)');
title(sprintf('x-slice at y=0.5   eps=%.4g', cfg.vareps));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('xslice_%s.png', ftag)));

%% --- Figure 3: ADMM residual ---
figure('Name', 'ADMM residual', 'Position', [800 600 600 300]);
semilogy(result.residual, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol = %.1e', cfg.tol));
xlabel('Iteration');
ylabel('Residual');
title(sprintf('ADMM residual   iters=%d,  converged=%d', result.iters, result.converged));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('residual_%s.png', ftag)));

%% --- Figure 4: L2 error in rho vs analytical (averaged over y) ---
err = sqrt(dx * dy * sum(sum((rho_num_cc - rho_ana_cc).^2, 2), 3));   % (nt x 1)
t_cc = ((1:nt)' - 0.5) * dt;

figure('Name', 'L2 error', 'Position', [800 100 600 300]);
plot(t_cc, err, 'b-', 'LineWidth', 1.5);
xlabel('t');
ylabel('||\rho_{LADMM} - \rho_{ana}||_{L^2(x,y)}');
title('L^2 error in \rho vs analytical SB');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('l2error_%s.png', ftag)));

%% --- Kinetic energy ---
obj = compute_objective(result.rho_stag, result.mx_stag, result.my_stag, problem);
fprintf('Kinetic energy (staggered): %.6f\n', obj);
