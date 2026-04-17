% TEST_SB_GAUSSIAN  Quick test: 2D Schrödinger bridge between two Gaussians.
%
%   Runs sinkhorn_hopf_cole on the 2D Gaussian problem, compares density
%   evolution against the analytical SB solution, and plots:
%     Figure 1: density snapshots at several times (Sinkhorn vs analytical)
%     Figure 2: Sinkhorn convergence history
%     Figure 3: L2 error in rho vs analytical over time
%
%   Run setup_paths first, or call from the repo root after adding
%   the required folders to the MATLAB path.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

% Output directory for figures
fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
cfg      = cfg_sinkhorn_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

%% --- Run ---
fprintf('Running sinkhorn_hopf_cole  (nt=%d, nx=%d, ny=%d, eps=%.4g, kernel=%s)...\n', ...
    cfg.nt, cfg.nx, cfg.ny, cfg.vareps, func2str(cfg.precomp_heat));

result = sinkhorn_hopf_cole(problem, cfg);

fprintf('  kernel=%s,  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.heat_name, result.iters, result.converged, result.error, result.walltime);

% Filename tag encoding key parameters
ftag = sprintf('nt%d_nx%d_eps%g_%s', cfg.nt, cfg.nx, cfg.vareps, result.heat_name);

nt = problem.nt;   ntm = nt - 1;
nx = problem.nx;
ny = problem.ny;
dt = problem.dt;
dx = problem.dx;
dy = problem.dy;
xx = problem.xx;   % (nx x 1)
yy = problem.yy;   % (1  x ny)

%% --- Analytical solution ---
% analytical_sb_gaussian returns PDF (sum*dx*dy = 1) on interior times 1..ntm
[rho_ana, ~, ~] = analytical_sb_gaussian(problem, cfg.vareps);   % (ntm x nx x ny)

% Sinkhorn result: rho_traj(k+1,:,:) at t=k*dt.
% Convert Sinkhorn output to PDF (sum=1 mass -> divide by dx*dy) for fair comparison.
use_pdf = isfield(cfg, 'use_pdf_marginals') && cfg.use_pdf_marginals;

% Safe heat_name for LaTeX titles
heat_name_tex = strrep(result.heat_name, '_', ' ');

%% --- Figure 1: density snapshots ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
n_t     = numel(t_fracs);

fig1 = figure('Name', sprintf('SB Gaussian 2D density  eps=%.4g  kernel=%s', ...
    cfg.vareps, result.heat_name), 'Position', [50 50 1400 500]);

for p = 1:n_t
    k = max(1, round(t_fracs(p) * nt));   % time index into result.rho (1..nt)
    t_k = k * dt;

    % Sinkhorn density: convert mass -> PDF
    rho_num_k = squeeze(result.rho(k+1, :, :)) / (dx * dy);   % (nx x ny) PDF

    % Analytical density at inner time k (analytical is on k=1..ntm, so shift index)
    k_ana = min(k, ntm);
    rho_ana_k = squeeze(rho_ana(k_ana, :, :));                  % (nx x ny) PDF

    % Clamp to same color scale for fair comparison
    clim_max = max(max(rho_ana_k(:)), max(rho_num_k(:)));

    subplot(2, n_t, p);
    imagesc(yy(:), xx(:), rho_ana_k);
    clim([0, clim_max]);
    colorbar; axis xy equal tight;
    title(sprintf('Analytical  t=%.2f', t_k));
    xlabel('y'); ylabel('x');

    subplot(2, n_t, n_t + p);
    imagesc(yy(:), xx(:), rho_num_k);
    clim([0, clim_max]);
    colorbar; axis xy equal tight;
    title(sprintf('Sinkhorn  t=%.2f', t_k));
    xlabel('y'); ylabel('x');
end
sgtitle(sprintf('Density evolution   eps=%.4g,  kernel: %s', cfg.vareps, heat_name_tex));
saveas(fig1, fullfile(fig_dir, sprintf('density_%s.png', ftag)));

%% --- Figure 2: Sinkhorn convergence ---
fig2 = figure('Name', sprintf('Sinkhorn convergence  eps=%.4g', cfg.vareps), ...
    'Position', [100 600 600 300]);

semilogy(result.errors, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol = %.1e', cfg.tol));
xlabel('Sinkhorn iteration');
ylabel('Left-marginal L^2 error');
title(sprintf('Sinkhorn convergence   iters=%d,  converged=%d', ...
    result.iters, result.converged));
grid on;
saveas(fig2, fullfile(fig_dir, sprintf('convergence_%s.png', ftag)));

%% --- Figure 3: L2 error in rho vs analytical over time ---
% Compare at inner times k=1..ntm
t_inner = (1:ntm)' * dt;
err_t   = zeros(ntm, 1);

for k = 1:ntm
    rho_num_k = squeeze(result.rho(k+1, :, :)) / (dx * dy);   % PDF
    rho_ana_k = squeeze(rho_ana(k,   :, :));                    % PDF
    err_t(k)  = sqrt(dx * dy) * norm(rho_num_k(:) - rho_ana_k(:));
end

fig3 = figure('Name', sprintf('L2 error  eps=%.4g  kernel=%s', cfg.vareps, result.heat_name), ...
    'Position', [750 600 600 300]);

plot(t_inner, err_t, 'b-', 'LineWidth', 1.5);
xlabel('t');
ylabel('||rho_num - rho_ana||_{L^2(x,y)}');
title(sprintf('L^2 error vs Analytical   eps=%.4g,  kernel: %s', cfg.vareps, heat_name_tex));
grid on;
saveas(fig3, fullfile(fig_dir, sprintf('l2error_%s.png', ftag)));
