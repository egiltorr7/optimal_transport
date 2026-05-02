% TEST_SB_GAUSSIAN  Schrödinger bridge between two 2D Gaussians.
%
%   Runs discretize_then_optimize on the 2D Gaussian problem.
%
%   Comparison strategy:
%     - When eps is small (mass stays inside [0,1]^2), the analytical
%       Gaussian SB is a valid reference.
%     - When eps is large (mass escapes the domain), the Neumann-BC
%       Sinkhorn solution is a better reference because it uses the same
%       reflected-BM physics as the FP projection.
%
%   By default both comparisons are run and saved to the .mat file.
%   Disable Sinkhorn with: cfg.run_sinkhorn_comparison = false;
%
%   Works in both interactive and headless (batch) modes.
%   Set cfg.use_gpu = true in cfg_ladmm_gaussian to run on GPU.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));
clear functions   % flush MATLAB's function cache so cfg overrides are picked up

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
fig_dir = fullfile(res_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
if exist('cfg_ladmm_gaussian_run', 'file')
    cfg = cfg_ladmm_gaussian_run();
    fprintf('Config: cfg_ladmm_gaussian_run (local override)\n');
else
    cfg = cfg_ladmm_gaussian();
    fprintf('Config: cfg_ladmm_gaussian (defaults)\n');
end
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

nt = problem.nt;  dt = problem.dt;
nx = problem.nx;  dx = problem.dx;
ny = problem.ny;  dy = problem.dy;
xx = problem.xx;  yy = problem.yy;

%% --- Run ---
fprintf('Running %s  (nt=%d, nx=%d, ny=%d, eps=%.4g, gpu=%d)...\n', ...
    cfg.name, cfg.nt, cfg.nx, cfg.ny, cfg.vareps, ...
    isfield(cfg,'use_gpu') && cfg.use_gpu);

result = cfg.pipeline(cfg, problem);

fprintf('  iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);
fprintf('  time/iter=%.3fs  throughput=%.1f iter/s\n', ...
    result.time_per_iter, result.throughput);
if result.gpu_total_mb > 0
    fprintf('  GPU mem: %.0f / %.0f MB used/total\n', ...
        result.gpu_mem_post_mb, result.gpu_total_mb);
end

proj_label = strrep(func2str(cfg.projection), 'proj_fokker_planck_', '');
ftag = sprintf('nt%d_nx%d_ny%d_gam%g_tau%g_eps%g_proj_%s', ...
    cfg.nt, cfg.nx, cfg.ny, cfg.gamma, cfg.tau, cfg.vareps, proj_label);

%% --- Analytical solution ---
[rho_ana, ~, ~] = analytical_sb_gaussian(problem, cfg.vareps);

ops  = problem.ops;
rho0 = problem.rho0;
rho1 = problem.rho1;

rho_ana_cc = ops.interp_t_at_phi(rho_ana, rho0, rho1);   % (nt x nx x ny)
rho_num_cc = result.rho_cc;

%% --- Sinkhorn-Neumann comparison ---
% Run Sinkhorn with Neumann BCs on the same grid and eps.
% This is the correct reference when eps is large enough that mass
% escapes the domain (reflected BM matches the FP projection BCs).
%
% Set cfg.run_sinkhorn_comparison = false to skip (e.g. pure OT, eps=0).
run_sink = ~isfield(cfg, 'run_sinkhorn_comparison') || cfg.run_sinkhorn_comparison;
if run_sink && cfg.vareps > 0
    fprintf('Running Sinkhorn-Neumann (nt=%d, nx=%d, ny=%d, eps=%.4g)...\n', ...
        cfg.nt, cfg.nx, cfg.ny, cfg.vareps);
    cfg_sink.vareps            = cfg.vareps;
    cfg_sink.max_iter          = 2000;
    cfg_sink.tol               = 1e-10;
    cfg_sink.precomp_heat      = @precomp_heat_neumann_2d;
    cfg_sink.use_pdf_marginals = true;   % keep density in PDF units (matches rho_cc scale)
    t_sink = tic;
    sink_result = sinkhorn_hopf_cole(problem, cfg_sink);
    fprintf('  Sinkhorn: iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
        sink_result.iters, sink_result.converged, sink_result.error, toc(t_sink));

    % Interpolate Sinkhorn rho from integer time nodes (nt+1 slices) to
    % LADMM cell-centre half-integer times (nt slices):
    %   rho_sink_cc(k,:,:) = 0.5 * (rho_sink(k,:,:) + rho_sink(k+1,:,:))
    rho_sink_cc = 0.5 * (sink_result.rho(1:nt,:,:) + sink_result.rho(2:nt+1,:,:));
else
    if cfg.vareps == 0
        fprintf('Skipping Sinkhorn comparison (eps=0, pure OT).\n');
    end
    sink_result = [];
    rho_sink_cc = [];
end

%% --- Save results to .mat ---
obj = compute_objective(result.rho_stag, result.mx_stag, result.my_stag, problem);
fprintf('Kinetic energy (staggered): %.6f\n', obj);

mat_file = fullfile(res_dir, sprintf('result_%s.mat', ftag));
save(mat_file, 'result', 'cfg', 'problem', 'rho_ana_cc', 'rho_sink_cc', 'sink_result', 'obj', 'ftag');
fprintf('Results saved to %s\n', mat_file);

%% --- Skip figures in batch/headless mode ---
if ~usejava('desktop')
    fprintf('Headless mode: skipping figures. Run postprocess_sb_gaussian.m locally.\n');
    return;
end

%% --- Figures (interactive mode only) ---
newfig = @(name) figure('Name', name, 'Visible', 'off', ...
                         'Position', [0 0 1200 500]);

% Figure 1: density heatmaps at selected times
t_fracs = [0.1, 0.3, 0.5, 0.7, 0.9];
n_t     = numel(t_fracs);
clim_max = max(rho_ana_cc(:));

fig1 = newfig('Density snapshots');
fig1.Position = [0 0 1200 500];
for p = 1:n_t
    k = max(1, round(t_fracs(p) * nt));

    subplot(2, n_t, p);
    imagesc(xx, yy, squeeze(rho_ana_cc(k,:,:))');
    axis xy; colorbar; clim([0, clim_max]);
    title(sprintf('Ana  t=%.2f', (k-0.5)*dt));
    xlabel('x'); ylabel('y');

    subplot(2, n_t, n_t + p);
    imagesc(xx, yy, squeeze(rho_num_cc(k,:,:))');
    axis xy; colorbar; clim([0, clim_max]);
    title(sprintf('LADMM  t=%.2f', (k-0.5)*dt));
    xlabel('x'); ylabel('y');
end
sgtitle(sprintf('Density  gamma=%.4g  tau=%.4g  eps=%.4g', ...
    cfg.gamma, cfg.tau, cfg.vareps));
exportgraphics(fig1, fullfile(fig_dir, sprintf('density_%s.png', ftag)), 'Resolution', 150);
close(fig1);

% Figure 2: x-slice at y = 0.5
[~, iy] = min(abs(yy - 0.5));
t_fracs2 = [0.1, 0.25, 0.5, 0.75, 0.9];
n_t2     = numel(t_fracs2);
colors   = parula(n_t2);

fig2 = newfig('x-slice');
fig2.Position = [0 0 700 400];
hold on;
leg_handles = gobjects(n_t2, 2);
for p = 1:n_t2
    k = max(1, round(t_fracs2(p) * nt));
    leg_handles(p,1) = plot(xx, rho_ana_cc(k,:,iy), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    leg_handles(p,2) = plot(xx, rho_num_cc(k,:,iy), 'o--','Color', colors(p,:), 'MarkerSize', 4);
end
leg_str = cell(n_t2, 2);
for p = 1:n_t2
    k = max(1, round(t_fracs2(p) * nt));
    leg_str{p,1} = sprintf('Ana   t=%.2f', (k-0.5)*dt);
    leg_str{p,2} = sprintf('LADMM t=%.2f', (k-0.5)*dt);
end
legend(leg_handles(:), leg_str(:), 'Location', 'best', 'FontSize', 7);
xlabel('x'); ylabel('\rho(t,x,y=0.5)');
title(sprintf('x-slice at y=0.5   eps=%.4g', cfg.vareps));
grid on;
exportgraphics(fig2, fullfile(fig_dir, sprintf('xslice_%s.png', ftag)), 'Resolution', 150);
close(fig2);

% Figure 3: ADMM residual
fig3 = newfig('ADMM residual');
fig3.Position = [0 0 600 300];
semilogy(result.residual, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol = %.1e', cfg.tol));
xlabel('Iteration'); ylabel('Residual');
title(sprintf('ADMM residual   iters=%d,  converged=%d', result.iters, result.converged));
grid on;
exportgraphics(fig3, fullfile(fig_dir, sprintf('residual_%s.png', ftag)), 'Resolution', 150);
close(fig3);

% Figure 4: L2 error vs analytical
err  = sqrt(dx * dy * sum(sum((rho_num_cc - rho_ana_cc).^2, 2), 3));
t_cc = ((1:nt)' - 0.5) * dt;

fig4 = newfig('L2 error');
fig4.Position = [0 0 600 300];
plot(t_cc, err, 'b-', 'LineWidth', 1.5);
xlabel('t'); ylabel('L2 error in rho');
title('L2 error vs analytical SB');
grid on;
exportgraphics(fig4, fullfile(fig_dir, sprintf('l2error_%s.png', ftag)), 'Resolution', 150);
close(fig4);

fprintf('Figures saved to %s\n', fig_dir);
