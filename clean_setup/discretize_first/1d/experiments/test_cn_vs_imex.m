% TEST_CN_VS_IMEX  Compare CN-ADMM vs IMEX-ADMM, both against Sinkhorn reference.
%
%   Runs three solvers on the Gaussian-to-Gaussian SB problem:
%     - CN ADMM   (cfg_ladmm_gaussian,      proj_fokker_planck_banded)
%     - IMEX ADMM (cfg_ladmm_gaussian_imex,  proj_fokker_planck_imex)
%     - Sinkhorn-Hopf-Cole (reference, Neumann BCs)
%
%   Grid alignment for comparison:
%     res_cn.rho_stag    (ntm x nx) at t = dt, 2dt, ..., (nt-1)*dt
%     res_imex.rho_stag  (ntm x nx) at same staggered times
%     res_sink.rho(2:nt,:) (ntm x nx) at same staggered times
%
%   Figures (saved to results/figures/):
%     cn_vs_imex_density_<tag>.png   -- density evolution, all three overlaid
%     cn_vs_imex_l2err_<tag>.png     -- L2 error vs Sinkhorn, CN and IMEX

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% --- Config ---
VAREPS = 1.0;
NT     = 256;
NX     = 256;

cfg_cn          = cfg_ladmm_gaussian();
cfg_cn.vareps   = VAREPS;
cfg_cn.nt       = NT;
cfg_cn.nx       = NX;

cfg_imex        = cfg_ladmm_gaussian_imex();
cfg_imex.vareps = VAREPS;
cfg_imex.nt     = NT;
cfg_imex.nx     = NX;

cfg_sink.vareps       = VAREPS;
cfg_sink.max_iter     = 500;
cfg_sink.tol          = 1e-10;
cfg_sink.precomp_heat = @precomp_heat_neumann;

prob_def = prob_gaussian();
problem  = setup_problem(cfg_cn, prob_def);   % same grid for both ADMM variants

ftag = sprintf('nt%d_nx%d_eps%g', NT, NX, VAREPS);

%% --- Solve ---
fprintf('Running Sinkhorn  (eps=%.4g)...\n', VAREPS);
res_sink = sinkhorn_hopf_cole(problem, cfg_sink);
fprintf('  iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    res_sink.iters, res_sink.converged, res_sink.error, res_sink.walltime);

fprintf('Running CN-ADMM   (nt=%d, nx=%d, eps=%.4g)...\n', NT, NX, VAREPS);
res_cn = discretize_then_optimize(cfg_cn, problem);
fprintf('  iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    res_cn.iters, res_cn.converged, res_cn.error, res_cn.walltime);

fprintf('Running IMEX-ADMM (nt=%d, nx=%d, eps=%.4g)...\n', NT, NX, VAREPS);
res_imex = discretize_then_optimize(cfg_imex, problem);
fprintf('  iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    res_imex.iters, res_imex.converged, res_imex.error, res_imex.walltime);

%% --- Align grids ---
nt  = problem.nt;   ntm = nt - 1;
dt  = problem.dt;   dx  = problem.dx;
xx  = problem.xx;   nx  = problem.nx;

rho_sink_stag = res_sink.rho(2:nt, :);   % (ntm x nx), same times as rho_stag
t_stag_vec    = (1:ntm)' * dt;           % (ntm x 1)

%% --- L2 errors vs Sinkhorn ---
err_cn   = sqrt(dx * sum((res_cn.rho_stag   - rho_sink_stag).^2, 2));
err_imex = sqrt(dx * sum((res_imex.rho_stag - rho_sink_stag).^2, 2));

%% --- Figure 1: Density evolution (all three overlaid) ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
n_t     = numel(t_fracs);
cmap    = lines(n_t);
stride  = max(1, floor(nx / 60));

FS = 11;   LW = 1.5;   MS = 4;

fig1 = figure('Units','centimeters','Position',[2 2 18 11]);
hold on;

for p = 1:n_t
    k   = max(1, min(ntm, round(t_fracs(p) * nt)));
    idx = 1:stride:nx;
    col = cmap(p,:);

    plot(xx, rho_sink_stag(k,:), '-',  'Color', col, 'LineWidth', LW, ...
        'HandleVisibility', 'off');
    plot(xx(idx), res_cn.rho_stag(k,idx), 'o', 'Color', col, ...
        'MarkerSize', MS, 'MarkerFaceColor', 'none', 'LineWidth', 0.9, ...
        'HandleVisibility', 'off');
    plot(xx(idx), res_imex.rho_stag(k,idx), '^', 'Color', col, ...
        'MarkerSize', MS, 'MarkerFaceColor', 'none', 'LineWidth', 0.9, ...
        'HandleVisibility', 'off');
end

% Legend: method markers
h1 = plot(nan,nan, 'k-',  'LineWidth', LW,  'DisplayName', 'Sinkhorn');
h2 = plot(nan,nan, 'ko',  'MarkerSize', MS, 'MarkerFaceColor','none', ...
    'LineWidth', 0.9, 'DisplayName', 'CN-ADMM');
h3 = plot(nan,nan, 'k^',  'MarkerSize', MS, 'MarkerFaceColor','none', ...
    'LineWidth', 0.9, 'DisplayName', 'IMEX-ADMM');

% Legend: time slices
h_t = gobjects(n_t, 1);
for p = 1:n_t
    h_t(p) = plot(nan,nan, '-', 'Color', cmap(p,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$t=%.2f$', t_fracs(p)));
end

xlabel('$x$', 'FontSize', FS);
ylabel('$\tilde{\rho}(t,x)$', 'FontSize', FS);
title(sprintf('Density evolution: CN vs IMEX vs Sinkhorn  ($\\varepsilon=%.4g$)', VAREPS), ...
    'FontSize', FS);
legend([h1; h2; h3; h_t], 'Location','best', 'FontSize', FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

saveas(fig1, fullfile(fig_dir, sprintf('cn_vs_imex_density_%s.png', ftag)));

%% --- Figure 2: L2 error vs Sinkhorn ---
fig2 = figure('Units','centimeters','Position',[2 2 14 9]);
semilogy(t_stag_vec, err_cn,   'b-',  'LineWidth', LW, 'DisplayName', 'CN-ADMM');
hold on;
semilogy(t_stag_vec, err_imex, 'r--', 'LineWidth', LW, 'DisplayName', 'IMEX-ADMM');
xlabel('$t$', 'FontSize', FS);
ylabel('$\|\tilde{\rho}^* - \rho_\mathrm{Sink}\|_{L^2(x)}$', 'FontSize', FS);
title(sprintf('$L^2$ error vs Sinkhorn  ($\\varepsilon=%.4g$)', VAREPS), 'FontSize', FS);
legend('Location','best', 'FontSize', FS, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

saveas(fig2, fullfile(fig_dir, sprintf('cn_vs_imex_l2err_%s.png', ftag)));

%% --- Summary ---
fprintf('\n--- Summary (eps=%.4g, nt=%d, nx=%d) ---\n', VAREPS, NT, NX);
fprintf('  Sinkhorn:  wall=%.2fs  iters=%d\n', res_sink.walltime, res_sink.iters);
fprintf('  CN-ADMM:   wall=%.2fs  iters=%d  max_err=%.3e\n', ...
    res_cn.walltime,   res_cn.iters,   max(err_cn));
fprintf('  IMEX-ADMM: wall=%.2fs  iters=%d  max_err=%.3e\n', ...
    res_imex.walltime, res_imex.iters, max(err_imex));
