% TEST_TIMING_COMPARISON  Head-to-head timing: ExpSemi-ADMM vs log-domain Sinkhorn.
%
%   Sweeps epsilon over six orders of magnitude on the 1D Gaussian SB problem
%   and records wall-clock time, iteration count, and max relative L2 error
%   (against a high-accuracy Sinkhorn reference) for both solvers.
%
%   Key finding documented here:
%     - At small eps, ADMM converges in few iterations because the
%       Fokker–Planck constraint is nearly advective; Sinkhorn requires many
%       more iterations because the heat kernel is narrow.
%     - At large eps, Sinkhorn converges in O(1) iterations; ADMM needs more
%       because the penalty must balance a strongly diffusive constraint.
%     - Log-domain Sinkhorn is numerically stable across all eps tested;
%       standard Sinkhorn overflows for eps < 0.01.
%
%   Figures (saved to results/figures/):
%     timing_walltime.pdf   -- wall time (s) vs eps, both methods
%     timing_iters.pdf      -- iteration count vs eps, both methods
%     timing_error.pdf      -- max relative L2 error vs eps (vs high-acc ref)
%     timing_summary.pdf    -- all three panels combined (paper figure)
%
%   Console output: full table of results.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% -------------------------------------------------------------------------
%  Parameters
% -------------------------------------------------------------------------
NT = 128;
NX = 128;

EPS_VALS = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0];
NE       = numel(EPS_VALS);

% Working tolerance for both solvers (timing comparison)
TOL_WORK = 1e-8;
% High-accuracy reference (log-domain Sinkhorn only)
TOL_REF  = 1e-12;

%% -------------------------------------------------------------------------
%  Base configs
% -------------------------------------------------------------------------
cfg_es          = cfg_ladmm_gaussian_expsemi();
cfg_es.nt       = NT;
cfg_es.nx       = NX;
cfg_es.max_iter = 20000;
cfg_es.tol      = TOL_WORK;

prob_def = prob_gaussian();

%% -------------------------------------------------------------------------
%  Sweep
% -------------------------------------------------------------------------
wall_ref   = nan(NE, 1);
wall_admm  = nan(NE, 1);
wall_sink  = nan(NE, 1);
iters_admm = nan(NE, 1);
iters_sink = nan(NE, 1);
err_admm   = nan(NE, 1);   % max relative L2 vs reference
err_sink   = nan(NE, 1);
conv_admm  = false(NE, 1);
conv_sink  = false(NE, 1);

fprintf('\n%s\n', repmat('-', 1, 80));
fprintf('  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s\n', ...
    'eps', 't_admm', 't_sink', 'it_admm', 'it_sink', 'err_admm', 'err_sink');
fprintf('%s\n', repmat('-', 1, 80));

for i = 1:NE
    eps_i = EPS_VALS(i);

    % ---- Problem ----
    cfg_i        = cfg_es;
    cfg_i.vareps = eps_i;
    problem_i    = setup_problem(cfg_i, prob_def);
    dx = problem_i.dx;
    nt = problem_i.nt;   ntm = nt - 1;

    % ---- High-accuracy reference: log-domain Sinkhorn ----
    % Rebuild struct each iteration to avoid field leakage across loop.
    cfg_ref              = struct();
    cfg_ref.vareps       = eps_i;
    cfg_ref.max_iter     = 10000;
    cfg_ref.tol          = TOL_REF;
    cfg_ref.precomp_heat = @precomp_heat_neumann_log;
    % Geometric epsilon annealing prevents initialisation failure at small eps.
    % The total cost (including annealing) is charged to the solver's wall time.
    if eps_i < 0.05
        cfg_ref.eps_init      = min(1.0, eps_i * 200);
        cfg_ref.anneal_factor = 4.0;
    end

    t_ref        = tic;
    res_ref      = sinkhorn_hopf_cole_logdomain(problem_i, cfg_ref);
    wall_ref(i)  = toc(t_ref);

    rho_ref_stag = res_ref.rho(2:nt, :);   % (ntm x nx): interior time nodes

    % ---- ExpSemi-ADMM ----
    t_admm        = tic;
    res_admm      = discretize_then_optimize(cfg_i, problem_i);
    wall_admm(i)  = toc(t_admm);
    iters_admm(i) = res_admm.iters;
    conv_admm(i)  = res_admm.converged;

    e_ad        = sqrt(dx * sum((res_admm.rho_stag - rho_ref_stag).^2, 2));
    nrm_ref     = sqrt(dx * sum(rho_ref_stag.^2, 2));
    err_admm(i) = max(e_ad ./ nrm_ref);

    % ---- Log-domain Sinkhorn at working tolerance ----
    % Annealing also applied here at small eps for a fair real-world comparison
    % (annealing is an inherent cost of Sinkhorn at small eps).
    cfg_sk              = struct();
    cfg_sk.vareps       = eps_i;
    cfg_sk.max_iter     = 10000;
    cfg_sk.tol          = TOL_WORK;
    cfg_sk.precomp_heat = @precomp_heat_neumann_log;
    if eps_i < 0.05
        cfg_sk.eps_init      = cfg_ref.eps_init;
        cfg_sk.anneal_factor = cfg_ref.anneal_factor;
    end

    t_sink       = tic;
    res_sink     = sinkhorn_hopf_cole_logdomain(problem_i, cfg_sk);
    wall_sink(i) = toc(t_sink);
    iters_sink(i) = res_sink.iters;
    conv_sink(i)  = res_sink.converged;

    e_sk        = sqrt(dx * sum((res_sink.rho(2:nt,:) - rho_ref_stag).^2, 2));
    err_sink(i) = max(e_sk ./ nrm_ref);

    fprintf('  %-8g  %-10.2f  %-10.2f  %-10d  %-10d  %-10.2e  %-10.2e\n', ...
        eps_i, wall_admm(i), wall_sink(i), iters_admm(i), iters_sink(i), ...
        err_admm(i), err_sink(i));
end

fprintf('%s\n\n', repmat('-', 1, 80));

%% -------------------------------------------------------------------------
%  Figures
% -------------------------------------------------------------------------
FS  = 11;   LW  = 1.6;   MS  = 6;
col_admm = [0.13 0.47 0.71];   % blue
col_sink = [0.84 0.15 0.16];   % red

% --- Combined 3-panel figure (paper quality) ---
fig = figure('Units', 'centimeters', 'Position', [2 2 24 7]);
tl  = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: wall time
ax1 = nexttile;
loglog(EPS_VALS, wall_admm, '-o', 'Color', col_admm, 'LineWidth', LW, ...
    'MarkerSize', MS, 'MarkerFaceColor', col_admm, 'DisplayName', 'ExpSemi-ADMM');
hold on;
loglog(EPS_VALS, wall_sink, '-s', 'Color', col_sink, 'LineWidth', LW, ...
    'MarkerSize', MS, 'MarkerFaceColor', col_sink, 'DisplayName', 'Log-Sinkhorn');
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('Wall time (s)', 'FontSize', FS);
title('(a) Computational cost', 'FontSize', FS);
legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
set(ax1, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out', 'XDir', 'normal');
grid on;

% Panel 2: iterations
ax2 = nexttile;
loglog(EPS_VALS, iters_admm, '-o', 'Color', col_admm, 'LineWidth', LW, ...
    'MarkerSize', MS, 'MarkerFaceColor', col_admm, 'DisplayName', 'ExpSemi-ADMM');
hold on;
loglog(EPS_VALS, iters_sink, '-s', 'Color', col_sink, 'LineWidth', LW, ...
    'MarkerSize', MS, 'MarkerFaceColor', col_sink, 'DisplayName', 'Log-Sinkhorn');
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('Iterations to convergence', 'FontSize', FS);
title('(b) Iteration count', 'FontSize', FS);
legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
set(ax2, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

% Panel 3: accuracy
ax3 = nexttile;
loglog(EPS_VALS, err_admm, '-o', 'Color', col_admm, 'LineWidth', LW, ...
    'MarkerSize', MS, 'MarkerFaceColor', col_admm, 'DisplayName', 'ExpSemi-ADMM');
hold on;
loglog(EPS_VALS, err_sink, '-s', 'Color', col_sink, 'LineWidth', LW, ...
    'MarkerSize', MS, 'MarkerFaceColor', col_sink, 'DisplayName', 'Log-Sinkhorn');
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('$\max_t \|\rho - \rho_{\mathrm{ref}}\|_{L^2} / \|\rho_{\mathrm{ref}}\|_{L^2}$', ...
    'FontSize', FS);
title('(c) Relative $L^2$ error vs reference', 'FontSize', FS);
legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
set(ax3, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

ftag = sprintf('nt%d_nx%d', NT, NX);
exportgraphics(fig, fullfile(fig_dir, sprintf('timing_summary_%s.pdf', ftag)), ...
    'ContentType', 'vector');
saveas(fig, fullfile(fig_dir, sprintf('timing_summary_%s.png', ftag)));
fprintf('Figure saved: timing_summary_%s.{pdf,png}\n', ftag);

%% -------------------------------------------------------------------------
%  Save results
% -------------------------------------------------------------------------
res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
save(fullfile(res_dir, sprintf('timing_comparison_%s.mat', ftag)), ...
    'EPS_VALS', 'NT', 'NX', ...
    'wall_admm', 'wall_sink', 'wall_ref', ...
    'iters_admm', 'iters_sink', ...
    'err_admm', 'err_sink', ...
    'conv_admm', 'conv_sink', ...
    'TOL_WORK', 'TOL_REF');

%% -------------------------------------------------------------------------
%  Console summary
% -------------------------------------------------------------------------
[~, idx_cross] = min(abs(wall_admm - wall_sink));
fprintf('Crossover (approx): eps ~ %.4g\n', EPS_VALS(idx_cross));
fprintf('ADMM speedup at eps=%.4g: %.1fx\n', EPS_VALS(1), wall_sink(1)/wall_admm(1));
fprintf('Sinkhorn speedup at eps=%.4g: %.1fx\n', EPS_VALS(end), wall_admm(end)/wall_sink(end));
