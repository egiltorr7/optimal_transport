% TEST_EXPSEMI  Compare exact-semigroup ADMM vs Sinkhorn.
%
%   Figures (saved to results/figures/):
%     expsemi_density_<tag>.png   -- density evolution, expsemi vs Sinkhorn
%     expsemi_l2err_<tag>.png     -- absolute + relative L2 error vs Sinkhorn over time
%     expsemi_eps_sweep.png       -- max absolute + relative L2 error vs eps

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% --- Config ---
VAREPS = 0;
NT     = 128;
NX     = 128;

cfg_es          = cfg_ladmm_gaussian_expsemi();
cfg_es.vareps   = VAREPS;
cfg_es.nt       = NT;
cfg_es.nx       = NX;

cfg_sink.vareps       = VAREPS;
cfg_sink.max_iter     = 5000;
cfg_sink.tol          = -1;
cfg_sink.precomp_heat = @precomp_heat_neumann;

prob_def = prob_gaussian();
problem  = setup_problem(cfg_es, prob_def);

nt  = problem.nt;   ntm = nt - 1;
dt  = problem.dt;   dx  = problem.dx;
xx  = problem.xx;   nx  = problem.nx;

ftag = sprintf('nt%d_nx%d_eps%g', NT, NX, VAREPS);

%% --- Solve ---
EPS_ANA_THRESH = 3e-5;   % below this, Sinkhorn is unreliable; use analytical SB

fprintf('Running Expsemi-ADMM (eps=%.4g, nt=%d)...\n', VAREPS, NT);
res_es = discretize_then_optimize(cfg_es, problem);
fprintf('  iters=%d  converged=%d  wall=%.1fs\n', ...
    res_es.iters, res_es.converged, res_es.walltime);

if VAREPS < EPS_ANA_THRESH
    fprintf('eps < %.0e: using analytical SB reference.\n', EPS_ANA_THRESH);
    [rho_ana, ~]  = analytical_sb_gaussian(problem, VAREPS);
    rho_ref_stag  = rho_ana;                % (ntm x nx)
    ref_label     = 'Analytical SB';
else
    fprintf('Running Sinkhorn     (eps=%.4g)...\n', VAREPS);
    res_sink     = sinkhorn_hopf_cole(problem, cfg_sink);
    fprintf('  iters=%d  converged=%d  wall=%.1fs\n', ...
        res_sink.iters, res_sink.converged, res_sink.walltime);
    rho_ref_stag = res_sink.rho(2:nt, :);  % (ntm x nx)
    ref_label    = 'Sinkhorn';
end

t_stag_vec = (1:ntm)' * dt;

err_es     = sqrt(dx * sum((res_es.rho_stag - rho_ref_stag).^2, 2));
norm_ref   = sqrt(dx * sum(rho_ref_stag.^2, 2));
delta_ref  = 0.0;
rel_err_es = err_es ./ (norm_ref + delta_ref);

%% --- Figure 1: density evolution ---
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
    plot(xx, rho_ref_stag(k,:), '-',  'Color', col, 'LineWidth', LW, ...
        'HandleVisibility', 'off');
    plot(xx(idx), res_es.rho_stag(k,idx), 'o', 'Color', col, ...
        'MarkerSize', MS, 'MarkerFaceColor', 'none', 'LineWidth', 0.9, ...
        'HandleVisibility', 'off');
end

h1 = plot(nan,nan, 'k-',  'LineWidth', LW, 'DisplayName', ref_label);
h2 = plot(nan,nan, 'ko',  'MarkerSize', MS, 'MarkerFaceColor','none', ...
    'LineWidth', 0.9, 'DisplayName', 'Expsemi-ADMM');
h_t = gobjects(n_t, 1);
for p = 1:n_t
    h_t(p) = plot(nan,nan, '-', 'Color', cmap(p,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$t=%.2f$', t_fracs(p)));
end

xlabel('$x$', 'FontSize', FS);
ylabel('$\tilde{\rho}(t,x)$', 'FontSize', FS);
title(sprintf('Density: Expsemi-ADMM vs %s  ($\\varepsilon=%.4g$)', ref_label, VAREPS), 'FontSize', FS);
legend([h1; h2; h_t], 'Location','best', 'FontSize', FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;
saveas(fig1, fullfile(fig_dir, sprintf('expsemi_density_%s.png', ftag)));

%% --- Figure 2: absolute + relative L2 error vs Sinkhorn ---
fig2 = figure('Units','centimeters','Position',[2 2 22 9]);

subplot(1,2,1);
semilogy(t_stag_vec, err_es, 'r-', 'LineWidth', LW);
xlabel('$t$', 'FontSize', FS);
ylabel(sprintf('$\\|\\tilde{\\rho}^* - \\rho_\\mathrm{%s}\\|_{L^2(x)}$', ref_label), 'FontSize', FS);
title(sprintf('Absolute $L^2$ error ($\\varepsilon=%.4g$)', VAREPS), 'FontSize', FS);
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

subplot(1,2,2);
semilogy(t_stag_vec, rel_err_es, 'b-', 'LineWidth', LW);
xlabel('$t$', 'FontSize', FS);
ylabel(sprintf('$\\|\\tilde{\\rho}^* - \\rho_\\mathrm{%s}\\|_{L^2} / \\|\\rho_\\mathrm{%s}\\|_{L^2}$', ref_label, ref_label), 'FontSize', FS);
title(sprintf('Relative $L^2$ error ($\\varepsilon=%.4g$)', VAREPS), 'FontSize', FS);
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

saveas(fig2, fullfile(fig_dir, sprintf('expsemi_l2err_%s.png', ftag)));

%% --- Figure 3: max error vs eps sweep ---
eps_vals = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 4.0];
ne       = numel(eps_vals);
max_err     = nan(1, ne);
max_rel_err = nan(1, ne);
sweep_labels = cell(1, ne);

fprintf('\nEps sweep (NT=%d, NX=%d):\n', NT, NX);
fprintf('  %-8s  %-12s  %-14s  %-14s\n', 'eps', 'ref', 'max_abs_err', 'max_rel_err');
for i = 1:ne
    eps_i        = eps_vals(i);
    cfg_i        = cfg_es;
    cfg_i.vareps = eps_i;
    prob_i       = setup_problem(cfg_i, prob_def);
    res_es_i     = discretize_then_optimize(cfg_i, prob_i);

    if eps_i < EPS_ANA_THRESH
        [rho_ana_i, ~] = analytical_sb_gaussian(prob_i, eps_i);
        rho_ref_i      = rho_ana_i;
        sweep_labels{i} = 'Ana';
    else
        cfg_sk_i.vareps       = eps_i;
        cfg_sk_i.max_iter     = 500;
        cfg_sk_i.tol          = 1e-10;
        cfg_sk_i.precomp_heat = @precomp_heat_neumann;
        res_sk_i       = sinkhorn_hopf_cole(prob_i, cfg_sk_i);
        rho_ref_i      = res_sk_i.rho(2:prob_i.nt, :);
        sweep_labels{i} = 'Sink';
    end

    e              = sqrt(prob_i.dx * sum((res_es_i.rho_stag - rho_ref_i).^2, 2));
    nrm_ref_i      = sqrt(prob_i.dx * sum(rho_ref_i.^2, 2));
    rel_e          = e ./ nrm_ref_i;
    max_err(i)     = max(e);
    max_rel_err(i) = max(rel_e);

    fprintf('  %-8g  %-12s  %-14.2e  %-14.2e\n', eps_i, sweep_labels{i}, max_err(i), max_rel_err(i));
end

fig3 = figure('Units','centimeters','Position',[2 2 22 9]);

subplot(1,2,1);
loglog(eps_vals, max_err, 'r^-', 'LineWidth', LW, 'MarkerSize', 6);
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('$\max_t \|\tilde{\rho} - \rho_\mathrm{ref}\|_{L^2(x)}$', 'FontSize', FS);
title(sprintf('Max absolute error vs $\\varepsilon$  ($N_T=%d$, $N_x=%d$)', NT, NX), 'FontSize', FS);
xline(EPS_ANA_THRESH, 'k--', 'LineWidth', 1);
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

subplot(1,2,2);
loglog(eps_vals, max_rel_err, 'b^-', 'LineWidth', LW, 'MarkerSize', 6);
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('$\max_t \|\tilde{\rho} - \rho_\mathrm{ref}\|_{L^2} / \|\rho_\mathrm{ref}\|_{L^2}$', 'FontSize', FS);
title(sprintf('Max relative error vs $\\varepsilon$  ($N_T=%d$, $N_x=%d$)', NT, NX), 'FontSize', FS);
xline(EPS_ANA_THRESH, 'k--', 'LineWidth', 1);
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

saveas(fig3, fullfile(fig_dir, 'expsemi_eps_sweep.png'));

%% --- Summary ---
fprintf('\n--- Summary (eps=%.4g, nt=%d, nx=%d, ref=%s) ---\n', VAREPS, NT, NX, ref_label);
if VAREPS >= EPS_ANA_THRESH
    fprintf('  Sinkhorn:     wall=%.2fs  iters=%d\n', res_sink.walltime, res_sink.iters);
end
fprintf('  Expsemi-ADMM: wall=%.2fs  iters=%d  max_abs_err=%.3e  max_rel_err=%.3e\n', ...
    res_es.walltime, res_es.iters, max(err_es), max(rel_err_es));
