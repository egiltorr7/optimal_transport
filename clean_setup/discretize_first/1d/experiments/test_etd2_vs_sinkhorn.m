% TEST_ETD2_VS_SINKHORN  Compare ETD1 and ETD2 ADMM against log-domain Sinkhorn.
%
%   Runs both ETD1 (expsemi, m at half-integer times) and ETD2 (m at integer
%   times, endpoint-weighted) against a high-accuracy log-domain Sinkhorn
%   reference on the 1D Gaussian Schrodinger bridge problem.
%
%   Figures (saved to results/figures/):
%     etd2_density_<tag>.png    -- density profiles at selected t
%     etd2_l2err_<tag>.png      -- absolute + relative L2 error vs t
%     etd2_eps_sweep_<tag>.png  -- max relative L2 error vs eps (ETD1 vs ETD2)

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
VAREPS = 1e-2;   % single-eps test: large eps to expose ETD1 midpoint bias
NT     = 128;
NX     = 128;

prob_def = prob_gaussian();

%% -------------------------------------------------------------------------
%  Configs
% -------------------------------------------------------------------------
cfg_base        = cfg_ladmm_gaussian_expsemi();
cfg_base.nt     = NT;
cfg_base.nx     = NX;
cfg_base.vareps = VAREPS;

cfg_etd1 = cfg_base;   % ETD1: proj_fokker_planck_expsemi, half-int m

cfg_etd2          = cfg_ladmm_gaussian_expsemi_etd2();
cfg_etd2.nt       = NT;
cfg_etd2.nx       = NX;
cfg_etd2.vareps   = VAREPS;

cfg_sink.vareps       = VAREPS;
cfg_sink.max_iter     = 5000;
cfg_sink.tol          = 1e-12;
cfg_sink.precomp_heat = @precomp_heat_neumann_log;

problem = setup_problem(cfg_etd1, prob_def);
nt  = problem.nt;   ntm = nt - 1;
dt  = problem.dt;   dx  = problem.dx;
xx  = problem.xx;   nx  = problem.nx;

ftag = sprintf('nt%d_nx%d_eps%g', NT, NX, VAREPS);

%% -------------------------------------------------------------------------
%  Solve (single eps)
% -------------------------------------------------------------------------
fprintf('--- Single-eps test: eps=%.4g, NT=%d, NX=%d ---\n', VAREPS, NT, NX);

fprintf('Running log-domain Sinkhorn...\n');
res_sink = sinkhorn_hopf_cole_logdomain(problem, cfg_sink);
fprintf('  iters=%d  converged=%d  wall=%.2fs\n', ...
    res_sink.iters, res_sink.converged, res_sink.walltime);

fprintf('Running ETD1-ADMM...\n');
res_etd1 = discretize_then_optimize(cfg_etd1, problem);
fprintf('  iters=%d  converged=%d  wall=%.2fs\n', ...
    res_etd1.iters, res_etd1.converged, res_etd1.walltime);

fprintf('Running ETD2-ADMM...\n');
res_etd2 = discretize_then_optimize_etd2(cfg_etd2, problem);
fprintf('  iters=%d  converged=%d  wall=%.2fs\n', ...
    res_etd2.iters, res_etd2.converged, res_etd2.walltime);

%% -------------------------------------------------------------------------
%  Align grids
% -------------------------------------------------------------------------
rho_ref  = res_sink.rho(2:nt, :);   % (ntm x nx): interior integer times

t_vec   = (1:ntm)' * dt;

err_etd1    = sqrt(dx * sum((res_etd1.rho_stag - rho_ref).^2, 2));   % (ntm x 1)
err_etd2    = sqrt(dx * sum((res_etd2.rho_stag - rho_ref).^2, 2));
nrm_ref     = sqrt(dx * sum(rho_ref.^2, 2));
rel_err_etd1 = err_etd1 ./ nrm_ref;
rel_err_etd2 = err_etd2 ./ nrm_ref;

fprintf('\nMax relative L2 error vs Sinkhorn:\n');
fprintf('  ETD1: %.3e\n', max(rel_err_etd1));
fprintf('  ETD2: %.3e\n', max(rel_err_etd2));

%% -------------------------------------------------------------------------
%  Figure 1: density profiles at selected times
% -------------------------------------------------------------------------
FS = 11;   LW = 1.5;   MS = 4;
col_etd1 = [0.13 0.47 0.71];   % blue
col_etd2 = [0.84 0.15 0.16];   % red
col_sink = [0.2  0.6  0.2 ];   % green

t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
np      = numel(t_fracs);
cmap    = lines(np);
stride  = max(1, floor(nx / 50));

fig1 = figure('Units','centimeters','Position',[2 2 20 11]);
hold on;

for p = 1:np
    k   = max(1, min(ntm, round(t_fracs(p) * nt)));
    col = cmap(p,:);
    plot(xx, rho_ref(k,:),              '-',  'Color', col, 'LineWidth', LW, ...
        'HandleVisibility', 'off');
    plot(xx(1:stride:end), res_etd1.rho_stag(k,1:stride:end), 'o', ...
        'Color', col, 'MarkerSize', MS, 'MarkerFaceColor','none', 'LineWidth',0.9, ...
        'HandleVisibility', 'off');
    plot(xx(1:stride:end), res_etd2.rho_stag(k,1:stride:end), 's', ...
        'Color', col, 'MarkerSize', MS, 'MarkerFaceColor','none', 'LineWidth',0.9, ...
        'HandleVisibility', 'off');
end

h_sink = plot(nan,nan, 'k-',  'LineWidth', LW, 'DisplayName', 'Log-Sinkhorn (ref)');
h_etd1 = plot(nan,nan, 'ko',  'MarkerSize', MS, 'LineWidth', 0.9, ...
    'MarkerFaceColor','none', 'DisplayName', 'ETD1-ADMM');
h_etd2 = plot(nan,nan, 'ks',  'MarkerSize', MS, 'LineWidth', 0.9, ...
    'MarkerFaceColor','none', 'DisplayName', 'ETD2-ADMM');
h_t = gobjects(np,1);
for p = 1:np
    h_t(p) = plot(nan,nan, '-', 'Color', cmap(p,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$t=%.2f$', t_fracs(p)));
end

xlabel('$x$', 'FontSize', FS);
ylabel('$\rho(t,x)$', 'FontSize', FS);
title(sprintf('Density: ETD1 vs ETD2 vs Log-Sinkhorn  ($\\varepsilon=%.4g$)', VAREPS), ...
    'FontSize', FS);
legend([h_sink; h_etd1; h_etd2; h_t], 'Location','best', 'FontSize',FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;
saveas(fig1, fullfile(fig_dir, sprintf('etd2_density_%s.png', ftag)));
fprintf('Saved: etd2_density_%s.png\n', ftag);

%% -------------------------------------------------------------------------
%  Figure 2: L2 error vs t
% -------------------------------------------------------------------------
fig2 = figure('Units','centimeters','Position',[2 2 22 9]);

subplot(1,2,1);
semilogy(t_vec, err_etd1, '-',  'Color', col_etd1, 'LineWidth', LW, ...
    'DisplayName', 'ETD1-ADMM');
hold on;
semilogy(t_vec, err_etd2, '--', 'Color', col_etd2, 'LineWidth', LW, ...
    'DisplayName', 'ETD2-ADMM');
xlabel('$t$', 'FontSize', FS);
ylabel('$\Vert\rho - \rho_{\mathrm{ref}}\Vert_{L^2}$', 'FontSize', FS);
title(sprintf('Absolute $L^2$ error  ($\\varepsilon=%.4g$)', VAREPS), 'FontSize', FS);
legend('Location','best', 'FontSize',FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

subplot(1,2,2);
semilogy(t_vec, rel_err_etd1, '-',  'Color', col_etd1, 'LineWidth', LW, ...
    'DisplayName', 'ETD1-ADMM');
hold on;
semilogy(t_vec, rel_err_etd2, '--', 'Color', col_etd2, 'LineWidth', LW, ...
    'DisplayName', 'ETD2-ADMM');
xlabel('$t$', 'FontSize', FS);
ylabel('$\Vert\rho - \rho_{\mathrm{ref}}\Vert_{L^2} / \Vert\rho_{\mathrm{ref}}\Vert_{L^2}$', ...
    'FontSize', FS);
title(sprintf('Relative $L^2$ error  ($\\varepsilon=%.4g$)', VAREPS), 'FontSize', FS);
legend('Location','best', 'FontSize',FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

saveas(fig2, fullfile(fig_dir, sprintf('etd2_l2err_%s.png', ftag)));
fprintf('Saved: etd2_l2err_%s.png\n', ftag);

%% -------------------------------------------------------------------------
%  Epsilon sweep: max relative L2 error vs eps
% -------------------------------------------------------------------------
EPS_VALS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0];
NE       = numel(EPS_VALS);

max_rel_etd1 = nan(NE,1);
max_rel_etd2 = nan(NE,1);
conv_etd1    = false(NE,1);
conv_etd2    = false(NE,1);

fprintf('\n--- Epsilon sweep (NT=%d, NX=%d) ---\n', NT, NX);
fprintf('  %-8s  %-14s  %-14s  %-8s  %-8s\n', ...
    'eps', 'rel_etd1', 'rel_etd2', 'it_etd1', 'it_etd2');
fprintf('  %s\n', repmat('-',1,60));

for i = 1:NE
    eps_i = EPS_VALS(i);

    cfg_i1        = cfg_etd1;
    cfg_i1.vareps = eps_i;

    cfg_i2        = cfg_etd2;
    cfg_i2.vareps = eps_i;

    cfg_sk_i.vareps       = eps_i;
    cfg_sk_i.max_iter     = 5000;
    cfg_sk_i.tol          = 1e-12;
    cfg_sk_i.precomp_heat = @precomp_heat_neumann_log;
    if eps_i < 0.05
        cfg_sk_i.eps_init      = min(1.0, eps_i * 100);
        cfg_sk_i.anneal_factor = 4.0;
    end

    prob_i  = setup_problem(cfg_i1, prob_def);
    nti     = prob_i.nt;
    dxi     = prob_i.dx;

    r_sk   = sinkhorn_hopf_cole_logdomain(prob_i, cfg_sk_i);
    rho_ri = r_sk.rho(2:nti, :);
    nrm_i  = sqrt(dxi * sum(rho_ri.^2, 2));

    r1 = discretize_then_optimize(cfg_i1, prob_i);
    e1 = sqrt(dxi * sum((r1.rho_stag - rho_ri).^2, 2));
    max_rel_etd1(i) = max(e1 ./ nrm_i);
    conv_etd1(i)    = r1.converged;

    r2 = discretize_then_optimize_etd2(cfg_i2, prob_i);
    e2 = sqrt(dxi * sum((r2.rho_stag - rho_ri).^2, 2));
    max_rel_etd2(i) = max(e2 ./ nrm_i);
    conv_etd2(i)    = r2.converged;

    fprintf('  %-8g  %-14.3e  %-14.3e  %-8d  %-8d\n', ...
        eps_i, max_rel_etd1(i), max_rel_etd2(i), r1.iters, r2.iters);
end

%% -------------------------------------------------------------------------
%  Figure 3: max relative error vs eps
% -------------------------------------------------------------------------
fig3 = figure('Units','centimeters','Position',[2 2 14 9]);
loglog(EPS_VALS, max_rel_etd1, '-o', 'Color', col_etd1, 'LineWidth', LW, ...
    'MarkerSize', 6, 'MarkerFaceColor', col_etd1, 'DisplayName', 'ETD1-ADMM');
hold on;
loglog(EPS_VALS, max_rel_etd2, '--s', 'Color', col_etd2, 'LineWidth', LW, ...
    'MarkerSize', 6, 'MarkerFaceColor', col_etd2, 'DisplayName', 'ETD2-ADMM');
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('$\max_t \Vert\rho - \rho_{\mathrm{ref}}\Vert_{L^2} / \Vert\rho_{\mathrm{ref}}\Vert_{L^2}$', ...
    'FontSize', FS);
title(sprintf('Max relative $L^2$ error vs $\\varepsilon$  ($N_T=%d$, $N_x=%d$)', NT, NX), ...
    'FontSize', FS);
legend('Location','best', 'FontSize',FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

ftag_sweep = sprintf('nt%d_nx%d', NT, NX);
saveas(fig3, fullfile(fig_dir, sprintf('etd2_eps_sweep_%s.png', ftag_sweep)));
fprintf('\nSaved: etd2_eps_sweep_%s.png\n', ftag_sweep);

%% -------------------------------------------------------------------------
%  Console summary
% -------------------------------------------------------------------------
fprintf('\n--- Summary: ETD1 vs ETD2 vs log-Sinkhorn (eps=%.4g, NT=%d, NX=%d) ---\n', ...
    VAREPS, NT, NX);
fprintf('  Log-Sinkhorn: wall=%.2fs  iters=%d\n', res_sink.walltime, res_sink.iters);
fprintf('  ETD1-ADMM:    wall=%.2fs  iters=%d  max_rel_err=%.3e\n', ...
    res_etd1.walltime, res_etd1.iters, max(rel_err_etd1));
fprintf('  ETD2-ADMM:    wall=%.2fs  iters=%d  max_rel_err=%.3e\n', ...
    res_etd2.walltime, res_etd2.iters, max(rel_err_etd2));
