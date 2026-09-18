% TEST_SB_EXTENDED_DOMAIN  Boundary-effect study on [0,L] for the Gaussian SB.
%
%   Places two Gaussians at L/2 ± 1/6 on a domain wide enough that boundary
%   effects are negligible (L=8 gives > 5*sigma_max from both walls).
%
%   Compares four methods:
%     (1) CN-ADMM          (Neumann BCs, discretize-then-optimize)
%     (2) Sinkhorn-Neumann (reflected BM, DCT heat kernel)
%     (3) Sinkhorn-FS      (free-space BM, Gaussian convolution)
%     (4) Analytical       (exact SB marginal on R)
%
%   Figures saved to results/figures/.
%   Run setup_paths first, or call from the repo root.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% --- Config ---
% L=4 gives dx=4/1024=0.0039 (same resolution as the validated L=1/nx=256 case).
% At t=0.5 the Gaussian has sigma_max~0.71; the center is L/2=2 from each wall,
% so the Neumann mirror correction is exp(-(2/0.71)^2/2) ~ 1e-4: negligible.
% This is the right tradeoff — L=8 would need nx=2048 to match this resolution.
L    = 4;
VAREPS = 1.0;

cfg_admm         = cfg_ladmm_gaussian();
cfg_admm.nt      = 256;
cfg_admm.nx      = 1024;
cfg_admm.vareps  = VAREPS;
cfg_admm.L       = L;
cfg_admm.max_iter = 10000;

prob_def = prob_gaussian_centered(L);
problem  = setup_problem(cfg_admm, prob_def);

ftag = sprintf('L%g_nt%d_nx%d_eps%g', L, cfg_admm.nt, cfg_admm.nx, VAREPS);

%% --- Run CN-ADMM ---
fprintf('Running CN-ADMM  (nt=%d, nx=%d, L=%g, eps=%.4g)...\n', ...
    cfg_admm.nt, cfg_admm.nx, L, VAREPS);
res_admm = discretize_then_optimize(cfg_admm, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    res_admm.iters, res_admm.converged, res_admm.error, res_admm.walltime);

%% --- Run Sinkhorn (Neumann) ---
cfg_sink_neu.vareps       = VAREPS;
cfg_sink_neu.max_iter     = 500;
cfg_sink_neu.tol          = 1e-10;
cfg_sink_neu.precomp_heat = @precomp_heat_neumann;

fprintf('Running Sinkhorn-Neumann  (eps=%.4g)...\n', VAREPS);
res_neu = sinkhorn_hopf_cole(problem, cfg_sink_neu);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    res_neu.iters, res_neu.converged, res_neu.error, res_neu.walltime);

%% --- Run Sinkhorn (free-space) ---
cfg_sink_fs.vareps            = VAREPS;
cfg_sink_fs.max_iter          = 500;
cfg_sink_fs.tol               = 1e-10;
cfg_sink_fs.precomp_heat      = @precomp_heat_free_space;
cfg_sink_fs.use_pdf_marginals = true;   % free-space: use raw PDF, not discrete mass

fprintf('Running Sinkhorn-FS  (eps=%.4g)...\n', VAREPS);
res_fs = sinkhorn_hopf_cole(problem, cfg_sink_fs);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    res_fs.iters, res_fs.converged, res_fs.error, res_fs.walltime);

%% --- Analytical solution ---
% On staggered times (for ADMM comparison): (ntm x nx)
[rho_ana_stag, ~] = analytical_sb_gaussian(problem, VAREPS);
ops = problem.ops;
rho_ana_cc = ops.interp_t_at_phi(rho_ana_stag, problem.rho0, problem.rho1);  % (nt x nx)

% On edge times (for Sinkhorn comparison): (nt+1 x nx), raw PDF
rho_ana_edge_pdf = analytical_sb_gaussian_R(problem, VAREPS);   % raw PDF values

nt = problem.nt;   dt = problem.dt;
nx = problem.nx;   dx = problem.dx;
xx = problem.xx;

% Normalize all outputs to discrete mass (sum=1 per row) for consistent comparison.
% Sinkhorn-FS uses use_pdf_marginals=true so its raw output is in PDF units (~1/dx larger);
% row-normalizing brings it to the same convention as Neumann and ADMM.
rho_ana_edge_norm = rho_ana_edge_pdf ./ sum(rho_ana_edge_pdf, 2);   % (nt+1 x nx)
rho_neu_norm      = res_neu.rho      ./ sum(res_neu.rho,      2);   % (nt+1 x nx)
rho_fs_norm       = res_fs.rho       ./ sum(res_fs.rho,       2);   % (nt+1 x nx)

%% --- Time indices for plotting (cosine-clustered, interior grid) ---
n_plot = 7;
ntm_   = nt - 1;
t_idx  = unique(round(1 + (ntm_-1)/2 * (1 - cos(pi*(1:n_plot)/(n_plot+1)))));
t_idx(1) = 2;
t_idx(end) = ntm-1;
n_t    = numel(t_idx);
colors = parula(n_t);
stride = max(1, floor(nx / 80));

%% --- Figure 1: Density evolution — all four methods at selected times ---
figure('Name', sprintf('Extended domain  L=%g  eps=%.4g', L, VAREPS), ...
    'Position', [50 50 1300 400]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel A: CN-ADMM vs Analytical
nexttile; hold on;
for p = 1:n_t
    k = t_idx(p);   t_cc = (k - 0.5)*dt;
    plot(xx, rho_ana_cc(k,:),            '-',  'Color', colors(p,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
    plot(xx(1:stride:nx), res_admm.rho_cc(k,1:stride:nx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 3, 'MarkerFaceColor', colors(p,:), 'HandleVisibility', 'off');
end
h_a = plot(nan,nan,'k-', 'LineWidth',1.5, 'DisplayName','Analytical');
h_b = plot(nan,nan,'ko', 'MarkerSize',3,  'MarkerFaceColor',[.5 .5 .5], 'DisplayName','CN-ADMM');
h_t = gobjects(n_t,1);
for p=1:n_t
    h_t(p)=plot(nan,nan,'-','Color',colors(p,:),'LineWidth',2, ...
        'DisplayName',sprintf('$t=%.3f$',(t_idx(p)-0.5)*dt));
end
legend([h_a;h_b;h_t],'Location','best','FontSize',7,'Box','off');
xlabel('$x$'); ylabel('$\rho$');
title(sprintf('CN-ADMM  ($L=%g$, $\\varepsilon$=%.4g)', L, VAREPS));
grid on;

% Panel B: Sinkhorn-Neumann vs Analytical (normalized mass)
nexttile; hold on;
for p = 1:n_t
    k = t_idx(p);
    plot(xx, rho_ana_edge_norm(k+1,:),      '-',  'Color', colors(p,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
    plot(xx(1:stride:nx), rho_neu_norm(k+1,1:stride:nx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 3, 'MarkerFaceColor', colors(p,:), 'HandleVisibility', 'off');
end
h_a2=plot(nan,nan,'k-', 'LineWidth',1.5, 'DisplayName','Analytical');
h_b2=plot(nan,nan,'ko', 'MarkerSize',3,  'MarkerFaceColor',[.5 .5 .5], 'DisplayName','Sinkhorn-Neumann');
h_t2=gobjects(n_t,1);
for p=1:n_t
    h_t2(p)=plot(nan,nan,'-','Color',colors(p,:),'LineWidth',2, ...
        'DisplayName',sprintf('$t=%.3f$',t_idx(p)*dt));
end
legend([h_a2;h_b2;h_t2],'Location','best','FontSize',7,'Box','off');
xlabel('$x$'); ylabel('$\rho$ (mass, sum=1)');
title(sprintf('Sinkhorn-Neumann  ($L=%g$, $\\varepsilon$=%.4g)', L, VAREPS));
grid on;

% Panel C: Sinkhorn-FS vs Analytical (normalized mass — same units as Panel B)
nexttile; hold on;
for p = 1:n_t
    k = t_idx(p);
    plot(xx, rho_ana_edge_norm(k+1,:),    '-',  'Color', colors(p,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
    plot(xx(1:stride:nx), rho_fs_norm(k+1,1:stride:nx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 3, 'MarkerFaceColor', colors(p,:), 'HandleVisibility', 'off');
end
h_a3=plot(nan,nan,'k-', 'LineWidth',1.5, 'DisplayName','Analytical');
h_b3=plot(nan,nan,'ko', 'MarkerSize',3,  'MarkerFaceColor',[.5 .5 .5], 'DisplayName','Sinkhorn-FS');
h_t3=gobjects(n_t,1);
for p=1:n_t
    h_t3(p)=plot(nan,nan,'-','Color',colors(p,:),'LineWidth',2, ...
        'DisplayName',sprintf('$t=%.3f$',t_idx(p)*dt));
end
legend([h_a3;h_b3;h_t3],'Location','best','FontSize',7,'Box','off');
xlabel('$x$'); ylabel('$\rho$ (mass, sum=1)');
title(sprintf('Sinkhorn-FS  ($L=%g$, $\\varepsilon$=%.4g)', L, VAREPS));
grid on;

sgtitle(sprintf('Density evolution   $L=%g$,  $\\varepsilon$=%.4g  (solid=analytical, circles=numerical)', ...
    L, VAREPS));
saveas(gcf, fullfile(fig_dir, sprintf('extended_density_%s.png', ftag)));

%% --- Figure 2: L2 error vs time for all three methods ---
err_admm  = sqrt(dx * sum((res_admm.rho_cc                   - rho_ana_cc).^2,               2));
err_neu   = sqrt(dx * sum((rho_neu_norm(2:nt,:)              - rho_ana_edge_norm(2:nt,:)).^2, 2));
err_fs    = sqrt(dx * sum((rho_fs_norm(2:nt,:)               - rho_ana_edge_norm(2:nt,:)).^2, 2));

t_cc        = ((1:nt)' - 0.5) * dt;
t_edge_inner = (1:nt-1)' * dt;

figure('Name', sprintf('L2 error  L=%g  eps=%.4g', L, VAREPS), 'Position', [50 500 700 320]);
hold on;
plot(t_cc,          err_admm, 'b-',  'LineWidth', 1.5, 'DisplayName', 'CN-ADMM');
plot(t_edge_inner,  err_neu,  'r--', 'LineWidth', 1.5, 'DisplayName', 'Sinkhorn-Neumann');
plot(t_edge_inner,  err_fs,   'g-.', 'LineWidth', 1.5, 'DisplayName', 'Sinkhorn-FS');
xlabel('$t$'); ylabel('$\|\rho_\mathrm{num} - \rho_\mathrm{ana}\|_{L^2(x)}$');
title(sprintf('$L^2$ error vs analytical   $L=%g$,  $\\varepsilon$=%.4g', L, VAREPS));
legend('Location','best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('extended_l2error_%s.png', ftag)));

%% --- Figure 3: Sinkhorn convergence ---
figure('Name', 'Sinkhorn convergence', 'Position', [800 50 600 360]);
semilogy(res_neu.errors, 'r-',  'LineWidth', 1.5, 'DisplayName', 'Sinkhorn-Neumann');
hold on;
semilogy(res_fs.errors,  'g--', 'LineWidth', 1.5, 'DisplayName', 'Sinkhorn-FS');
yline(cfg_sink_neu.tol, 'k--', sprintf('tol=%.1e', cfg_sink_neu.tol), 'HandleVisibility','off');
xlabel('Sinkhorn iteration'); ylabel('Left-marginal $L^2$ error');
title(sprintf('Sinkhorn convergence   $L=%g$,  $\\varepsilon$=%.4g', L, VAREPS));
legend('Location','best'); grid on;
saveas(gcf, fullfile(fig_dir, sprintf('extended_sinkhorn_conv_%s.png', ftag)));

%% --- Figure 4: ADMM residual ---
figure('Name', 'ADMM residual', 'Position', [800 430 600 280]);
semilogy(res_admm.residual, 'b-', 'LineWidth', 1.5);
yline(cfg_admm.tol, 'r--', sprintf('tol=%.1e', cfg_admm.tol));
xlabel('ADMM iteration'); ylabel('$\|y^{k+1} - y^k\|$');
title(sprintf('ADMM residual   iters=%d,  converged=%d', res_admm.iters, res_admm.converged));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('extended_admm_residual_%s.png', ftag)));

%% --- Figure 5: Density evolution — ADMM vs both Sinkhhorns ---
% Use staggered grid t = k*dt (k=1..ntm): same for res_admm.rho_stag and res_sink.rho(2:nt,:)
ntm     = nt - 1;
t_stag  = (1:ntm)' * dt;

rho_stag_neu = rho_neu_norm(2:nt, :);   % (ntm x nx), normalized mass
rho_stag_fs  = rho_fs_norm(2:nt, :);   % (ntm x nx), normalized mass

figure('Name', sprintf('ADMM vs Sinkhorn  L=%g  eps=%.4g', L, VAREPS), ...
    'Position', [50 50 1100 400]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel A: ADMM vs Sinkhorn-Neumann
nexttile; hold on;
for p = 1:n_t
    k   = t_idx(p);
    idx = 1:stride:nx;
    plot(xx, rho_stag_neu(k,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
    plot(xx(idx), res_admm.rho_stag(k,idx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 3, 'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
end
h_r1 = plot(nan,nan,'k-',  'LineWidth',1.5, 'DisplayName','Sinkhorn-Neumann');
h_n1 = plot(nan,nan,'ko',  'MarkerSize',3,  'MarkerFaceColor','none', 'DisplayName','CN-ADMM');
h_t1 = gobjects(n_t,1);
for p = 1:n_t
    h_t1(p) = plot(nan,nan,'-','Color',colors(p,:),'LineWidth',2, ...
        'DisplayName',sprintf('$t=%.3f$', t_idx(p)*dt));
end
legend([h_r1;h_n1;h_t1],'Location','best','FontSize',7,'Box','off');
xlabel('$x$'); ylabel('$\rho$');
title(sprintf('CN-ADMM vs Sinkhorn-Neumann  ($L=%g$, $\\varepsilon$=%.4g)', L, VAREPS));
grid on;

% Panel B: ADMM vs Sinkhorn-FS
nexttile; hold on;
for p = 1:n_t
    k   = t_idx(p);
    idx = 1:stride:nx;
    plot(xx, rho_stag_fs(k,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
    plot(xx(idx), res_admm.rho_stag(k,idx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 3, 'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
end
h_r2 = plot(nan,nan,'k-',  'LineWidth',1.5, 'DisplayName','Sinkhorn-FS');
h_n2 = plot(nan,nan,'ko',  'MarkerSize',3,  'MarkerFaceColor','none', 'DisplayName','CN-ADMM');
h_t2 = gobjects(n_t,1);
for p = 1:n_t
    h_t2(p) = plot(nan,nan,'-','Color',colors(p,:),'LineWidth',2, ...
        'DisplayName',sprintf('$t=%.3f$', t_idx(p)*dt));
end
legend([h_r2;h_n2;h_t2],'Location','best','FontSize',7,'Box','off');
xlabel('$x$'); ylabel('$\rho$');
title(sprintf('CN-ADMM vs Sinkhorn-FS  ($L=%g$, $\\varepsilon$=%.4g)', L, VAREPS));
grid on;

sgtitle(sprintf('CN-ADMM vs Sinkhorn   $L=%g$,  $\\varepsilon$=%.4g  (solid=Sinkhorn, circles=CN-ADMM)', ...
    L, VAREPS));
saveas(gcf, fullfile(fig_dir, sprintf('extended_admm_vs_sink_%s.png', ftag)));

%% --- Figure 6: L2 error — ADMM vs both Sinkhhorns ---
% All on the same staggered time grid t = k*dt, k=1..ntm.
% Use normalized mass (sum=1 per row) throughout for a fair comparison.
rho_admm_stag_norm = res_admm.rho_stag ./ sum(res_admm.rho_stag, 2);

err_vs_neu = sqrt(dx * sum((rho_admm_stag_norm - rho_stag_neu).^2, 2));   % (ntm x 1)
err_vs_fs  = sqrt(dx * sum((rho_admm_stag_norm - rho_stag_fs).^2,  2));   % (ntm x 1)

figure('Name', sprintf('ADMM vs Sinkhorn L2  L=%g  eps=%.4g', L, VAREPS), ...
    'Position', [50 500 700 320]);
semilogy(t_stag, err_vs_neu, 'r-',  'LineWidth', 1.5, 'DisplayName', 'vs Sinkhorn-Neumann');
hold on;
semilogy(t_stag, err_vs_fs,  'g--', 'LineWidth', 1.5, 'DisplayName', 'vs Sinkhorn-FS');
xlabel('$t$');
ylabel('$\|\rho_\mathrm{ADMM} - \rho_\mathrm{Sink}\|_{L^2(x)}$');
title(sprintf('CN-ADMM vs Sinkhorn   $L=%g$,  $\\varepsilon$=%.4g', L, VAREPS));
legend('Location','best'); grid on;
saveas(gcf, fullfile(fig_dir, sprintf('extended_admm_vs_sink_l2_%s.png', ftag)));

%% --- Summary ---
fprintf('\n--- Summary (L=%g, eps=%.4g, nt=%d, nx=%d) ---\n', L, VAREPS, nt, nx);
fprintf('  CN-ADMM:   wall=%.2fs  iters=%d  max_L2_vs_ana=%.3e\n', ...
    res_admm.walltime, res_admm.iters, max(err_admm));
fprintf('  Sink-Neu:  wall=%.2fs  iters=%d  max_L2_vs_ana=%.3e\n', ...
    res_neu.walltime, res_neu.iters, max(err_neu));
fprintf('  Sink-FS:   wall=%.2fs  iters=%d  max_L2_vs_ana=%.3e\n', ...
    res_fs.walltime, res_fs.iters, max(err_fs));
fprintf('  ADMM vs Sink-Neu:  max_L2=%.3e\n', max(err_vs_neu));
fprintf('  ADMM vs Sink-FS:   max_L2=%.3e\n', max(err_vs_fs));
fprintf('Figures saved to: %s\n', fig_dir);
