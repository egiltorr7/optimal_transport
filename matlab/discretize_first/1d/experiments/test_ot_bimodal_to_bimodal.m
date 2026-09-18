%% test_ot_bimodal_to_bimodal.m  [discretize_first / LADMM]
%
% OT / SB between two bimodal distributions (inward motion).
%
%   rho0 = N(0.20, 0.04^2) + N(0.80, 0.04^2)
%   rho1 = N(0.35, 0.04^2) + N(0.65, 0.04^2)
%
% OT (eps=0) non-crossing plan:
%   left  peak: 0.20 -> 0.35  (v_L = +0.15)
%   right peak: 0.80 -> 0.65  (v_R = -0.15)
%   KE = W_2^2 = 0.0225
%
% Checks:
%   1. KE(LADMM, eps=0) -> 0.0225 (non-crossing plan found).
%   2. rho_stag and mx_stag vs analytical displacement interpolation.
%   3. KE(eps) monotone increasing from W_2^2 with eps.
%   4. Grid refinement O(h^2).

clear; clc; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config ---
cfg    = cfg_ladmm_gaussian();
cfg.nt = 64;   cfg.nx = 128;

prob_def    = prob_bimodal_to_bimodal();
problem     = setup_problem(cfg, prob_def);
ke_exact_ot = prob_def.ke_ot;   % 0.0225

ftag_main = sprintf('nt%d_nx%d_gam%g_tau%g', cfg.nt, cfg.nx, cfg.gamma, cfg.tau);

nt  = problem.nt;   ntm = nt-1;   dt = problem.dt;
nx  = problem.nx;   dx  = problem.dx;
xx  = problem.xx;   t_rho = (1:ntm)'*dt;

%% --- OT (eps=0) ---
cfg_ot = cfg;   cfg_ot.vareps = 0;
fprintf('Running LADMM OT (eps=0): bimodal->bimodal  nt=%d nx=%d ...\n', cfg.nt, cfg.nx);
result_ot = cfg_ot.pipeline(cfg_ot, problem);
fprintf('  iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    result_ot.iters, result_ot.converged, result_ot.error, result_ot.walltime);

[rho_ana, mx_ana] = analytical_ot_bimodal_to_bimodal(problem);

ke_ot  = compute_objective(result_ot.rho_stag, result_ot.mx_stag, problem) / dx;
ke_ana = compute_objective(rho_ana, mx_ana, problem) / dx;
err_rho = sqrt(dx * sum((result_ot.rho_stag - rho_ana).^2, 2));
err_mx  = sqrt(dx * sum((result_ot.mx_stag  - mx_ana ).^2, 2));

fprintf('\n--- OT checks ---\n');
fprintf('  KE (LADMM)        = %.6f\n', ke_ot);
fprintf('  KE (W_2^2 exact)  = %.6f\n', ke_exact_ot);
fprintf('  KE (analytical)   = %.6f\n', ke_ana);
fprintf('  KE crossing plan  = %.6f   (solver should avoid)\n', 2*0.5*0.45^2);
fprintf('  max L2 err rho    = %.2e\n', max(err_rho));
fprintf('  max L2 err mx     = %.2e\n', max(err_mx));

%% --- SB (eps=0.05) ---
vareps = 5e-2;
cfg_sb = cfg;   cfg_sb.vareps = vareps;
fprintf('\nRunning LADMM SB (eps=%.2g) ...\n', vareps);
result_sb = cfg_sb.pipeline(cfg_sb, problem);
ke_sb = compute_objective(result_sb.rho_stag, result_sb.mx_stag, problem) / dx;
fprintf('  KE (SB eps=%.2g) = %.6f\n', vareps, ke_sb);

%% --- Video 1: OT density evolution ---
vid_ot = VideoWriter(fullfile(fig_dir, ...
    sprintf('ladmm_ot_bimodal_to_bimodal__ot__%s.mp4', ftag_main)), 'MPEG-4');
vid_ot.FrameRate = 30;
open(vid_ot);

fig_v = figure('Name', 'ladmm_ot_bimodal_to_bimodal__video_ot', ...
               'Position', [100,100,720,420], 'Visible', 'on');
y_max = max([max(problem.rho0), max(problem.rho1)]) * 1.15;

% build full time axis including t=0 and t=1
t_all  = [0; (1:ntm)'*dt; 1];
n_all  = numel(t_all);
for fi = 1:n_all
    clf(fig_v);   ax = axes(fig_v);   hold(ax, 'on');
    t_cur = t_all(fi);
    if fi == 1
        row_a = problem.rho0;   row_n = problem.rho0;
    elseif fi == n_all
        row_a = problem.rho1;   row_n = problem.rho1;
    else
        k = fi - 1;   % index into rho_stag (1..ntm)
        row_a = rho_ana(k,:);
        row_n = result_ot.rho_stag(k,:);
    end
    plot(ax, xx, row_a, 'k-',  'LineWidth', 2.0, 'DisplayName', 'Exact');
    plot(ax, xx, row_n, 'b--', 'LineWidth', 1.5, 'DisplayName', 'LADMM');
    xlim(ax, [0 1]);   ylim(ax, [0 y_max]);
    xlabel(ax, '$x$', 'Interpreter','latex');
    ylabel(ax, '$\rho$', 'Interpreter','latex');
    title(ax, sprintf('OT ($\\varepsilon=0$): $t=%.3f$   KE=%.4f vs %.4f', ...
                      t_cur, ke_ot, ke_exact_ot), 'Interpreter','latex');
    legend(ax, 'Interpreter','latex', 'Location','best');
    grid(ax, 'on');
    drawnow;
    writeVideo(vid_ot, getframe(fig_v));
end
close(vid_ot);
close(fig_v);
fprintf('Saved OT video.\n');

%% --- Video 2: SB density evolution ---
vid_sb = VideoWriter(fullfile(fig_dir, ...
    sprintf('ladmm_ot_bimodal_to_bimodal__sb_eps%g__%s.mp4', vareps, ftag_main)), 'MPEG-4');
vid_sb.FrameRate = 30;
open(vid_sb);

fig_v2 = figure('Name', 'ladmm_ot_bimodal_to_bimodal__video_sb', ...
                'Position', [100,100,720,420], 'Visible', 'on');
y_max_sb = max([max(result_sb.rho_stag(:)), max(problem.rho0), max(problem.rho1)]) * 1.15;

for fi = 1:n_all
    clf(fig_v2);   ax = axes(fig_v2);   hold(ax, 'on');
    t_cur = t_all(fi);
    if fi == 1
        row_ot = problem.rho0;   row_sb = problem.rho0;
    elseif fi == n_all
        row_ot = problem.rho1;   row_sb = problem.rho1;
    else
        k = fi - 1;
        row_ot = result_ot.rho_stag(k,:);
        row_sb = result_sb.rho_stag(k,:);
    end
    plot(ax, xx, row_ot, 'b-',  'LineWidth', 1.5, 'DisplayName', ...
         sprintf('OT ($\\varepsilon=0$)  KE=%.4f', ke_ot));
    plot(ax, xx, row_sb, 'r-',  'LineWidth', 1.5, 'DisplayName', ...
         sprintf('SB ($\\varepsilon=%.2g$)  KE=%.4f', vareps, ke_sb));
    xlim(ax, [0 1]);   ylim(ax, [0 y_max_sb]);
    xlabel(ax, '$x$', 'Interpreter','latex');
    ylabel(ax, '$\rho$', 'Interpreter','latex');
    title(ax, sprintf('OT vs SB: $t=%.3f$', t_cur), 'Interpreter','latex');
    legend(ax, 'Interpreter','latex', 'Location','best');
    grid(ax, 'on');
    drawnow;
    writeVideo(vid_sb, getframe(fig_v2));
end
close(vid_sb);
close(fig_v2);
fprintf('Saved SB video.\n');

%% --- Figure 2: OT vs SB at t=0.5 ---
k_mid = round(0.5*nt);
figure('Name', 'ladmm_ot_bimodal_to_bimodal__ot_vs_sb_at_t05', 'Position', [100,540,700,340]);
plot(xx, rho_ana(k_mid,:),            'k-',  'LineWidth', 1.5); hold on;
plot(xx, result_ot.rho_stag(k_mid,:), 'b--', 'LineWidth', 1.5);
plot(xx, result_sb.rho_stag(k_mid,:), 'r-',  'LineWidth', 1.5);
xlabel('$x$','Interpreter','latex'); ylabel('$\rho(t=0.5,x)$','Interpreter','latex');
title('Density at $t=0.5$: OT vs SB','Interpreter','latex');
legend('OT exact', sprintf('OT LADMM  KE=%.4f', ke_ot), ...
       sprintf('SB LADMM $\\varepsilon=%.2g$  KE=%.4f', vareps, ke_sb), ...
       'Interpreter','latex','Location','best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_ot_bimodal_to_bimodal__ot_vs_sb_at_t05__%s.png', ftag_main)));

%% --- Figure 3: L2 error vs time (OT) ---
t_mx = ((1:nt)' - 0.5) * dt;

figure('Name', 'ladmm_ot_bimodal_to_bimodal__l2error_vs_time', 'Position', [820,540,560,320]);
plot(t_rho, err_rho, 'b-', 'LineWidth', 1.5); hold on;
plot(t_mx, err_mx, 'r-', 'LineWidth', 1.5);
xlabel('$t$','Interpreter','latex'); ylabel('$L^2$ error','Interpreter','latex');
legend('$\|\rho_\mathrm{LADMM}-\rho_\mathrm{exact}\|$', ...
       '$\|m_\mathrm{LADMM}-m_\mathrm{exact}\|$','Interpreter','latex','Location','best');
title('$L^2$ error vs time (OT, bimodal$\to$bimodal)','Interpreter','latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_ot_bimodal_to_bimodal__l2error_vs_time__%s.png', ftag_main)));

%% --- Figure 4: eps sweep at t=0.5 ---
eps_vals   = [0, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1];
ke_eps     = zeros(numel(eps_vals),1);
colors_eps = lines(numel(eps_vals));
k_half     = round(0.5*nt);

figure('Name', 'ladmm_ot_bimodal_to_bimodal__eps_sweep', 'Position', [100,870,720,420]);
hold on;
fprintf('\neps sweep:\n');
for j = 1:numel(eps_vals)
    cfg_j = cfg;   cfg_j.vareps = eps_vals(j);
    r_j   = cfg_j.pipeline(cfg_j, problem);
    ke_eps(j) = compute_objective(r_j.rho_stag, r_j.mx_stag, problem) / dx;
    fprintf('  eps=%.1e  KE=%.6f\n', eps_vals(j), ke_eps(j));
    plot(xx, r_j.rho_stag(k_half,:), '-', 'Color', colors_eps(j,:), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('$\\varepsilon=%.0e$  KE=%.4f', eps_vals(j), ke_eps(j)));
end
xlabel('$x$','Interpreter','latex'); ylabel('$\rho(t=0.5,x)$','Interpreter','latex');
title('LADMM bimodal$\to$bimodal: $\varepsilon$ sweep at $t=0.5$','Interpreter','latex');
legend('Interpreter','latex','Location','best','FontSize',8,'NumColumns',2);
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_ot_bimodal_to_bimodal__eps_sweep__%s.png', ftag_main)));

%% --- Figure 5: KE vs eps ---
figure('Name', 'ladmm_ot_bimodal_to_bimodal__ke_vs_eps', 'Position', [840,870,540,340]);
semilogx(eps_vals(2:end), ke_eps(2:end), 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
yline(ke_exact_ot,'r--','LineWidth',1.5,'Label',sprintf('W_2^2=%.4f',ke_exact_ot));
scatter(eps_vals(1), ke_eps(1), 80, 'b', 'filled');
xlabel('$\varepsilon$','Interpreter','latex','FontSize',13);
ylabel('Kinetic energy','Interpreter','latex');
title('LADMM: KE monotone increasing with $\varepsilon$','Interpreter','latex');
legend('KE($\varepsilon>0$)','$W_2^2$','KE($\varepsilon=0$)','Interpreter','latex','Location','best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_ot_bimodal_to_bimodal__ke_vs_eps__%s.png', ftag_main)));

%% --- Grid refinement (eps=0) ---
nt_vals = [16, 32, 64, 128];   nx_vals = [32, 64, 128, 256];   n_grids = numel(nt_vals);
ke_grid  = zeros(n_grids,1);   err_grid = zeros(n_grids,1);

fprintf('\nGrid refinement (eps=0):\n');
for k = 1:n_grids
    cfg_k = cfg;   cfg_k.vareps = 0;   cfg_k.nt = nt_vals(k);   cfg_k.nx = nx_vals(k);
    prob_k = setup_problem(cfg_k, prob_def);
    r = cfg_k.pipeline(cfg_k, prob_k);
    [rho_a, mx_a] = analytical_ot_bimodal_to_bimodal(prob_k);
    ke_grid(k)  = compute_objective(r.rho_stag, r.mx_stag, prob_k) / prob_k.dx;
    err_r = norm(r.rho_stag(:) - rho_a(:)) * sqrt(prob_k.dt * prob_k.dx);
    err_m = norm(r.mx_stag(:)  - mx_a(:))  * sqrt(prob_k.dt * prob_k.dx);
    err_grid(k) = sqrt(err_r^2 + err_m^2);
    fprintf('  nt=%3d nx=%3d  KE=%.6f  L2=%.2e\n', nt_vals(k), nx_vals(k), ke_grid(k), err_grid(k));
end
fprintf('Exact W_2^2 = %.6f\n', ke_exact_ot);

figure('Name', 'ladmm_ot_bimodal_to_bimodal__grid_refinement', 'Position', [100,1280,1100,340]);
subplot(1,2,1);
semilogx(nx_vals, ke_grid, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
yline(ke_exact_ot,'r--','LineWidth',1.5,'Label',sprintf('W_2^2=%.4f',ke_exact_ot));
xlabel('$n_x$','Interpreter','latex','FontSize',13); ylabel('Kinetic energy','Interpreter','latex');
title('KE $\to W_2^2=0.0225$','Interpreter','latex');
legend('LADMM KE','$W_2^2$','Interpreter','latex','Location','best');
grid on;

subplot(1,2,2);
loglog(nx_vals, err_grid, 'ks-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
loglog(nx_vals, err_grid(1)*(nx_vals(1)./nx_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_x$','Interpreter','latex','FontSize',13); ylabel('$L^2$ error','Interpreter','latex');
title('Grid refinement: $L^2$ error','Interpreter','latex');
legend('LADMM','$\mathcal{O}(h^2)$','Interpreter','latex','Location','southwest');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_ot_bimodal_to_bimodal__grid_refinement__%s.png', ftag_main)));

fprintf('\nFigures saved to: %s\n', fig_dir);
