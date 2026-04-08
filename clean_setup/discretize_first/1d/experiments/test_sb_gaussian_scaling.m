%% test_sb_gaussian_scaling.m  [discretize_first / LADMM]
%
% Schrödinger bridge: narrow Gaussian -> wide Gaussian.
%
%   rho0 = N(0.3, 0.04^2),  rho1 = N(0.7, 0.08^2)
%
% Analytical SB solution available via analytical_sb_gaussian_general.
% Checks:
%   1. rho_stag and mx_stag vs analytical at multiple times.
%   2. L2 error vs time and grid refinement (expect O(h^2)).
%   3. eps->0 limit agrees with McCann displacement interpolation.

clear; clc; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config ---
vareps  = 1e-1;
nt_main = 64;
nx_main = 128;

cfg          = cfg_ladmm_gaussian();
cfg.vareps   = vareps;
cfg.nt       = nt_main;
cfg.nx       = nx_main;

prob_def = prob_gaussian_scaling();
problem  = setup_problem(cfg, prob_def);

ftag_main = sprintf('nt%d_nx%d_gam%g_tau%g_eps%g', ...
    cfg.nt, cfg.nx, cfg.gamma, cfg.tau, cfg.vareps);

%% --- Run ---
fprintf('Running LADMM SB: N(%.2f,%.4g)->N(%.2f,%.4g)  eps=%.2g  nt=%d nx=%d ...\n', ...
    prob_def.mu0, prob_def.sigma0^2, prob_def.mu1, prob_def.sigma1^2, ...
    vareps, nt_main, nx_main);
result = cfg.pipeline(cfg, problem);
fprintf('  iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

[rho_ana, mx_ana] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, vareps);

nt  = problem.nt;   ntm = nt-1;   dt = problem.dt;
nx  = problem.nx;   dx  = problem.dx;   nxm = nx-1;
xx  = problem.xx;   x_mx = (1:nxm)*dx;
t_rho = (1:ntm)'*dt;

%% --- Figure 1: density evolution ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
colors  = parula(numel(t_fracs));

figure('Name', sprintf('ladmm_sb_gaussian_scaling__density__eps%g', vareps), ...
       'Position', [100,100,720,400]);
hold on;
for p = 1:numel(t_fracs)
    k = max(1, min(ntm, round(t_fracs(p)*nt)));
    stride = max(1, floor(nx/60));   idx = 1:stride:nx;
    plot(xx, rho_ana(k,:), '-', 'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result.rho_stag(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end
leg = cell(2*numel(t_fracs),1);
for p = 1:numel(t_fracs)
    k = max(1,min(ntm,round(t_fracs(p)*nt)));
    leg{2*p-1} = sprintf('Exact  t=%.2f', k*dt);
    leg{2*p}   = sprintf('LADMM  t=%.2f', k*dt);
end
legend(leg, 'Location', 'best', 'FontSize', 7);
xlabel('$x$','Interpreter','latex'); ylabel('$\rho$','Interpreter','latex');
title(sprintf('LADMM SB: $N(0.3,0.04^2)\\to N(0.7,0.08^2)$  $\\varepsilon=%.2g$', vareps), ...
      'Interpreter','latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_sb_gaussian_scaling__density__%s.png', ftag_main)));

%% --- Figure 2: L2 error vs time ---
err_rho = sqrt(dx * sum((result.rho_stag - rho_ana).^2, 2));
err_mx  = sqrt(dx * sum((result.mx_stag  - mx_ana ).^2, 2));

t_mx = ((1:nt)' - 0.5) * dt;

figure('Name', 'ladmm_sb_gaussian_scaling__l2error_vs_time', 'Position', [100,520,620,300]);
plot(t_rho, err_rho, 'b-', 'LineWidth', 1.5); hold on;
plot(t_mx, err_mx, 'r-', 'LineWidth', 1.5);
xlabel('$t$','Interpreter','latex');
ylabel('$L^2$ error','Interpreter','latex');
legend('$\|\rho_\mathrm{LADMM}-\rho_\mathrm{exact}\|$', ...
       '$\|m_\mathrm{LADMM}-m_\mathrm{exact}\|$', 'Interpreter','latex','Location','best');
title(sprintf('$L^2$ error vs time  ($\\varepsilon=%.2g$)', vareps),'Interpreter','latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_sb_gaussian_scaling__l2error_vs_time__%s.png', ftag_main)));

%% --- Figure 3: eps->0 limit ---
cfg_ot = cfg;   cfg_ot.vareps = 1e-6;
fprintf('Running eps=1e-6 (near-OT) ...\n');
result_ot = cfg_ot.pipeline(cfg_ot, problem);
[rho_ot, ~] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, 1e-6);
err_ot = sqrt(dx * sum((result_ot.rho_stag - rho_ot).^2, 2));

figure('Name', 'ladmm_sb_gaussian_scaling__ot_limit', 'Position', [730,520,600,300]);
plot(t_rho, err_ot, 'r-', 'LineWidth', 1.5); hold on;
plot(t_rho, err_rho, 'b-', 'LineWidth', 1.5);
xlabel('$t$','Interpreter','latex'); ylabel('$L^2$ error','Interpreter','latex');
legend('$\varepsilon=10^{-6}$ (near-OT)', sprintf('$\\varepsilon=%.2g$ (SB)', vareps), ...
       'Interpreter','latex','Location','best');
title('$L^2$ error: near-OT vs SB limits','Interpreter','latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_sb_gaussian_scaling__ot_limit__%s.png', ftag_main)));

%% --- Grid refinement ---
nt_vals = [16, 32, 64];   nx_vals = [32, 64, 128];   n_grids = numel(nt_vals);
ke_admm = zeros(n_grids,1);   ke_ana_g = zeros(n_grids,1);
err_total = zeros(n_grids,1);

fprintf('\nGrid refinement (eps=%.2g):\n', vareps);
for k = 1:n_grids
    cfg_k = cfg;   cfg_k.nt = nt_vals(k);   cfg_k.nx = nx_vals(k);
    prob_k = setup_problem(cfg_k, prob_def);
    r = cfg_k.pipeline(cfg_k, prob_k);
    [rho_a, mx_a] = analytical_sb_gaussian_general(prob_k, ...
        prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, vareps);
    ke_admm(k)   = compute_objective(r.rho_stag, r.mx_stag, prob_k) / prob_k.dx;
    ke_ana_g(k)  = compute_objective(rho_a, mx_a, prob_k) / prob_k.dx;
    err_r = norm(r.rho_stag(:) - rho_a(:)) * sqrt(prob_k.dt * prob_k.dx);
    err_m = norm(r.mx_stag(:)  - mx_a(:))  * sqrt(prob_k.dt * prob_k.dx);
    err_total(k) = sqrt(err_r^2 + err_m^2);
    fprintf('  nt=%3d nx=%3d  KE=%.6f  L2=%.2e\n', nt_vals(k), nx_vals(k), ke_admm(k), err_total(k));
end

figure('Name', 'ladmm_sb_gaussian_scaling__grid_refinement', 'Position', [100,870,1100,340]);
subplot(1,2,1);
loglog(nx_vals, err_total, 'ks-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
loglog(nx_vals, err_total(1)*(nx_vals(1)./nx_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_x$','Interpreter','latex','FontSize',13);
ylabel('$L^2$ error','Interpreter','latex');
title('Grid refinement: $L^2$ error','Interpreter','latex');
legend('LADMM','$\mathcal{O}(h^2)$','Interpreter','latex','Location','southwest');
grid on;

subplot(1,2,2);
semilogx(nx_vals, ke_admm, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
semilogx(nx_vals, ke_ana_g, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('$n_x$','Interpreter','latex','FontSize',13);
ylabel('Kinetic energy','Interpreter','latex');
title(sprintf('KE convergence  ($\\varepsilon=%.2g$)', vareps),'Interpreter','latex');
legend('LADMM','Analytical','Interpreter','latex','Location','best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_sb_gaussian_scaling__grid_refinement__%s.png', ftag_main)));

fprintf('\n%-12s  %-12s  %-12s  %-12s\n', 'nt x nx', 'KE (LADMM)', 'KE (exact)', 'L2 error');
fprintf('%s\n', repmat('-',1,54));
for k = 1:n_grids
    fprintf('%3d x %-6d  %-12.6f  %-12.6f  %-12.2e\n', ...
            nt_vals(k), nx_vals(k), ke_admm(k), ke_ana_g(k), err_total(k));
end
fprintf('\nFigures saved to: %s\n', fig_dir);
