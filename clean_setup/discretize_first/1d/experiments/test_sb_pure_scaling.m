%% test_sb_pure_scaling.m  [discretize_first / LADMM]
%
% Pure spreading: N(0.5, 0.03^2) -> N(0.5, 0.10^2).
%
% No translation; velocity is purely radial (antisymmetric about x=0.5).
%
% Checks:
%   1. Centre of mass stays at 0.5 for all t.
%   2. rho_stag and mx_stag vs analytical_sb_gaussian_general.
%   3. m(t,x) is antisymmetric about x=0.5.
%   4. For eps=0: KE = W_2^2 = (sigma1-sigma0)^2 = 0.07^2 = 0.0049.
%   5. Grid-refinement O(h^2) convergence.

clear; clc; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config ---
vareps = 1e-1;

cfg        = cfg_ladmm_gaussian();
cfg.nt     = 128;   cfg.nx = 128;
cfg.vareps = vareps;

prob_def = prob_gaussian_pure_scaling();
problem  = setup_problem(cfg, prob_def);

ftag_main = sprintf('nt%d_nx%d_gam%g_tau%g_eps%g', ...
    cfg.nt, cfg.nx, cfg.gamma, cfg.tau, cfg.vareps);

nt  = problem.nt;   ntm = nt-1;   dt = problem.dt;
nx  = problem.nx;   dx  = problem.dx;   nxm = nx-1;
xx  = problem.xx;   x_mx = (1:nxm)*dx;
t_rho = (1:ntm)'*dt;

%% --- Run SB and OT ---
fprintf('Running LADMM SB (eps=%.2g): pure scaling ...\n', vareps);
result_sb = cfg.pipeline(cfg, problem);
fprintf('  iters=%d  converged=%d  error=%.2e\n', ...
    result_sb.iters, result_sb.converged, result_sb.error);

cfg_ot = cfg;   cfg_ot.vareps = 0;
fprintf('Running LADMM OT (eps=0): pure scaling ...\n');
result_ot = cfg_ot.pipeline(cfg_ot, problem);
fprintf('  iters=%d  converged=%d  error=%.2e\n', ...
    result_ot.iters, result_ot.converged, result_ot.error);

[rho_ana_sb, mx_ana_sb] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, vareps);
[rho_ana_ot, ~] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, 1e-8);

%% --- Checks ---
com_sb = sum(xx .* result_sb.rho_stag, 2) / sum(result_sb.rho_stag(1,:));
com_ot = sum(xx .* result_ot.rho_stag, 2) / sum(result_ot.rho_stag(1,:));
ke_ot  = compute_objective(result_ot.rho_stag, result_ot.mx_stag, problem) / dx;
ke_sb  = compute_objective(result_sb.rho_stag, result_sb.mx_stag, problem) / dx;
ke_ana_sb_val = compute_objective(rho_ana_sb, mx_ana_sb, problem) / dx;
ke_exact_ot   = (prob_def.sigma1 - prob_def.sigma0)^2;

nxm_half = floor(nxm/2);
m_flip   = fliplr(result_sb.mx_stag(:, 1:nxm));
antisym_err = max(abs(result_sb.mx_stag(:,1:nxm_half) + m_flip(:,nxm+1-nxm_half:nxm)));

fprintf('\n--- Checks ---\n');
fprintf('  CoM SB: max|CoM-0.5| = %.2e  (should be ~0)\n', max(abs(com_sb-0.5)));
fprintf('  CoM OT: max|CoM-0.5| = %.2e  (should be ~0)\n', max(abs(com_ot-0.5)));
fprintf('  OT KE (LADMM)  = %.6f,  W_2^2 = %.6f\n', ke_ot, ke_exact_ot);
fprintf('  SB KE (LADMM)  = %.6f,  analytical = %.6f\n', ke_sb, ke_ana_sb_val);
fprintf('  max|m(t,x)+m(t,1-x)| = %.2e  (should be ~0)\n', antisym_err);

%% --- Figure 1: density OT vs SB ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
colors  = parula(numel(t_fracs));

figure('Name', sprintf('ladmm_sb_pure_scaling__density__eps%g', vareps), ...
       'Position', [100,100,1100,400]);

subplot(1,2,1);
hold on;
for p = 1:numel(t_fracs)
    k = max(1,min(ntm,round(t_fracs(p)*nt)));
    stride = max(1,floor(nx/60));   idx = 1:stride:nx;
    plot(xx, rho_ana_ot(k,:), '-', 'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result_ot.rho_stag(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end
xlabel('$x$','Interpreter','latex'); ylabel('$\rho$','Interpreter','latex');
title(sprintf('OT ($\\varepsilon\\approx0$),  KE=%.4f vs %.4f', ke_ot, ke_exact_ot),'Interpreter','latex');
grid on;

subplot(1,2,2);
hold on;
for p = 1:numel(t_fracs)
    k = max(1,min(ntm,round(t_fracs(p)*nt)));
    stride = max(1,floor(nx/60));   idx = 1:stride:nx;
    plot(xx, rho_ana_sb(k,:), '-', 'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result_sb.rho_stag(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end
xlabel('$x$','Interpreter','latex'); ylabel('$\rho$','Interpreter','latex');
title(sprintf('SB ($\\varepsilon=%.2g$),  KE=%.4f vs %.4f', vareps, ke_sb, ke_ana_sb_val),'Interpreter','latex');
grid on;
sgtitle('LADMM pure scaling: $\mathcal{N}(0.5,0.03^2)\to\mathcal{N}(0.5,0.10^2)$  (solid=exact, dots=LADMM)', ...
        'Interpreter','latex','FontSize',12);
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_sb_pure_scaling__density__%s.png', ftag_main)));

%% --- Figure 2: momentum at t=0.5 ---
k_mid = round(0.5*nt);
figure('Name', 'ladmm_sb_pure_scaling__momentum_at_t05', 'Position', [100,530,700,320]);
plot(x_mx, mx_ana_sb(k_mid,:), 'r-', 'LineWidth', 1.5); hold on;
plot(x_mx, result_sb.mx_stag(k_mid,:), 'b--', 'LineWidth', 1.5);
xline(0.5,'k:','LineWidth',1.0);
xlabel('$x$','Interpreter','latex'); ylabel('$m(t=0.5,x)$','Interpreter','latex');
title('Momentum at $t=0.5$ — antisymmetric about $x=0.5$','Interpreter','latex');
legend('Analytical','LADMM','Interpreter','latex','Location','best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_sb_pure_scaling__momentum_at_t05__%s.png', ftag_main)));

%% --- Figure 3: centre of mass vs time ---
figure('Name', 'ladmm_sb_pure_scaling__com_vs_time', 'Position', [820,530,500,300]);
plot(t_rho, com_sb, 'b-', 'LineWidth', 1.5); hold on;
plot(t_rho, com_ot, 'r-', 'LineWidth', 1.5);
yline(0.5,'k--','LineWidth',1.2);
xlabel('$t$','Interpreter','latex'); ylabel('$\bar{x}(t)$','Interpreter','latex');
title('Centre of mass (must stay at 0.5)','Interpreter','latex');
legend(sprintf('SB $\\varepsilon=%.2g$',vareps),'OT','Interpreter','latex','Location','best');
ylim([0.45,0.55]);   grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_sb_pure_scaling__com_vs_time__%s.png', ftag_main)));

%% --- Grid refinement ---
nt_vals = [16, 32, 64, 128];   nx_vals = [32, 64, 128, 256];   n_grids = numel(nt_vals);
ke_grid  = zeros(n_grids,1);   err_grid = zeros(n_grids,1);

fprintf('\nGrid refinement (eps=%.2g):\n', vareps);
for k = 1:n_grids
    cfg_k = cfg;   cfg_k.nt = nt_vals(k);   cfg_k.nx = nx_vals(k);
    prob_k = setup_problem(cfg_k, prob_def);
    r = cfg_k.pipeline(cfg_k, prob_k);
    [rho_a, mx_a] = analytical_sb_gaussian_general(prob_k, ...
        prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, vareps);
    ke_grid(k)  = compute_objective(r.rho_stag, r.mx_stag, prob_k) / prob_k.dx;
    err_r = norm(r.rho_stag(:) - rho_a(:)) * sqrt(prob_k.dt * prob_k.dx);
    err_m = norm(r.mx_stag(:)  - mx_a(:))  * sqrt(prob_k.dt * prob_k.dx);
    err_grid(k) = sqrt(err_r^2 + err_m^2);
    fprintf('  nt=%3d nx=%3d  KE=%.6f  L2=%.2e\n', nt_vals(k), nx_vals(k), ke_grid(k), err_grid(k));
end

figure('Name', 'ladmm_sb_pure_scaling__grid_refinement', 'Position', [100,870,1100,350]);
subplot(1,2,1);
loglog(nx_vals, err_grid, 'ks-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
loglog(nx_vals, err_grid(1)*(nx_vals(1)./nx_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_x$','Interpreter','latex','FontSize',13); ylabel('$L^2$ error','Interpreter','latex');
title('Grid refinement: $L^2$ error','Interpreter','latex');
legend('LADMM','$\mathcal{O}(h^2)$','Interpreter','latex','Location','southwest');
grid on;

subplot(1,2,2);
semilogx(nx_vals, ke_grid, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
yline(ke_ana_sb_val,'r--','LineWidth',1.5);
xlabel('$n_x$','Interpreter','latex','FontSize',13); ylabel('Kinetic energy','Interpreter','latex');
title(sprintf('KE convergence  ($\\varepsilon=%.2g$)', vareps),'Interpreter','latex');
legend('LADMM','Analytical','Interpreter','latex','Location','best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_sb_pure_scaling__grid_refinement__%s.png', ftag_main)));

fprintf('\nFigures saved to: %s\n', fig_dir);
