%% test_stationary.m  [discretize_first / LADMM]
%
% Trivial transport: rho0 = rho1 = N(0.5, 0.05^2).
%
% For eps=0 (OT): exact solution is rho(t,x)=rho0(x), m=0, KE=0.
%   Any nonzero KE or momentum is a solver artefact.
%
% For eps>0 (SB): the "looping" Gaussian SB — distribution spreads at
%   mid-time then contracts back.  Analytical: analytical_sb_gaussian_general.
%
% KE(eps) should be monotone increasing from 0 as eps grows.

clear; clc; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config ---
cfg    = cfg_ladmm_gaussian();
cfg.nt = 256;   cfg.nx = 128;

prob_def = prob_stationary();
problem  = setup_problem(cfg, prob_def);

ftag_base = sprintf('nt%d_nx%d_gam%g_tau%g', cfg.nt, cfg.nx, cfg.gamma, cfg.tau);

nt  = problem.nt;   ntm = nt-1;   dt = problem.dt;
nx  = problem.nx;   dx  = problem.dx;
xx  = problem.xx;   t_rho = (1:ntm)'*dt;

%% --- OT (eps=0) ---
cfg_ot = cfg;   cfg_ot.vareps = 0;
fprintf('Running LADMM OT (eps=0): stationary ...\n');
result_ot = cfg_ot.pipeline(cfg_ot, problem);
fprintf('  iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    result_ot.iters, result_ot.converged, result_ot.error, result_ot.walltime);

ke_ot   = compute_objective(result_ot.rho_stag, result_ot.mx_stag, problem) / dx;
mx_max  = max(abs(result_ot.mx_stag(:)));
rho_drift = sqrt(dx * dt * sum(sum((result_ot.rho_stag - repmat(problem.rho0, ntm, 1)).^2)));

fprintf('\n--- OT (eps=0) checks ---\n');
fprintf('  KE                   = %.2e   (should be ~0)\n', ke_ot);
fprintf('  max|m(t,x)|          = %.2e   (should be ~0)\n', mx_max);
fprintf('  ||rho(t)-rho0||_L2   = %.2e   (should be ~0)\n', rho_drift);

%% --- SB (eps>0) ---
eps_sb = 1e-1;
cfg_sb = cfg;   cfg_sb.vareps = eps_sb;
fprintf('\nRunning LADMM SB (eps=%.2g): looping Gaussian ...\n', eps_sb);
result_sb = cfg_sb.pipeline(cfg_sb, problem);
fprintf('  iters=%d  converged=%d  error=%.2e\n', ...
    result_sb.iters, result_sb.converged, result_sb.error);

[rho_ana, mx_ana] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, eps_sb);

ke_sb     = compute_objective(result_sb.rho_stag, result_sb.mx_stag, problem) / dx;
ke_ana_sb = compute_objective(rho_ana, mx_ana, problem) / dx;
err_rho   = sqrt(dx * sum((result_sb.rho_stag - rho_ana).^2, 2));

fprintf('\n--- SB (eps=%.2g) checks ---\n', eps_sb);
fprintf('  KE (LADMM)       = %.6f\n', ke_sb);
fprintf('  max L2 err rho   = %.2e\n', max(err_rho));

%% --- Figure 1: density OT vs SB ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
colors  = parula(numel(t_fracs));

figure('Name', sprintf('ladmm_stationary__density__eps_ot_vs_sb%g', eps_sb), ...
       'Position', [100,100,1100,420]);

subplot(1,2,1);
hold on;
h_leg = gobjects(numel(t_fracs), 1);
for p = 1:numel(t_fracs)
    k = max(1,min(ntm,round(t_fracs(p)*nt)));
    stride = max(1,floor(nx/60));   idx = 1:stride:nx;
    h_leg(p) = plot(xx, problem.rho0, '-', 'Color', colors(p,:), 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('$t=%.2f$', t_fracs(p)));
    plot(xx(idx), result_ot.rho_stag(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:), 'HandleVisibility', 'off');
end
legend(h_leg, 'Interpreter','latex','Location','best','FontSize',8);
xlabel('$x$','Interpreter','latex'); ylabel('$\rho$','Interpreter','latex');
title(sprintf('OT ($\\varepsilon=0$),  KE=%.2e\n(solid=exact, dots=LADMM)', ke_ot), ...
      'Interpreter','latex');
grid on;

subplot(1,2,2);
hold on;
h_leg2 = gobjects(numel(t_fracs), 1);
for p = 1:numel(t_fracs)
    k = max(1,min(ntm,round(t_fracs(p)*nt)));
    stride = max(1,floor(nx/60));   idx = 1:stride:nx;
    h_leg2(p) = plot(xx, rho_ana(k,:), '-', 'Color', colors(p,:), 'LineWidth', 1.5, ...
                     'DisplayName', sprintf('$t=%.2f$', t_fracs(p)));
    plot(xx(idx), result_sb.rho_stag(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:), 'HandleVisibility', 'off');
end
legend(h_leg2, 'Interpreter','latex','Location','best','FontSize',8);
xlabel('$x$','Interpreter','latex'); ylabel('$\rho$','Interpreter','latex');
title(sprintf('Looping SB ($\\varepsilon=%.2g$): spread then contract\n(solid=exact, dots=LADMM)', eps_sb), ...
      'Interpreter','latex');
grid on;
sgtitle('LADMM stationary: $\rho_0=\rho_1=\mathcal{N}(0.5,0.05^2)$', ...
        'Interpreter','latex','FontSize',13);
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_stationary__density__%s.png', ftag_base)));

%% --- Figure 2: distribution width vs time ---
sigma_ot = zeros(ntm,1);   sigma_sb = zeros(ntm,1);   sigma_an = zeros(ntm,1);
for k = 1:ntm
    mass_ot = sum(result_ot.rho_stag(k,:)) * dx;
    mass_sb = sum(result_sb.rho_stag(k,:)) * dx;
    mass_an = sum(rho_ana(k,:)) * dx;
    mu_ot = sum(xx .* result_ot.rho_stag(k,:)) * dx / mass_ot;
    mu_sb = sum(xx .* result_sb.rho_stag(k,:)) * dx / mass_sb;
    mu_an = sum(xx .* rho_ana(k,:)) * dx / mass_an;
    sigma_ot(k) = sqrt(sum((xx-mu_ot).^2 .* result_ot.rho_stag(k,:)) * dx / mass_ot);
    sigma_sb(k) = sqrt(sum((xx-mu_sb).^2 .* result_sb.rho_stag(k,:)) * dx / mass_sb);
    sigma_an(k) = sqrt(sum((xx-mu_an).^2 .* rho_ana(k,:)) * dx / mass_an);
end

figure('Name', 'ladmm_stationary__width_vs_time', 'Position', [100,540,700,320]);
plot(t_rho, sigma_ot, 'b-', 'LineWidth', 1.5); hold on;
plot(t_rho, sigma_sb, 'r-', 'LineWidth', 1.5);
plot(t_rho, sigma_an, 'r--', 'LineWidth', 1.5);
yline(prob_def.sigma0, 'k:', 'LineWidth', 1.2, ...
      'Label', sprintf('\\sigma_0=\\sigma_1=%.2g  (boundary std)', prob_def.sigma0), ...
      'LabelHorizontalAlignment', 'left', 'FontSize', 9);
xlabel('$t$','Interpreter','latex');
ylabel('$\sigma(t)$','Interpreter','latex');
legend('OT ($\varepsilon=0$)', sprintf('SB ($\\varepsilon=%.2g$)',eps_sb), ...
       'Analytical SB','Interpreter','latex','Location','best');
title('Width vs time (symmetric about $t=0.5$ for SB)','Interpreter','latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_stationary__width_vs_time__%s.png', ftag_base)));

%% --- Figure 3: L2 error vs time (SB) ---
figure('Name', sprintf('ladmm_stationary__l2error_sb__eps%g', eps_sb), ...
       'Position', [820,540,560,320]);
plot(t_rho, err_rho, 'b-', 'LineWidth', 1.5);
xlabel('$t$','Interpreter','latex');
ylabel('$\|\rho_\mathrm{LADMM}-\rho_\mathrm{exact}\|_{L^2(x)}$','Interpreter','latex');
title(sprintf('$L^2$ error: looping SB  ($\\varepsilon=%.2g$)', eps_sb),'Interpreter','latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_stationary__l2error_sb__eps%g__%s.png', eps_sb, ftag_base)));

%% --- Figure 4: KE vs eps ---
eps_vals = [0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1];
ke_eps   = zeros(numel(eps_vals),1);
ke_ana_eps = zeros(numel(eps_vals),1);

fprintf('\neps sweep:\n');
for j = 1:numel(eps_vals)
    cfg_j = cfg;   cfg_j.vareps = eps_vals(j);
    r_j   = cfg_j.pipeline(cfg_j, problem);
    ke_eps(j) = compute_objective(r_j.rho_stag, r_j.mx_stag, problem) / dx;
    if eps_vals(j) == 0
        ke_ana_eps(j) = 0;
    else
        [ra, ma] = analytical_sb_gaussian_general(problem, ...
            prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, eps_vals(j));
        ke_ana_eps(j) = compute_objective(ra, ma, problem) / dx;
    end
    fprintf('  eps=%.0e  KE=%.6f  KE_ana=%.6f\n', eps_vals(j), ke_eps(j), ke_ana_eps(j));
end

figure('Name', 'ladmm_stationary__ke_vs_eps', 'Position', [100,870,600,340]);
semilogx(eps_vals(2:end), ke_eps(2:end), 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
semilogx(eps_vals(2:end), ke_ana_eps(2:end), 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
yline(ke_eps(1), 'k--', sprintf('OT KE=%.2e', ke_eps(1)), 'LineWidth', 1.2);
xlabel('$\varepsilon$','Interpreter','latex','FontSize',13);
ylabel('Kinetic energy','Interpreter','latex');
title('Stationary: KE increases with $\varepsilon$ (looping cost)','Interpreter','latex');
legend('LADMM','Analytical','Interpreter','latex','Location','best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_stationary__ke_vs_eps__%s.png', ftag_base)));

fprintf('\nFigures saved to: %s\n', fig_dir);
