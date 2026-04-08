%% test_ot_uniform.m  [discretize_first / LADMM]
%
% OT / SB between Uniform[0.2,0.4] -> Uniform[0.6,0.8].
%
% Three runs compared side by side:
%   (A) Sharp uniform,     eps=0    -- shows Gibbs oscillations (DCT artifact)
%   (B) Mollified uniform, eps=0    -- sigmoid-smoothed edges, no Gibbs
%   (C) Sharp uniform,     eps=1e-3 -- tiny diffusion suppresses oscillations
%
% Exact OT solution (sharp): rho(t,x)=Uniform[0.2+0.4t,0.4+0.4t], KE=0.16
% Mollified KE is slightly above 0.16 (O(sigma/width) ~ 5% deviation).

clear; clc; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config ---
cfg        = cfg_ladmm_gaussian();
cfg.nt     = 256;
cfg.nx     = 256;

ke_exact   = 0.16;
t_fracs    = [0.1, 0.25, 0.5, 0.75, 0.9];
colors     = parula(numel(t_fracs));

ftag_main  = sprintf('nt%d_nx%d_gam%g_tau%g', cfg.nt, cfg.nx, cfg.gamma, cfg.tau);

%% --- Problem definitions ---
prob_sharp = prob_uniform();   % step function (from optimize_first/problems/)

sigma_moll = 0.01;
soft       = @(x, c) 1 ./ (1 + exp(-(x - c) / sigma_moll));
prob_moll  = prob_sharp;
prob_moll.name      = 'uniform_mollified';
prob_moll.rho0_func = @(xx) soft(xx, 0.2) .* (1 - soft(xx, 0.4));
prob_moll.rho1_func = @(xx) soft(xx, 0.6) .* (1 - soft(xx, 0.8));

%% --- Run (A): sharp, eps=0 ---
cfg_A = cfg;   cfg_A.vareps = 0;
prob_A = setup_problem(cfg_A, prob_sharp);
fprintf('(A) Sharp uniform, eps=0  nt=%d nx=%d ...\n', cfg.nt, cfg.nx);
result_A = cfg_A.pipeline(cfg_A, prob_A);
fprintf('    iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    result_A.iters, result_A.converged, result_A.error, result_A.walltime);
ke_A = compute_objective(result_A.rho_stag, result_A.mx_stag, prob_A) / prob_A.dx;

%% --- Run (B): mollified, eps=0 ---
cfg_B = cfg;   cfg_B.vareps = 0;
prob_B = setup_problem(cfg_B, prob_moll);
fprintf('(B) Mollified uniform (sigma=%.3g), eps=0 ...\n', sigma_moll);
result_B = cfg_B.pipeline(cfg_B, prob_B);
fprintf('    iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    result_B.iters, result_B.converged, result_B.error, result_B.walltime);
ke_B = compute_objective(result_B.rho_stag, result_B.mx_stag, prob_B) / prob_B.dx;

%% --- Run (C): sharp, eps sweep ---
eps_vals = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1];
ke_C     = zeros(numel(eps_vals), 1);
results_C = cell(numel(eps_vals), 1);
for j = 1:numel(eps_vals)
    cfg_j = cfg;   cfg_j.vareps = eps_vals(j);
    prob_j = setup_problem(cfg_j, prob_sharp);
    fprintf('(C) Sharp uniform, eps=%.0e ...\n', eps_vals(j));
    results_C{j} = cfg_j.pipeline(cfg_j, prob_j);
    fprintf('    iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
        results_C{j}.iters, results_C{j}.converged, results_C{j}.error, results_C{j}.walltime);
    ke_C(j) = compute_objective(results_C{j}.rho_stag, results_C{j}.mx_stag, prob_j) / prob_j.dx;
end

[rho_ana, ~] = analytical_ot_uniform(prob_A);

fprintf('\n%-40s  KE\n', 'Run');
fprintf('%s\n', repmat('-',1,52));
fprintf('(A) Sharp,     eps=0          (exact=0.16)  %.6f\n', ke_A);
fprintf('(B) Mollified, eps=0          (exact~0.16)  %.6f\n', ke_B);
for j = 1:numel(eps_vals)
    fprintf('(C) Sharp,     eps=%.0e                      %.6f\n', eps_vals(j), ke_C(j));
end

%% --- Shared grid info ---
nt  = prob_A.nt;   ntm = nt-1;   dt = prob_A.dt;
nx  = prob_A.nx;   dx  = prob_A.dx;   %#ok<NASGU>
xx  = prob_A.xx;

%% --- Figure 1: sharp OT (Gibbs) ---
figure('Name', 'ladmm_ot_uniform__sharp_ot', 'Position', [100,100,680,400]);
hold on;
plot_density_evolution(xx, rho_ana, result_A.rho_stag, t_fracs, nt, ntm, dt, nx, colors);
title(sprintf('(A) Sharp uniform, $\\varepsilon=0$  KE=%.4f vs %.4f\n(solid=exact sliding box, dots=LADMM)', ...
              ke_A, ke_exact), 'Interpreter','latex');
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_ot_uniform__sharp_ot__%s.png', ftag_main)));

%% --- Figure 2: mollified OT ---
figure('Name', 'ladmm_ot_uniform__mollified_ot', 'Position', [800,100,680,400]);
hold on;
plot_density_evolution(xx, rho_ana, result_B.rho_stag, t_fracs, nt, ntm, dt, nx, colors);
title(sprintf('(B) Mollified uniform ($\\sigma=%.3g$), $\\varepsilon=0$  KE=%.4f\n(solid=sharp exact, dots=LADMM mollified)', ...
              sigma_moll, ke_B), 'Interpreter','latex');
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_ot_uniform__mollified_ot__%s.png', ftag_main)));

%% --- Figure 3: eps sweep at t=0.5 ---
t_show  = 0.5;
k_show  = max(1, min(ntm, round(t_show * nt)));
colors_eps = lines(numel(eps_vals));

figure('Name', 'ladmm_ot_uniform__eps_sweep', 'Position', [100,540,720,400]);
hold on;
plot(xx, rho_ana(k_show,:), 'k-', 'LineWidth', 2.0, 'DisplayName', 'Exact OT ($\varepsilon=0$)');
plot(xx, result_A.rho_stag(k_show,:), 'k--', 'LineWidth', 1.5, ...
     'DisplayName', sprintf('Sharp LADMM $\\varepsilon=0$  KE=%.4f', ke_A));
for j = 1:numel(eps_vals)
    plot(xx, results_C{j}.rho_stag(k_show,:), '-', 'Color', colors_eps(j,:), ...
         'LineWidth', 1.5, ...
         'DisplayName', sprintf('$\\varepsilon=%.0e$  KE=%.4f', eps_vals(j), ke_C(j)));
end
xlabel('$x$','Interpreter','latex');
ylabel(sprintf('$\\rho(t=%.1f,x)$', t_show),'Interpreter','latex');
title(sprintf('(C) Sharp uniform: $\\varepsilon$ sweep at $t=%.1f$\n(increasing $\\varepsilon$ smooths Gibbs oscillations)', t_show), ...
      'Interpreter','latex');
legend('Interpreter','latex','Location','best','FontSize',8);
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_ot_uniform__eps_sweep__%s.png', ftag_main)));

fprintf('\nFigures saved to: %s\n', fig_dir);

%% --- Local functions ---
function plot_density_evolution(xx, rho_exact, rho_num, t_fracs, nt, ntm, dt, nx, colors)
    stride = max(1, floor(nx/60));
    h_leg  = gobjects(numel(t_fracs), 1);
    for p = 1:numel(t_fracs)
        k = max(1, min(ntm, round(t_fracs(p)*nt)));
        idx = 1:stride:nx;
        h_leg(p) = plot(xx, rho_exact(k,:), '-', 'Color', colors(p,:), ...
                        'LineWidth', 1.5, 'DisplayName', sprintf('$t=%.2f$', k*dt));
        plot(xx(idx), rho_num(k,idx), 'o', 'Color', colors(p,:), ...
             'MarkerSize', 4, 'MarkerFaceColor', colors(p,:), 'HandleVisibility','off');
    end
    legend(h_leg, 'Interpreter','latex','Location','best','FontSize',8);
    xlabel('$x$','Interpreter','latex');
    ylabel('$\rho$','Interpreter','latex');
    grid on;
end
