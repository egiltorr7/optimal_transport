%% test_time_reversal.m  [discretize_first / LADMM]
%
% Verify time-reversal symmetry of the Schrödinger bridge.
%
%   rho_fwd(t, x)   =  rho_bwd(1-t, x)
%   m_fwd(t, x)     = -m_bwd(1-t, x)
%   KE_fwd           =  KE_bwd
%
% Tested on:
%   (A) gaussian_scaling: N(0.3,0.04^2) -> N(0.7,0.08^2)   [has analytical SB]
%   (B) bimodal -> unimodal                                  [no analytical]
%
% Symmetry error should decay O(h^2) with grid refinement.

clear; clc; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config ---
cfg        = cfg_ladmm_gaussian();
cfg.nt     = 128;   cfg.nx = 128;
cfg.vareps = 1e-1;

ftag_main = sprintf('nt%d_nx%d_gam%g_tau%g_eps%g', ...
    cfg.nt, cfg.nx, cfg.gamma, cfg.tau, cfg.vareps);

%% -----------------------------------------------------------------------
%% Part A: gaussian_scaling
%% -----------------------------------------------------------------------
prob_fwd = prob_gaussian_scaling();

prob_bwd           = prob_fwd;
prob_bwd.rho0_func = prob_fwd.rho1_func;
prob_bwd.rho1_func = prob_fwd.rho0_func;
prob_bwd.mu0 = prob_fwd.mu1;   prob_bwd.sigma0 = prob_fwd.sigma1;
prob_bwd.mu1 = prob_fwd.mu0;   prob_bwd.sigma1 = prob_fwd.sigma0;
prob_bwd.name = 'gaussian_scaling_rev';

problem_fwd = setup_problem(cfg, prob_fwd);
problem_bwd = setup_problem(cfg, prob_bwd);

fprintf('=== Part A: gaussian_scaling (eps=%.2g) ===\n', cfg.vareps);
result_fwd = cfg.pipeline(cfg, problem_fwd);
result_bwd = cfg.pipeline(cfg, problem_bwd);

dx = problem_fwd.dx;   dt = problem_fwd.dt;

ke_fwd = compute_objective(result_fwd.rho_stag, result_fwd.mx_stag, problem_fwd) / dx;
ke_bwd = compute_objective(result_bwd.rho_stag, result_bwd.mx_stag, problem_bwd) / dx;

rho_bwd_flip = flipud(result_bwd.rho_stag);
mx_bwd_flip  = -flipud(result_bwd.mx_stag);

err_rho_tr = sqrt(dx * dt * sum(sum((result_fwd.rho_stag - rho_bwd_flip).^2)));
err_mx_tr  = sqrt(dx * dt * sum(sum((result_fwd.mx_stag  - mx_bwd_flip ).^2)));

fprintf('  Forward:  iters=%d  converged=%d\n', result_fwd.iters, result_fwd.converged);
fprintf('  Backward: iters=%d  converged=%d\n', result_bwd.iters, result_bwd.converged);
fprintf('\n--- Symmetry checks (A) ---\n');
fprintf('  KE fwd = %.6f,  KE bwd = %.6f,  |diff| = %.2e\n', ke_fwd, ke_bwd, abs(ke_fwd-ke_bwd));
fprintf('  ||rho_fwd - flip(rho_bwd)||_L2 = %.2e\n', err_rho_tr);
fprintf('  ||m_fwd  + flip(m_bwd)||_L2   = %.2e\n', err_mx_tr);

%% -----------------------------------------------------------------------
%% Part B: bimodal -> unimodal
%% -----------------------------------------------------------------------
fprintf('\n=== Part B: bimodal->unimodal (eps=%.2g) ===\n', cfg.vareps);

prob_bm_fwd = prob_bimodal();
prob_bm_bwd = prob_bm_fwd;
prob_bm_bwd.rho0_func = prob_bm_fwd.rho1_func;
prob_bm_bwd.rho1_func = prob_bm_fwd.rho0_func;
prob_bm_bwd.name = 'bimodal_rev';

problem_bm_fwd = setup_problem(cfg, prob_bm_fwd);
problem_bm_bwd = setup_problem(cfg, prob_bm_bwd);

result_bm_fwd = cfg.pipeline(cfg, problem_bm_fwd);
result_bm_bwd = cfg.pipeline(cfg, problem_bm_bwd);

ke_bm_fwd = compute_objective(result_bm_fwd.rho_stag, result_bm_fwd.mx_stag, problem_bm_fwd) / dx;
ke_bm_bwd = compute_objective(result_bm_bwd.rho_stag, result_bm_bwd.mx_stag, problem_bm_bwd) / dx;

rho_bm_flip = flipud(result_bm_bwd.rho_stag);
mx_bm_flip  = -flipud(result_bm_bwd.mx_stag);
err_bm_rho  = sqrt(dx*dt * sum(sum((result_bm_fwd.rho_stag - rho_bm_flip).^2)));
err_bm_mx   = sqrt(dx*dt * sum(sum((result_bm_fwd.mx_stag  - mx_bm_flip ).^2)));

fprintf('  Forward:  iters=%d  converged=%d\n', result_bm_fwd.iters, result_bm_fwd.converged);
fprintf('  Backward: iters=%d  converged=%d\n', result_bm_bwd.iters, result_bm_bwd.converged);
fprintf('\n--- Symmetry checks (B) ---\n');
fprintf('  KE fwd = %.6f,  KE bwd = %.6f,  |diff| = %.2e\n', ke_bm_fwd, ke_bm_bwd, abs(ke_bm_fwd-ke_bm_bwd));
fprintf('  ||rho_fwd - flip(rho_bwd)||_L2 = %.2e\n', err_bm_rho);
fprintf('  ||m_fwd  + flip(m_bwd)||_L2   = %.2e\n', err_bm_mx);

%% --- Figure 1: rho_fwd vs flip(rho_bwd) (A) ---
nt  = problem_fwd.nt;   ntm = nt-1;   nx = problem_fwd.nx;   xx = problem_fwd.xx;
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
colors  = parula(numel(t_fracs));

figure('Name', 'ladmm_time_reversal__rho_fwd_vs_flip_bwd', 'Position', [100,100,720,400]);
hold on;
h_leg = gobjects(numel(t_fracs), 1);
for p = 1:numel(t_fracs)
    k = max(1,min(ntm,round(t_fracs(p)*nt)));
    stride = max(1,floor(nx/60));   idx = 1:stride:nx;
    h_leg(p) = plot(xx, result_fwd.rho_stag(k,:), '-', 'Color', colors(p,:), ...
                    'LineWidth', 1.5, 'DisplayName', sprintf('$t=%.2f$', t_fracs(p)));
    plot(xx(idx), rho_bwd_flip(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:), 'HandleVisibility', 'off');
end
legend(h_leg, 'Interpreter','latex','Location','best','FontSize',8);
xlabel('$x$','Interpreter','latex'); ylabel('$\rho$','Interpreter','latex');
title(sprintf('Time-reversal: $\\rho_\\mathrm{fwd}(t)$ vs $\\rho_\\mathrm{bwd}(1{-}t)$  ($\\varepsilon=%.2g$)\nsolid = forward,  dots = time-reversed backward  (should overlap)', ...
              cfg.vareps),'Interpreter','latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_time_reversal__rho_fwd_vs_flip_bwd__%s.png', ftag_main)));

%% --- Figure 2: per-time L2 discrepancy (A) ---
t_rho     = (1:ntm)'*dt;
err_per_t = sqrt(dx * sum((result_fwd.rho_stag - rho_bwd_flip).^2, 2));

figure('Name', 'ladmm_time_reversal__per_time_error', 'Position', [100,540,600,300]);
plot(t_rho, err_per_t, 'b-', 'LineWidth', 1.5);
xlabel('$t$','Interpreter','latex');
ylabel('$\|\rho_\mathrm{fwd}(t)-\rho_\mathrm{bwd}(1{-}t)\|_{L^2}$','Interpreter','latex');
title(sprintf('LADMM symmetry error per time slice ($\\varepsilon=%.2g$)', cfg.vareps),'Interpreter','latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_time_reversal__per_time_error__%s.png', ftag_main)));

%% --- Figure 3: grid refinement of symmetry error (A) ---
nt_vals = [16, 32, 64, 128];   nx_vals = [32, 64, 128, 256];
ke_sym  = zeros(numel(nx_vals),2);   err_sym = zeros(numel(nx_vals),1);

fprintf('\nGrid refinement for symmetry error:\n');
for k = 1:numel(nt_vals)
    cfg_k = cfg;   cfg_k.nt = nt_vals(k);   cfg_k.nx = nx_vals(k);
    prob_f = setup_problem(cfg_k, prob_fwd);
    prob_b = setup_problem(cfg_k, prob_bwd);
    r_f = cfg_k.pipeline(cfg_k, prob_f);
    r_b = cfg_k.pipeline(cfg_k, prob_b);
    ke_sym(k,1) = compute_objective(r_f.rho_stag, r_f.mx_stag, prob_f) / prob_f.dx;
    ke_sym(k,2) = compute_objective(r_b.rho_stag, r_b.mx_stag, prob_b) / prob_b.dx;
    rho_flip = flipud(r_b.rho_stag);
    err_sym(k) = sqrt(prob_f.dx*prob_f.dt * sum(sum((r_f.rho_stag - rho_flip).^2)));
    fprintf('  nt=%3d nx=%3d  KE_fwd=%.6f  KE_bwd=%.6f  |diff|=%.2e  sym_err=%.2e\n', ...
            nt_vals(k), nx_vals(k), ke_sym(k,1), ke_sym(k,2), ...
            abs(ke_sym(k,1)-ke_sym(k,2)), err_sym(k));
end

figure('Name', 'ladmm_time_reversal__symmetry_error_vs_grid', 'Position', [730,540,560,320]);
loglog(nx_vals, err_sym, 'ks-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
loglog(nx_vals, err_sym(1)*(nx_vals(1)./nx_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_x$','Interpreter','latex','FontSize',13);
ylabel('Symmetry error','Interpreter','latex');
title('LADMM symmetry error should decay $\mathcal{O}(h^2)$','Interpreter','latex');
legend('Symmetry error','$\mathcal{O}(h^2)$','Interpreter','latex','Location','southwest');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ladmm_time_reversal__symmetry_error_vs_grid__%s.png', ftag_main)));

fprintf('\nFigures saved to: %s\n', fig_dir);
