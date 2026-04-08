%% test_sb_pure_scaling.m
%
% Pure spreading: N(0.5, 0.03^2) -> N(0.5, 0.10^2).
%
% Same centre (mu=0.5), different width.  No translation component.
% The velocity field is purely radial (antisymmetric about x=0.5):
%   v(t,x) = b_t * (x - 0.5)
%
% This isolates the sigma-evolution part of the SB formula from translation.
%
% Checks:
%   1. Centre of mass stays at 0.5 for all t (symmetry must be preserved).
%   2. ADMM rho and m agree with analytical_sb_gaussian_general.
%   3. m(t,x) is antisymmetric about x=0.5 (consequence of radial velocity).
%   4. For eps=0: KE = W_2^2 = (sigma1-sigma0)^2 = 0.07^2 = 0.0049.
%   5. Grid-refinement O(h^2) convergence.

clear; clc; close all;

base = fileparts(mfilename('fullpath'));
addpath(fullfile(base, '..'));
addpath(fullfile(base, '..', 'config'));
addpath(fullfile(base, '..', 'problems'));
addpath(fullfile(base, '..', 'discretization'));
addpath(fullfile(base, '..', 'projection'));
addpath(fullfile(base, '..', 'prox'));
addpath(fullfile(base, '..', 'pipelines'));
addpath(fullfile(base, '..', 'utils'));

%% --- Config ---
cfg            = cfg_staggered_gaussian();
cfg.nt         = 64;
cfg.nx         = 128;
cfg.projection = @proj_fokker_planck_banded;

prob_def = prob_gaussian_pure_scaling();
problem  = setup_problem(cfg, prob_def);

nt  = problem.nt;   ntm = nt - 1;   dt  = problem.dt;
nx  = problem.nx;   dx  = problem.dx;   nxm = nx - 1;
xx  = problem.xx;
x_mx  = (1:nxm) * dx;
t_rho = (1:ntm)' * dt;

%% -----------------------------------------------------------------------
%% Run ADMM for eps=0.1 and eps=0
%% -----------------------------------------------------------------------
vareps = 1e-1;

cfg_sb        = cfg;
cfg_sb.vareps = vareps;
fprintf('Running SB (eps=%.2g): pure scaling ...\n', vareps);
result_sb = cfg_sb.pipeline(cfg_sb, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e\n', ...
    result_sb.iters, result_sb.converged, result_sb.error);

cfg_ot        = cfg;
cfg_ot.vareps = 0;
fprintf('Running OT (eps=0): pure scaling ...\n');
result_ot = cfg_ot.pipeline(cfg_ot, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e\n', ...
    result_ot.iters, result_ot.converged, result_ot.error);

%% --- Analytical solutions ---
[rho_ana_sb, mx_ana_sb] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, vareps);

[rho_ana_ot, mx_ana_ot] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, 1e-8);

%% -----------------------------------------------------------------------
%% Check 1: Centre of mass at x=0.5 for all t
%% -----------------------------------------------------------------------
com_sb = sum(xx .* result_sb.rho, 2) / sum(result_sb.rho(1,:));   % (ntm x 1)
com_ot = sum(xx .* result_ot.rho, 2) / sum(result_ot.rho(1,:));

fprintf('\n--- Centre-of-mass check ---\n');
fprintf('  SB: max|CoM - 0.5| = %.2e   (should be ~0)\n', max(abs(com_sb - 0.5)));
fprintf('  OT: max|CoM - 0.5| = %.2e   (should be ~0)\n', max(abs(com_ot - 0.5)));

%% -----------------------------------------------------------------------
%% Check 2: KE(eps=0) = W_2^2 = (sigma1-sigma0)^2
%% -----------------------------------------------------------------------
ke_ot     = compute_objective(result_ot.rho, result_ot.mx, problem) / dx;
ke_sb     = compute_objective(result_sb.rho, result_sb.mx, problem) / dx;
ke_exact  = (prob_def.sigma1 - prob_def.sigma0)^2;

[~, ~, ke_ana_sb_val] = deal(0);
rho_a = rho_ana_sb;   mx_a = mx_ana_sb;
ke_ana_sb_val = compute_objective(rho_a, mx_a, problem) / dx;

fprintf('\n--- KE checks ---\n');
fprintf('  OT KE (ADMM)  = %.6f\n',    ke_ot);
fprintf('  OT KE (W_2^2) = %.6f   (exact)\n', ke_exact);
fprintf('  SB KE (ADMM)  = %.6f\n',    ke_sb);
fprintf('  SB KE (ana)   = %.6f\n', ke_ana_sb_val);

%% -----------------------------------------------------------------------
%% Check 3: Antisymmetry of m about x=0.5
%% -----------------------------------------------------------------------
% m(t,x) = -m(t, 1-x) for a purely radial velocity
nxm_half = floor(nxm/2);
m_mid     = result_sb.mx;
m_flip    = fliplr(m_mid(:, 1:nxm));   % flip x axis

antisym_err = max(abs(m_mid(:,1:nxm_half) + m_flip(:,nxm+1-nxm_half:nxm)));
fprintf('\n--- Momentum antisymmetry check ---\n');
fprintf('  max|m(t,x) + m(t,1-x)| = %.2e   (should be ~0)\n', antisym_err);

%% -----------------------------------------------------------------------
%% Figure 1: Density evolution — SB and OT
%% -----------------------------------------------------------------------
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
colors  = parula(numel(t_fracs));

figure('Name', 'Pure scaling: density', 'Position', [100, 100, 1100, 400]);

subplot(1,2,1);
hold on;
for p = 1:numel(t_fracs)
    k      = max(1, min(ntm, round(t_fracs(p) * nt)));
    stride = max(1, floor(nx/60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana_ot(k,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result_ot.rho(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('OT ($\\varepsilon\\approx0$),  KE=%.4f vs %.4f', ke_ot, ke_exact), ...
      'Interpreter', 'latex');
grid on;

subplot(1,2,2);
hold on;
for p = 1:numel(t_fracs)
    k      = max(1, min(ntm, round(t_fracs(p) * nt)));
    stride = max(1, floor(nx/60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana_sb(k,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result_sb.rho(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('SB ($\\varepsilon=%.2g$),  KE=%.4f vs %.4f', vareps, ke_sb, ke_ana_sb_val), ...
      'Interpreter', 'latex');
grid on;
sgtitle('Pure scaling: $\mathcal{N}(0.5,0.03^2)\to\mathcal{N}(0.5,0.10^2)$  (solid=exact, dots=ADMM)', ...
        'Interpreter', 'latex', 'FontSize', 12);

%% -----------------------------------------------------------------------
%% Figure 2: Momentum at mid-time — should be antisymmetric
%% -----------------------------------------------------------------------
k_mid = round(0.5 * nt);
figure('Name', 'Pure scaling: momentum at t=0.5', 'Position', [100, 530, 700, 320]);
plot(x_mx, mx_ana_sb(k_mid,:), 'r-',  'LineWidth', 1.5); hold on;
plot(x_mx, result_sb.mx(k_mid,:), 'b--', 'LineWidth', 1.5);
xline(0.5, 'k:', 'LineWidth', 1.0);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$m(t=0.5,x)$', 'Interpreter', 'latex');
title('Momentum at $t=0.5$ — must be antisymmetric about $x=0.5$', ...
      'Interpreter', 'latex');
legend('Analytical', 'ADMM', 'Interpreter', 'latex', 'Location', 'best');
grid on;

%% -----------------------------------------------------------------------
%% Figure 3: Centre of mass vs time
%% -----------------------------------------------------------------------
figure('Name', 'Pure scaling: centre of mass', 'Position', [820, 530, 500, 300]);
plot(t_rho, com_sb, 'b-', 'LineWidth', 1.5); hold on;
plot(t_rho, com_ot, 'r-', 'LineWidth', 1.5);
yline(0.5, 'k--', 'LineWidth', 1.2);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\bar{x}(t) = \int x\rho\,dx$', 'Interpreter', 'latex');
title('Centre of mass (must stay at 0.5)', 'Interpreter', 'latex');
legend(sprintf('SB $\\varepsilon=%.2g$', vareps), 'OT', ...
       'Interpreter', 'latex', 'Location', 'best');
ylim([0.45, 0.55]);
grid on;

%% -----------------------------------------------------------------------
%% Figure 4: Grid refinement — L2 error and KE
%% -----------------------------------------------------------------------
nt_vals   = [16, 32, 64, 128];
nx_vals   = [32, 64, 128, 256];
ke_grid   = zeros(numel(nx_vals), 1);
err_grid  = zeros(numel(nx_vals), 1);

fprintf('\nGrid refinement (eps=%.2g):\n', vareps);
for k = 1:numel(nt_vals)
    cfg_k    = cfg_sb;
    cfg_k.nt = nt_vals(k);
    cfg_k.nx = nx_vals(k);
    prob_k   = setup_problem(cfg_k, prob_def);

    r = cfg_k.pipeline(cfg_k, prob_k);
    [rho_a, mx_a] = analytical_sb_gaussian_general(prob_k, ...
        prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, vareps);

    ke_grid(k)  = compute_objective(r.rho, r.mx, prob_k) / prob_k.dx;
    err_rho_k   = norm(r.rho(:) - rho_a(:)) * sqrt(prob_k.dt * prob_k.dx);
    err_mx_k    = norm(r.mx(:)  - mx_a(:))  * sqrt(prob_k.dt * prob_k.dx);
    err_grid(k) = sqrt(err_rho_k^2 + err_mx_k^2);

    fprintf('  nt=%3d, nx=%3d  KE=%.6f  L2=%.2e\n', ...
            nt_vals(k), nx_vals(k), ke_grid(k), err_grid(k));
end

figure('Name', 'Pure scaling: grid refinement', 'Position', [100, 870, 1100, 350]);
subplot(1,2,1);
loglog(nx_vals, err_grid, 'ks-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
loglog(nx_vals, err_grid(1)*(nx_vals(1)./nx_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$L^2$ error', 'Interpreter', 'latex');
title('Grid refinement: $L^2$ error', 'Interpreter', 'latex');
legend('ADMM', '$\mathcal{O}(h^2)$', 'Interpreter', 'latex', 'Location', 'southwest');
grid on;

subplot(1,2,2);
semilogx(nx_vals, ke_grid, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
yline(ke_ana_sb_val, 'r--', 'LineWidth', 1.5);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Kinetic energy', 'Interpreter', 'latex');
title(sprintf('KE convergence  ($\\varepsilon=%.2g$)', vareps), 'Interpreter', 'latex');
legend('ADMM', 'Analytical', 'Interpreter', 'latex', 'Location', 'best');
grid on;

%% --- Save all figures ---
fig_dir = fullfile(base, '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

test_tag = 'sb_pure_scaling';
cfg_tag  = sprintf('nt%d_nx%d_eps%g', cfg.nt, cfg.nx, vareps);
figs = findall(0, 'Type', 'figure');
for fi = 1:numel(figs)
    raw   = figs(fi).Name;
    clean = regexprep(raw, '[^\w]', '_');
    clean = regexprep(clean, '_+', '_');
    clean = strtrim(clean);
    fname = sprintf('%s__%s__%s', test_tag, clean, cfg_tag);
    saveas(figs(fi), fullfile(fig_dir, [fname '.png']));
end
fprintf('\nFigures saved to: %s\n', fig_dir);
