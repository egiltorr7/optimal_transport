%% test_stationary.m
%
% Trivial transport: rho0 = rho1 = N(0.5, 0.05^2).
%
% For eps=0 (OT):
%   Exact solution is rho(t,x) = rho0(x), m = 0, KE = 0.
%   Any nonzero KE or momentum is a solver artefact.
%
% For eps>0 (SB):
%   The optimal process is the "looping" SB between identical Gaussians.
%   Diffusion causes the distribution to spread at mid-time and then
%   contract back.  The analytical solution is available via
%   analytical_sb_gaussian_general (mu0=mu1, sigma0=sigma1).
%   The KE is nonzero for eps>0 (cost of the looping diffusion path).
%
% Checks:
%   1. KE(eps=0) < solver tolerance (catches spurious transport).
%   2. max|m(t,x)| ~ 0 for eps=0.
%   3. ADMM rho(t,x) ≈ rho0(x) for all t (eps=0).
%   4. ADMM agrees with analytical looping SB for eps>0.
%   5. KE(eps) is monotone increasing in eps from 0.

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

prob_def = prob_stationary();
problem  = setup_problem(cfg, prob_def);

nt  = problem.nt;   ntm = nt - 1;   dt  = problem.dt;
nx  = problem.nx;   dx  = problem.dx;   nxm = nx - 1;
xx  = problem.xx;
t_rho = (1:ntm)' * dt;

%% -----------------------------------------------------------------------
%% Test 1: OT (eps=0) — KE and momentum should be (near) zero
%% -----------------------------------------------------------------------
cfg_ot        = cfg;
cfg_ot.vareps = 0;

fprintf('Running OT (eps=0): stationary ...\n');
result_ot = cfg_ot.pipeline(cfg_ot, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result_ot.iters, result_ot.converged, result_ot.error, result_ot.walltime);

ke_ot    = compute_objective(result_ot.rho, result_ot.mx, problem) / dx;
mx_max   = max(abs(result_ot.mx(:)));
rho_drift = sqrt(dx * dt * sum(sum((result_ot.rho - repmat(problem.rho0, ntm, 1)).^2)));

fprintf('\n--- OT (eps=0) checks ---\n');
fprintf('  KE                     = %.2e   (should be ~0)\n', ke_ot);
fprintf('  max|m(t,x)|            = %.2e   (should be ~0)\n', mx_max);
fprintf('  ||rho(t) - rho0||_L2   = %.2e   (should be ~0)\n', rho_drift);

%% -----------------------------------------------------------------------
%% Test 2: SB (eps>0) — compare against analytical looping Gaussian SB
%% -----------------------------------------------------------------------
eps_sb        = 1e-1;
cfg_sb        = cfg;
cfg_sb.vareps = eps_sb;

fprintf('\nRunning SB (eps=%.2g): looping Gaussian ...\n', eps_sb);
result_sb = cfg_sb.pipeline(cfg_sb, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e\n', ...
    result_sb.iters, result_sb.converged, result_sb.error);

[rho_ana, mx_ana] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, eps_sb);

ke_sb     = compute_objective(result_sb.rho, result_sb.mx, problem) / dx;
ke_ana_sb = compute_objective(rho_ana, mx_ana, problem) / dx;
err_rho   = sqrt(dx * sum((result_sb.rho - rho_ana).^2, 2));   % per-time L2 error

fprintf('\n--- SB (eps=%.2g) checks ---\n', eps_sb);
fprintf('  KE (ADMM)       = %.6f\n', ke_sb);
fprintf('  KE (analytical) = %.6f\n', ke_ana_sb);
fprintf('  max L2 error in rho vs time = %.2e\n', max(err_rho));

%% -----------------------------------------------------------------------
%% Figure 1: Density evolution for eps=0 and eps=eps_sb
%% -----------------------------------------------------------------------
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
colors  = parula(numel(t_fracs));

figure('Name', 'Stationary: density (eps=0 and eps>0)', 'Position', [100, 100, 1100, 420]);

% OT panel
subplot(1,2,1);
hold on;
for p = 1:numel(t_fracs)
    k      = max(1, min(ntm, round(t_fracs(p) * nt)));
    stride = max(1, floor(nx/60));
    idx    = 1:stride:nx;
    plot(xx, repmat(problem.rho0, 1, 1), '-', 'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result_ot.rho(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('OT ($\\varepsilon=0$),  KE=%.2e', ke_ot), 'Interpreter', 'latex');
grid on;

% SB panel
subplot(1,2,2);
hold on;
for p = 1:numel(t_fracs)
    k      = max(1, min(ntm, round(t_fracs(p) * nt)));
    stride = max(1, floor(nx/60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana(k,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result_sb.rho(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('Looping SB ($\\varepsilon=%.2g$): spread then contract', eps_sb), ...
      'Interpreter', 'latex');
grid on;
sgtitle('Stationary transport: $\rho_0 = \rho_1 = \mathcal{N}(0.5,\,0.05^2)$', ...
        'Interpreter', 'latex', 'FontSize', 13);

%% -----------------------------------------------------------------------
%% Figure 2: Width of rho vs time — should be symmetric about t=0.5 for SB
%% -----------------------------------------------------------------------
% Estimate standard deviation of rho at each time (measures spreading/contracting)
sigma_ot = zeros(ntm,1);
sigma_sb = zeros(ntm,1);
sigma_an = zeros(ntm,1);

for k = 1:ntm
    mu_ot = sum(xx .* result_ot.rho(k,:)) * dx;
    mu_sb = sum(xx .* result_sb.rho(k,:)) * dx;
    mu_an = sum(xx .* rho_ana(k,:)) * dx;
    sigma_ot(k) = sqrt(sum((xx - mu_ot).^2 .* result_ot.rho(k,:)) * dx);
    sigma_sb(k) = sqrt(sum((xx - mu_sb).^2 .* result_sb.rho(k,:)) * dx);
    sigma_an(k) = sqrt(sum((xx - mu_an).^2 .* rho_ana(k,:)) * dx);
end

figure('Name', 'Stationary: width vs time', 'Position', [100, 540, 700, 320]);
plot(t_rho, sigma_ot * sqrt(nx), 'b-',  'LineWidth', 1.5); hold on;
plot(t_rho, sigma_sb * sqrt(nx), 'r-',  'LineWidth', 1.5);
plot(t_rho, sigma_an * sqrt(nx), 'r--', 'LineWidth', 1.5);
yline(prob_def.sigma0 * sqrt(nx), 'k:', 'LineWidth', 1.2);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('Width (normalised)', 'Interpreter', 'latex');
legend('OT ($\varepsilon=0$)', sprintf('SB ($\\varepsilon=%.2g$)', eps_sb), ...
       'Analytical SB', '$\sigma_0$', ...
       'Interpreter', 'latex', 'Location', 'best');
title('Distribution width vs time (should be symmetric about $t=0.5$ for SB)', ...
      'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Figure 3: L2 error in rho vs time for SB
%% -----------------------------------------------------------------------
figure('Name', 'Stationary SB: L2 error vs time', 'Position', [820, 540, 560, 320]);
plot(t_rho, err_rho, 'b-', 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\|\rho_\mathrm{ADMM} - \rho_\mathrm{exact}\|_{L^2(x)}$', 'Interpreter', 'latex');
title(sprintf('$L^2$ error: looping SB  ($\\varepsilon=%.2g$)', eps_sb), 'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Figure 4: KE vs eps — should be monotone increasing from 0
%% -----------------------------------------------------------------------
eps_vals = [0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1];
ke_eps   = zeros(numel(eps_vals), 1);
ke_ana_eps = zeros(numel(eps_vals), 1);

fprintf('\neps sweep:\n');
for j = 1:numel(eps_vals)
    cfg_j        = cfg;
    cfg_j.vareps = eps_vals(j);
    r_j          = cfg_j.pipeline(cfg_j, problem);
    ke_eps(j)    = compute_objective(r_j.rho, r_j.mx, problem) / dx;

    if eps_vals(j) == 0
        ke_ana_eps(j) = 0;
    else
        [ra, ma] = analytical_sb_gaussian_general(problem, ...
            prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, eps_vals(j));
        ke_ana_eps(j) = compute_objective(ra, ma, problem) / dx;
    end
    fprintf('  eps=%.0e  KE=%.6f  KE_ana=%.6f\n', eps_vals(j), ke_eps(j), ke_ana_eps(j));
end

figure('Name', 'Stationary: KE vs eps', 'Position', [100, 870, 600, 340]);
semilogx(eps_vals(2:end), ke_eps(2:end), 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
semilogx(eps_vals(2:end), ke_ana_eps(2:end), 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
% Show OT value at smallest eps on left margin
yline(ke_eps(1), 'k--', sprintf('OT KE=%.2e', ke_eps(1)), 'LineWidth', 1.2);
xlabel('$\varepsilon$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Kinetic energy', 'Interpreter', 'latex');
title('Stationary: KE increases with $\varepsilon$ (looping cost)', 'Interpreter', 'latex');
legend('ADMM', 'Analytical', 'Interpreter', 'latex', 'Location', 'best');
grid on;

%% --- Save all figures ---
fig_dir = fullfile(base, '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

test_tag = 'stationary';
cfg_tag  = sprintf('nt%d_nx%d_eps_sweep', cfg.nt, cfg.nx);
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
