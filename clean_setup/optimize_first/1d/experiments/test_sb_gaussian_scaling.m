%% test_sb_gaussian_scaling.m
%
% Schrödinger bridge: narrow Gaussian -> wide Gaussian.
%
%   rho0 = N(0.3, 0.04^2),  rho1 = N(0.7, 0.08^2)
%
% The exact analytical solution is known for any two Gaussians (see
% analytical_sb_gaussian_general.m).  This script:
%
%   1. Runs the optimize_then_discretize ADMM pipeline.
%   2. Compares rho and m against the analytical SB solution at multiple
%      time slices (visual) and as an L2 error vs time.
%   3. Runs a small grid-refinement study and checks KE convergence.
%   4. Verifies the eps->0 limit: at eps=1e-6 the solver should agree
%      with the McCann displacement interpolation.

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

%% --- Parameters ---
vareps   = 1e-1;
nt_main  = 64;
nx_main  = 128;

%% --- Config + problem ---
cfg          = cfg_staggered_gaussian();
cfg.vareps   = vareps;
cfg.nt       = nt_main;
cfg.nx       = nx_main;
cfg.projection = @proj_fokker_planck_banded;

prob_def = prob_gaussian_scaling();
problem  = setup_problem(cfg, prob_def);

%% --- Run ADMM ---
fprintf('Running SB: N(%.2f,%.4g) -> N(%.2f,%.4g),  eps=%.2g,  nt=%d, nx=%d...\n', ...
    prob_def.mu0, prob_def.sigma0^2, prob_def.mu1, prob_def.sigma1^2, ...
    vareps, nt_main, nx_main);

result = cfg.pipeline(cfg, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

%% --- Analytical solution ---
[rho_ana, mx_ana] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, vareps);

%% -----------------------------------------------------------------------
%% Figure 1: Density evolution at selected times
%% -----------------------------------------------------------------------
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
nt      = problem.nt;
ntm     = nt - 1;
nx      = problem.nx;
dt      = problem.dt;
xx      = problem.xx;
colors  = parula(numel(t_fracs));

figure('Name', sprintf('SB gaussian_scaling  eps=%.2g', vareps), ...
       'Position', [100, 100, 720, 400]);
hold on;

for p = 1:numel(t_fracs)
    k    = max(1, min(ntm, round(t_fracs(p) * nt)));
    stride = max(1, floor(nx/60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana(k,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result.rho(k,idx), 'o', 'Color', colors(p,:), ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end

leg = cell(2*numel(t_fracs), 1);
for p = 1:numel(t_fracs)
    k = max(1, min(ntm, round(t_fracs(p)*nt)));
    leg{2*p-1} = sprintf('Exact  t=%.2f', k*dt);
    leg{2*p}   = sprintf('ADMM   t=%.2f', k*dt);
end
legend(leg, 'Location', 'best', 'FontSize', 7);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('SB: $N(0.3,0.04^2)\\to N(0.7,0.08^2)$,  $\\varepsilon=%.2g$', vareps), ...
      'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Figure 2: L2 error in rho vs time
%% -----------------------------------------------------------------------
dx      = problem.dx;
t_rho   = (1:ntm)' * dt;
err_rho = sqrt(dx * sum((result.rho - rho_ana).^2, 2));

figure('Name', 'L2 error in rho vs time', 'Position', [100, 520, 600, 300]);
plot(t_rho, err_rho, 'b-', 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\|\rho_\mathrm{ADMM} - \rho_\mathrm{exact}\|_{L^2(x)}$', 'Interpreter', 'latex');
title('$L^2$ error: SB gaussian scaling', 'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Figure 3: Momentum (m) at t=0.5 -- ADMM vs analytical
%% -----------------------------------------------------------------------
nxm     = nx - 1;
x_mx    = (1:nxm) * dx;
k_mid   = round(0.5 * nt);
t_val   = (k_mid - 0.5) * dt;

figure('Name', 'Momentum at t=0.5', 'Position', [730, 100, 600, 350]);
plot(x_mx, mx_ana(k_mid,:), 'r-',  'LineWidth', 1.5); hold on;
plot(x_mx, result.mx(k_mid,:), 'b--', 'LineWidth', 1.5);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$m(t,x)$', 'Interpreter', 'latex');
title(sprintf('Momentum $m$ at $t=%.2f$', t_val), 'Interpreter', 'latex');
legend('Exact', 'ADMM', 'Location', 'best');
grid on;

%% -----------------------------------------------------------------------
%% Figure 4: eps -> 0 limit (pure OT displacement interpolation)
%% -----------------------------------------------------------------------
% For very small eps the SB velocity approaches the McCann interpolation.
% Analytical OT: sigma_t = (1-t)*sigma0 + t*sigma1 (pure interpolation),
%                velocity v(t,x) = (mu1-mu0) + (sigma1-sigma0)/sigma_t*(x-mu_t).
cfg_ot          = cfg;
cfg_ot.vareps   = 1e-6;

fprintf('Running eps=1e-6 (near-OT) ...\n');
result_ot = cfg_ot.pipeline(cfg_ot, problem);
fprintf('  iters=%d,  error=%.2e\n', result_ot.iters, result_ot.error);

[rho_ot, ~] = analytical_sb_gaussian_general(problem, ...
    prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, 1e-6);

err_ot = sqrt(dx * sum((result_ot.rho - rho_ot).^2, 2));

figure('Name', 'OT limit (eps=1e-6)', 'Position', [730, 520, 600, 300]);
plot(t_rho, err_ot, 'r-', 'LineWidth', 1.5); hold on;
plot(t_rho, err_rho, 'b-', 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$L^2$ error', 'Interpreter', 'latex');
legend('$\varepsilon=10^{-6}$ (near-OT)', sprintf('$\\varepsilon=%.2g$ (SB)', vareps), ...
       'Interpreter', 'latex', 'Location', 'best');
title('$L^2$ error vs analytical for both $\varepsilon$ values', 'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Grid refinement: KE and L2 error
%% -----------------------------------------------------------------------
nt_vals = [16, 32, 64];
nx_vals = [32, 64, 128];
n_grids = numel(nt_vals);

ke_admm   = zeros(n_grids,1);
ke_ana    = zeros(n_grids,1);
err_total = zeros(n_grids,1);

fprintf('\nGrid refinement study (eps=%.2g):\n', vareps);
for k = 1:n_grids
    cfg_k    = cfg;
    cfg_k.nt = nt_vals(k);
    cfg_k.nx = nx_vals(k);
    prob_k   = setup_problem(cfg_k, prob_def);

    fprintf('  [%d/%d] nt=%d, nx=%d ...', k, n_grids, nt_vals(k), nx_vals(k));
    r = cfg_k.pipeline(cfg_k, prob_k);
    fprintf(' iters=%d, err=%.2e\n', r.iters, r.error);

    [rho_a, mx_a] = analytical_sb_gaussian_general(prob_k, ...
        prob_def.mu0, prob_def.sigma0, prob_def.mu1, prob_def.sigma1, vareps);

    ke_admm(k) = compute_objective(r.rho, r.mx, prob_k) / prob_k.dx;
    ke_ana(k)  = compute_objective(rho_a, mx_a, prob_k) / prob_k.dx;

    err_rho_k = norm(r.rho(:) - rho_a(:)) * sqrt(prob_k.dt * prob_k.dx);
    err_mx_k  = norm(r.mx(:)  - mx_a(:))  * sqrt(prob_k.dt * prob_k.dx);
    err_total(k) = sqrt(err_rho_k^2 + err_mx_k^2);
end

figure('Name', 'Grid refinement: L2 error', 'Position', [100, 870, 560, 320]);
loglog(nx_vals, err_total, 'ks-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
loglog(nx_vals, err_total(1)*(nx_vals(1)./nx_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$L^2$ error $\|(\rho,m)-(\rho,m)_\mathrm{exact}\|$', ...
       'Interpreter', 'latex', 'FontSize', 11);
title('Grid refinement: gaussian scaling SB', 'Interpreter', 'latex');
legend('ADMM error', '$\mathcal{O}(h^2)$', 'Interpreter', 'latex', 'Location', 'southwest');
grid on;

figure('Name', 'Grid refinement: KE', 'Position', [680, 870, 560, 320]);
semilogx(nx_vals, ke_admm, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
semilogx(nx_vals, ke_ana,  'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Kinetic energy', 'Interpreter', 'latex');
title(sprintf('KE convergence  ($\\varepsilon=%.2g$)', vareps), 'Interpreter', 'latex');
legend('ADMM', 'Analytical', 'Interpreter', 'latex', 'Location', 'best');
grid on;

%% --- Print summary ---
fprintf('\n%-12s  %-10s  %-12s  %-12s  %-12s\n', ...
        'nt x nx', 'iters', 'KE (ADMM)', 'KE (exact)', 'L2 error');
fprintf('%s\n', repmat('-', 1, 62));
for k = 1:n_grids
    fprintf('%3d x %-6d  %-10s  %-12.6f  %-12.6f  %-12.2e\n', ...
            nt_vals(k), nx_vals(k), '---', ke_admm(k), ke_ana(k), err_total(k));
end

%% --- Save all figures ---
fig_dir = fullfile(base, '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

test_tag = 'sb_gaussian_scaling';
cfg_tag  = sprintf('nt%d_nx%d_eps%g', nt_main, nx_main, vareps);
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
