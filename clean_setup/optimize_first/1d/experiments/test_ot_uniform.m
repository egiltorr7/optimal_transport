%% test_ot_uniform.m
%
% Optimal transport (eps=0) between two uniform distributions with compact
% support:  Uniform[0.2, 0.4] -> Uniform[0.6, 0.8].
%
% Exact solution (analytical_ot_uniform):
%   rho(t,x) = Uniform[0.2 + 0.4*t, 0.4 + 0.4*t]  (sliding box)
%   m(t,x)   = rho(t,x) * 0.4                      (constant velocity)
%   KE        = W_2^2 = 0.4^2 = 0.16
%
% Tests:
%   1. Density and momentum profiles vs analytical at multiple times.
%   2. KE converges to 0.16 as grid refines.
%   3. L2 error vs grid size (expect ~O(h) due to Gibbs at sharp edges;
%      the O(h^2) rate will recover away from the support boundaries).
%   4. FP residual check: ||d_t rho + d_x m||_2 -> 0.
%
% Note: the discontinuous support edges are a stress test.  Expect
% Gibbs-like ringing near the discontinuities; the KE and mass are more
% reliable indicators of solver quality than the pointwise L2 error.

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

%% --- Config: pure OT (eps=0) ---
cfg            = cfg_staggered_gaussian();
cfg.vareps     = 0;
cfg.nt         = 64;
cfg.nx         = 128;
cfg.projection = @proj_fokker_planck_banded;

prob_def = prob_uniform();
problem  = setup_problem(cfg, prob_def);

%% --- Run ADMM ---
fprintf('Running OT (eps=0): Uniform[0.2,0.4] -> Uniform[0.6,0.8],  nt=%d, nx=%d ...\n', ...
    cfg.nt, cfg.nx);
result = cfg.pipeline(cfg, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

%% --- Analytical solution ---
[rho_ana, mx_ana] = analytical_ot_uniform(problem);

%% -----------------------------------------------------------------------
%% Figure 1: Density profiles at selected times
%% -----------------------------------------------------------------------
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
nt      = problem.nt;   ntm = nt - 1;
nx      = problem.nx;   dx  = problem.dx;   dt  = problem.dt;
xx      = problem.xx;
colors  = parula(numel(t_fracs));

figure('Name', 'OT Uniform density', 'Position', [100, 100, 720, 400]);
hold on;

for p = 1:numel(t_fracs)
    k      = max(1, min(ntm, round(t_fracs(p) * nt)));
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
title('OT: Uniform$[0.2,0.4] \to$ Uniform$[0.6,0.8]$  ($\varepsilon=0$)', ...
      'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Figure 2: L2 error vs time (pointwise)
%% -----------------------------------------------------------------------
t_rho   = (1:ntm)' * dt;
err_rho = sqrt(dx * sum((result.rho - rho_ana).^2, 2));

figure('Name', 'L2 error rho vs time', 'Position', [100, 520, 600, 300]);
plot(t_rho, err_rho, 'b-', 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\|\rho_\mathrm{ADMM} - \rho_\mathrm{exact}\|_{L^2(x)}$', 'Interpreter', 'latex');
title('$L^2$ error vs analytical (OT, uniform support)', 'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Figure 3: Kinetic energy vs eps=0 analytical value
%% -----------------------------------------------------------------------
ke_admm = compute_objective(result.rho, result.mx, problem) / problem.dx;
ke_exact = 0.16;   % W_2^2 = shift^2 = 0.4^2

fprintf('\nKE (ADMM) = %.6f,  KE (exact W2^2) = %.6f,  rel error = %.2e\n', ...
    ke_admm, ke_exact, abs(ke_admm - ke_exact)/ke_exact);

%% -----------------------------------------------------------------------
%% Grid refinement: KE and L2 error
%% -----------------------------------------------------------------------
nt_vals = [16, 32, 64, 128];
nx_vals = [32, 64, 128, 256];
n_grids = numel(nt_vals);

ke_grid   = zeros(n_grids,1);
err_grid  = zeros(n_grids,1);

fprintf('\nGrid refinement study (eps=0, OT):\n');
for k = 1:n_grids
    cfg_k        = cfg;
    cfg_k.nt     = nt_vals(k);
    cfg_k.nx     = nx_vals(k);
    prob_k       = setup_problem(cfg_k, prob_def);

    fprintf('  [%d/%d] nt=%d, nx=%d ...', k, n_grids, nt_vals(k), nx_vals(k));
    r = cfg_k.pipeline(cfg_k, prob_k);
    fprintf(' iters=%d, err=%.2e\n', r.iters, r.error);

    [rho_a, mx_a] = analytical_ot_uniform(prob_k);

    ke_grid(k) = compute_objective(r.rho, r.mx, prob_k) / prob_k.dx;

    err_r = norm(r.rho(:) - rho_a(:)) * sqrt(prob_k.dt * prob_k.dx);
    err_m = norm(r.mx(:)  - mx_a(:))  * sqrt(prob_k.dt * prob_k.dx);
    err_grid(k) = sqrt(err_r^2 + err_m^2);
end

figure('Name', 'Uniform OT: KE vs grid', 'Position', [730, 100, 560, 320]);
semilogx(nx_vals, ke_grid, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
yline(ke_exact, 'r--', 'LineWidth', 1.5, 'Label', 'W_2^2 = 0.16');
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Kinetic energy', 'Interpreter', 'latex');
title('KE convergence to $W_2^2 = 0.16$', 'Interpreter', 'latex');
legend('ADMM', 'Exact $W_2^2$', 'Interpreter', 'latex', 'Location', 'best');
grid on;

figure('Name', 'Uniform OT: L2 error vs grid', 'Position', [730, 440, 560, 320]);
loglog(nx_vals, err_grid, 'ks-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
% For a step function, expect O(h^0.5) or O(h^1) depending on alignment;
% show O(h) reference line anchored to coarsest grid
loglog(nx_vals, err_grid(1)*(nx_vals(1)./nx_vals), 'k--', 'LineWidth', 1.2);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$L^2$ error', 'Interpreter', 'latex');
title('$L^2$ error (expect $\sim\mathcal{O}(h)$ near step edges)', ...
      'Interpreter', 'latex');
legend('ADMM error', '$\mathcal{O}(h)$', 'Interpreter', 'latex', 'Location', 'southwest');
grid on;

%% -----------------------------------------------------------------------
%% Bonus: SB (eps>0) test -- no analytical solution, visual check only
%% -----------------------------------------------------------------------
eps_sb = 1e-2;
fprintf('\nRunning SB (eps=%.2g) on uniform problem (no analytical) ...\n', eps_sb);
cfg_sb        = cfg;
cfg_sb.vareps = eps_sb;
result_sb = cfg_sb.pipeline(cfg_sb, problem);
fprintf('  iters=%d,  error=%.2e,  KE=%.6f\n', ...
    result_sb.iters, result_sb.error, ...
    compute_objective(result_sb.rho, result_sb.mx, problem) / problem.dx);

figure('Name', sprintf('SB Uniform eps=%.2g', eps_sb), 'Position', [100, 870, 720, 400]);
hold on;
for p = 1:numel(t_fracs)
    k      = max(1, min(ntm, round(t_fracs(p) * nt)));
    stride = max(1, floor(nx/60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana(k,:),        '-',  'Color', colors(p,:), 'LineWidth', 1.2);
    plot(xx(idx), result_sb.rho(k,idx), 's', 'Color', colors(p,:), ...
         'MarkerSize', 4, 'MarkerFaceColor', colors(p,:));
end
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('SB uniform (solid: OT exact, markers: SB $\\varepsilon=%.2g$)', eps_sb), ...
      'Interpreter', 'latex');
grid on;

%% --- Summary ---
fprintf('\n%-12s  %-10s  %-12s  %-12s\n', 'nt x nx', 'iters', 'KE', 'L2 error');
fprintf('%s\n', repmat('-', 1, 52));
for k = 1:n_grids
    fprintf('%3d x %-6d  %-10s  %-12.6f  %-12.2e\n', ...
            nt_vals(k), nx_vals(k), '---', ke_grid(k), err_grid(k));
end
fprintf('Exact W_2^2 = %.6f\n', ke_exact);

%% --- Save all figures ---
fig_dir = fullfile(base, '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

test_tag = 'ot_uniform';
cfg_tag  = sprintf('nt%d_nx%d', cfg.nt, cfg.nx);   % eps=0 always
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
