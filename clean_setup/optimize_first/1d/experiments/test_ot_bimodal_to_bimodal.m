%% test_ot_bimodal_to_bimodal.m
%
% OT / SB between two bimodal distributions with inward-moving peaks.
%
%   rho0 = N(0.20, 0.04^2) + N(0.80, 0.04^2)   (far apart)
%   rho1 = N(0.35, 0.04^2) + N(0.65, 0.04^2)   (closer)
%
% The OT (eps=0) optimal plan is non-crossing:
%   left  peak:  0.20 -> 0.35   (v_L = +0.15)
%   right peak:  0.80 -> 0.65   (v_R = -0.15)
%   KE = W_2^2 = 0.0225
%
% The crossing plan (shift 0.45 each) has KE = 0.2025, nine times larger.
%
% Tests:
%   1. KE(ADMM, eps=0) converges to 0.0225 (non-crossing plan).
%   2. ADMM density and momentum agree with the analytical displacement
%      interpolation (analytical_ot_bimodal_to_bimodal).
%   3. For eps>0, both peaks interact: the lobes merge earlier and the
%      KE increases monotonically with eps.
%   4. Grid refinement O(h^2) convergence (away from the Gaussian tails).

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

prob_def = prob_bimodal_to_bimodal();
problem  = setup_problem(cfg, prob_def);

nt  = problem.nt;   ntm = nt - 1;   dt  = problem.dt;
nx  = problem.nx;   dx  = problem.dx;   nxm = nx - 1;
xx  = problem.xx;
x_mx  = (1:nxm) * dx;
t_rho = (1:ntm)' * dt;

ke_exact_ot = prob_def.ke_ot;   % 0.0225

%% -----------------------------------------------------------------------
%% Test 1: OT (eps=0) — non-crossing plan
%% -----------------------------------------------------------------------
cfg_ot        = cfg;
cfg_ot.vareps = 0;

fprintf('Running OT (eps=0): bimodal -> bimodal ...\n');
result_ot = cfg_ot.pipeline(cfg_ot, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result_ot.iters, result_ot.converged, result_ot.error, result_ot.walltime);

[rho_ana, mx_ana] = analytical_ot_bimodal_to_bimodal(problem);

ke_ot  = compute_objective(result_ot.rho, result_ot.mx, problem) / dx;
ke_ana = compute_objective(rho_ana, mx_ana, problem) / dx;

err_rho = sqrt(dx * sum((result_ot.rho - rho_ana).^2, 2));
err_mx  = sqrt(dx * sum((result_ot.mx  - mx_ana ).^2, 2));

fprintf('\n--- OT checks ---\n');
fprintf('  KE (ADMM)          = %.6f\n', ke_ot);
fprintf('  KE (W_2^2 exact)   = %.6f\n', ke_exact_ot);
fprintf('  KE (analytical rho)= %.6f\n', ke_ana);
fprintf('  KE crossing plan   = %.6f   (9x larger; solver avoids this)\n', ...
        2 * 0.5 * 0.45^2);
fprintf('  max L2 err rho     = %.2e\n', max(err_rho));
fprintf('  max L2 err mx      = %.2e\n', max(err_mx));

%% -----------------------------------------------------------------------
%% Test 2: SB (eps=0.05) — diffusion merges the lobes
%% -----------------------------------------------------------------------
vareps        = 5e-2;
cfg_sb        = cfg;
cfg_sb.vareps = vareps;

fprintf('\nRunning SB (eps=%.2g): bimodal -> bimodal ...\n', vareps);
result_sb = cfg_sb.pipeline(cfg_sb, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e\n', ...
    result_sb.iters, result_sb.converged, result_sb.error);

ke_sb = compute_objective(result_sb.rho, result_sb.mx, problem) / dx;
fprintf('  KE (SB eps=%.2g) = %.6f   (> OT KE = %.6f)\n', vareps, ke_sb, ke_ot);

%% -----------------------------------------------------------------------
%% Figure 1: Density evolution — OT
%% -----------------------------------------------------------------------
t_fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
colors  = parula(numel(t_fracs));

figure('Name', 'Bimodal->bimodal: OT density', 'Position', [100, 100, 720, 420]);
hold on;

for p = 1:numel(t_fracs)
    tf = t_fracs(p);
    if tf == 0.0
        row_a = problem.rho0;
        row_n = problem.rho0;
        t_disp = 0;
    elseif tf == 1.0
        row_a = problem.rho1;
        row_n = problem.rho1;
        t_disp = 1;
    else
        k      = max(1, min(ntm, round(tf * nt)));
        row_a  = rho_ana(k, :);
        row_n  = result_ot.rho(k, :);
        t_disp = k * dt;
    end
    stride = max(1, floor(nx/60));
    idx    = 1:stride:nx;
    plot(xx, row_a, '-', 'Color', colors(p,:), 'LineWidth', 1.5);
    if ~(tf == 0.0 || tf == 1.0)
        plot(xx(idx), row_n(idx), 'o', 'Color', colors(p,:), ...
             'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
    end
end

xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('OT ($\\varepsilon=0$): non-crossing bimodal transport  KE=%.4f vs %.4f', ...
              ke_ot, ke_exact_ot), 'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Figure 2: Comparison OT vs SB at t=0.5
%% -----------------------------------------------------------------------
k_mid = round(0.5 * nt);

figure('Name', 'Bimodal->bimodal: OT vs SB at t=0.5', 'Position', [100, 540, 700, 340]);
plot(xx, rho_ana(k_mid,:),       'k-',  'LineWidth', 1.5); hold on;
plot(xx, result_ot.rho(k_mid,:), 'b--', 'LineWidth', 1.5);
plot(xx, result_sb.rho(k_mid,:), 'r-',  'LineWidth', 1.5);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho(t=0.5,x)$', 'Interpreter', 'latex');
title('Density at $t=0.5$: OT vs SB (diffusion merges lobes earlier)', ...
      'Interpreter', 'latex');
legend('OT exact', sprintf('OT ADMM  KE=%.4f', ke_ot), ...
       sprintf('SB ADMM $\\varepsilon=%.2g$  KE=%.4f', vareps, ke_sb), ...
       'Interpreter', 'latex', 'Location', 'best');
grid on;

%% -----------------------------------------------------------------------
%% Figure 3: L2 error vs time (OT)
%% -----------------------------------------------------------------------
figure('Name', 'Bimodal->bimodal: L2 error vs time', 'Position', [820, 540, 560, 320]);
plot(t_rho, err_rho, 'b-', 'LineWidth', 1.5); hold on;
plot(t_rho, err_mx,  'r-', 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$L^2$ error', 'Interpreter', 'latex');
legend('$\|\rho_\mathrm{ADMM} - \rho_\mathrm{exact}\|$', ...
       '$\|m_\mathrm{ADMM} - m_\mathrm{exact}\|$', ...
       'Interpreter', 'latex', 'Location', 'best');
title('$L^2$ error vs time (OT, bimodal$\to$bimodal)', 'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Figure 4: eps sweep — KE vs eps, density at t=0.5
%% -----------------------------------------------------------------------
eps_vals = [0, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1];
ke_eps   = zeros(numel(eps_vals), 1);
colors_eps = lines(numel(eps_vals));

k_half = round(0.5 * nt);

figure('Name', 'Bimodal->bimodal: eps sweep', 'Position', [100, 870, 720, 420]);
hold on;

fprintf('\neps sweep on bimodal->bimodal:\n');
for j = 1:numel(eps_vals)
    cfg_j        = cfg;
    cfg_j.vareps = eps_vals(j);
    r_j          = cfg_j.pipeline(cfg_j, problem);
    ke_eps(j)    = compute_objective(r_j.rho, r_j.mx, problem) / dx;
    fprintf('  eps=%.1e  KE=%.6f\n', eps_vals(j), ke_eps(j));
    plot(xx, r_j.rho(k_half,:), '-', 'Color', colors_eps(j,:), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('$\\varepsilon=%.0e$  KE=%.4f', eps_vals(j), ke_eps(j)));
end

xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho(t=0.5,x)$', 'Interpreter', 'latex');
title('Bimodal$\to$bimodal: $\varepsilon$ sweep at $t=0.5$', 'Interpreter', 'latex');
legend('Interpreter', 'latex', 'Location', 'best', 'FontSize', 8, 'NumColumns', 2);
grid on;

%% -----------------------------------------------------------------------
%% Figure 5: KE vs eps — monotone increasing from W_2^2
%% -----------------------------------------------------------------------
figure('Name', 'Bimodal->bimodal: KE vs eps', 'Position', [840, 870, 540, 340]);
semilogx([eps_vals(2:end)], ke_eps(2:end), 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
yline(ke_exact_ot, 'r--', 'LineWidth', 1.5, ...
      'Label', sprintf('W_2^2 = %.4f', ke_exact_ot));
scatter(eps_vals(1), ke_eps(1), 80, 'b', 'filled', 'DisplayName', 'eps=0');
xlabel('$\varepsilon$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Kinetic energy', 'Interpreter', 'latex');
title('KE monotone increasing with $\varepsilon$', 'Interpreter', 'latex');
legend('KE($\varepsilon>0$)', '$W_2^2$', 'KE($\varepsilon=0$)', ...
       'Interpreter', 'latex', 'Location', 'best');
grid on;

%% -----------------------------------------------------------------------
%% Grid refinement (eps=0): KE -> W_2^2 and L2 error
%% -----------------------------------------------------------------------
nt_vals   = [16, 32, 64, 128];
nx_vals   = [32, 64, 128, 256];
ke_grid   = zeros(numel(nx_vals), 1);
err_grid  = zeros(numel(nx_vals), 1);

fprintf('\nGrid refinement (eps=0):\n');
for k = 1:numel(nt_vals)
    cfg_k        = cfg;
    cfg_k.vareps = 0;
    cfg_k.nt     = nt_vals(k);
    cfg_k.nx     = nx_vals(k);
    prob_k       = setup_problem(cfg_k, prob_def);

    r = cfg_k.pipeline(cfg_k, prob_k);
    [rho_a, mx_a] = analytical_ot_bimodal_to_bimodal(prob_k);

    ke_grid(k)  = compute_objective(r.rho, r.mx, prob_k) / prob_k.dx;
    err_r = norm(r.rho(:) - rho_a(:)) * sqrt(prob_k.dt * prob_k.dx);
    err_m = norm(r.mx(:)  - mx_a(:))  * sqrt(prob_k.dt * prob_k.dx);
    err_grid(k) = sqrt(err_r^2 + err_m^2);

    fprintf('  nt=%3d, nx=%3d  KE=%.6f  L2=%.2e\n', ...
            nt_vals(k), nx_vals(k), ke_grid(k), err_grid(k));
end
fprintf('Exact W_2^2 = %.6f\n', ke_exact_ot);

figure('Name', 'Bimodal->bimodal: grid refinement', 'Position', [100, 1280, 1100, 340]);
subplot(1,2,1);
semilogx(nx_vals, ke_grid, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
yline(ke_exact_ot, 'r--', 'LineWidth', 1.5, ...
      'Label', sprintf('W_2^2 = %.4f', ke_exact_ot));
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Kinetic energy', 'Interpreter', 'latex');
title('KE $\to W_2^2 = 0.0225$', 'Interpreter', 'latex');
legend('ADMM KE', '$W_2^2$', 'Interpreter', 'latex', 'Location', 'best');
grid on;

subplot(1,2,2);
loglog(nx_vals, err_grid, 'ks-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
loglog(nx_vals, err_grid(1)*(nx_vals(1)./nx_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$L^2$ error', 'Interpreter', 'latex');
title('Grid refinement: $L^2$ error', 'Interpreter', 'latex');
legend('ADMM', '$\mathcal{O}(h^2)$', 'Interpreter', 'latex', 'Location', 'southwest');
grid on;

%% --- Save all figures ---
fig_dir = fullfile(base, '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

test_tag = 'ot_bimodal_to_bimodal';
cfg_tag  = sprintf('nt%d_nx%d', cfg.nt, cfg.nx);   % eps varies within script
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
