% TEST_ADMM_OSCILLATION_VS_PROJ
%
% Checks whether the ADMM oscillations at large eps are caused by projection
% inaccuracy or by the ADMM itself limit-cycling.
%
% For each eps:
%   - Runs ADMM and records the full residual history
%   - Computes the FP residual of the final staggered variable x
%   - Computes the oscillation amplitude of the ADMM residual (last 100 iters)
%
% Key comparison:
%   FP_res_x  : how well the final x satisfies the FP equation
%               (floor set by projection inaccuracy)
%   admm_osc  : amplitude of ADMM residual oscillations at the end
%
%   If admm_osc >> FP_res_x : ADMM is limit-cycling, projection is not the cause
%   If admm_osc ~  FP_res_x : projection inaccuracy is setting the floor

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config ---
cfg          = cfg_ladmm_gaussian();
cfg.nt       = 128;
cfg.nx       = 128;
cfg.max_iter = 2000;
cfg.tol      = 1e-12;   % very tight: don't stop early, let it run fully

prob_def = prob_gaussian();
eps_vals = [0.1, 0.5, 1, 2, 4];
n_eps    = numel(eps_vals);
tail     = 100;   % number of final iterations used to estimate oscillation amplitude

%% --- Allocate ---
fp_res_x  = zeros(n_eps, 1);
admm_osc  = zeros(n_eps, 1);
admm_mean = zeros(n_eps, 1);
iters     = zeros(n_eps, 1);

fprintf('%6s  %10s  %10s  %10s  %8s\n', ...
    'eps', 'FP_res_x', 'admm_osc', 'admm_mean', 'iters');

residual_curves = cell(n_eps, 1);

for i = 1:n_eps
    vareps     = eps_vals(i);
    cfg.vareps = vareps;

    problem = setup_problem(cfg, prob_def);
    result  = discretize_then_optimize(cfg, problem);

    % --- FP residual of final staggered x ---
    fp_res_x(i) = compute_fp_res_stag(result.rho_stag, result.mx_stag, problem, vareps);

    % --- ADMM residual oscillation in last `tail` iterations ---
    res_curve = result.residual;
    n_iters   = numel(res_curve);
    tail_i    = res_curve(max(1, n_iters-tail+1):end);
    admm_osc(i)  = (max(tail_i) - min(tail_i)) / 2;   % half peak-to-peak
    admm_mean(i) = mean(tail_i);
    iters(i)     = n_iters;

    residual_curves{i} = res_curve;

    fprintf('%6.3g  %10.3e  %10.3e  %10.3e  %8d\n', ...
        vareps, fp_res_x(i), admm_osc(i), admm_mean(i), n_iters);
end

%% --- Figure 1: ADMM residual curves ---
figure('Name', 'ADMM residual curves', 'Position', [50 50 900 350]);
colors = parula(n_eps);
hold on;
for i = 1:n_eps
    semilogy(residual_curves{i}, '-', 'Color', colors(i,:), 'LineWidth', 1.2, ...
        'DisplayName', sprintf('eps=%.3g', eps_vals(i)));
end
legend('Location', 'best', 'FontSize', 8);
xlabel('ADMM iteration');
ylabel('Residual ||y^{k+1} - y^k||');
title(sprintf('ADMM residual histories  (nt=%d, nx=%d, max_iter=%d)', ...
    cfg.nt, cfg.nx, cfg.max_iter));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('admm_osc_residuals_nt%d_nx%d.png', cfg.nt, cfg.nx)));

%% --- Figure 2: Comparison bar chart ---
figure('Name', 'Oscillation vs projection error', 'Position', [50 450 700 320]);
x_pos = 1:n_eps;
bar_w = 0.35;
b1 = bar(x_pos - bar_w/2, fp_res_x,  bar_w, 'FaceColor', [0.2 0.5 0.8]);
hold on;
b2 = bar(x_pos + bar_w/2, admm_osc, bar_w, 'FaceColor', [0.8 0.3 0.2]);
set(gca, 'YScale', 'log');
set(gca, 'XTick', x_pos, 'XTickLabel', arrayfun(@(e) sprintf('%.3g',e), eps_vals, 'UniformOutput', false));
xlabel('eps');
ylabel('Magnitude');
legend([b1 b2], {'FP res of final x (proj floor)', 'ADMM oscillation amplitude'}, ...
    'Location', 'northwest');
title('ADMM oscillation amplitude vs projection inaccuracy');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('admm_osc_vs_proj_nt%d_nx%d.png', cfg.nt, cfg.nx)));

fprintf('\nInterpretation:\n');
fprintf('  admm_osc >> FP_res_x  ->  ADMM is limit-cycling; projection not the cause\n');
fprintf('  admm_osc ~  FP_res_x  ->  projection inaccuracy is setting the floor\n');

%% =========================================================
%  Local function
%% =========================================================

function res = compute_fp_res_stag(rho_stag, mx_stag, problem, vareps)
    ops     = problem.ops;
    rho0    = problem.rho0;
    rho1    = problem.rho1;
    nt      = problem.nt;
    dt      = problem.dt;
    dx      = problem.dx;
    zeros_x = zeros(nt, 1);

    lap_mu = ops.deriv_x_at_phi( ...
                 ops.deriv_x_at_m( ...
                     ops.interp_t_at_phi(rho_stag, rho0, rho1)), ...
                 zeros_x, zeros_x);

    f = ops.deriv_t_at_phi(rho_stag, rho0, rho1) ...
      + ops.deriv_x_at_phi(mx_stag, zeros_x, zeros_x) ...
      - vareps * lap_mu;

    res = sqrt(dt * dx) * norm(f(:));
end
