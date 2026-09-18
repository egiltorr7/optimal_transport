% FIG_GRID_REFINEMENT_ETD1_VS_CN
%   Joint grid refinement study (nt = nx = n) for CN and ETD1 on the
%   Gaussian SB problem.  Error is measured against a reference solution:
%     eps <= EPS_SINKHORN_THRESH : analytical SB solution (analytical_sb_gaussian)
%     eps >  EPS_SINKHORN_THRESH : Sinkhorn / Hopf-Cole with Neumann BCs
%                                  (sinkhorn_hopf_cole), since the analytical
%                                  formula loses accuracy for larger eps.
%
%   Grid levels : n = nt = nx in [32, 64, 128, 256]
%   Methods     : CN  (proj_fokker_planck_banded)
%                 ETD1 (proj_fokker_planck_expsemi)
%
%   Output (results/paper/):
%     sb_grid_refinement_etd1_vs_cn_eps<VAREPS>.pdf
%     LaTeX convergence table printed to terminal

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'setup_paths.m'));

set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultAxesTickLabelInterpreter',     'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

%% -----------------------------------------------------------------------
%% Parameters
%% -----------------------------------------------------------------------
VAREPS   = 1e-2;
n_vals   = [32, 64, 128, 256];

GAMMA    = 100;
TAU      = 101;
MAX_ITER = 10000;
TOL      = 1e-10;

EPS_SINKHORN_THRESH = 5e-3;   % above this, analytical SB is inaccurate; use Sinkhorn

cfg_sink_tmpl.vareps       = VAREPS;
cfg_sink_tmpl.max_iter     = 500;
cfg_sink_tmpl.tol          = 1e-10;
cfg_sink_tmpl.precomp_heat = @precomp_heat_neumann;

prob_def = prob_gaussian();

cfg_cn           = cfg_ladmm_gaussian();
cfg_cn.vareps    = VAREPS;
cfg_cn.gamma     = GAMMA;
cfg_cn.tau       = TAU;
cfg_cn.max_iter  = MAX_ITER;
cfg_cn.tol       = TOL;

cfg_etd          = cfg_ladmm_gaussian_expsemi();
cfg_etd.vareps   = VAREPS;
cfg_etd.gamma    = GAMMA;
cfg_etd.tau      = TAU;
cfg_etd.max_iter = MAX_ITER;
cfg_etd.tol      = TOL;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results', 'paper');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -----------------------------------------------------------------------
%% Sweep: nt = nx = n
%% -----------------------------------------------------------------------
nl = numel(n_vals);
err_cn  = zeros(nl, 1);
err_etd = zeros(nl, 1);
iters_cn  = zeros(nl, 1);
iters_etd = zeros(nl, 1);
wall_cn   = zeros(nl, 1);
wall_etd  = zeros(nl, 1);
conv_cn   = false(nl, 1);
conv_etd  = false(nl, 1);

fprintf('=== Joint refinement  nt = nx = n,  eps = %.2g ===\n\n', VAREPS);

for k = 1:nl
    n = n_vals(k);
    fprintf('n = %d\n', n);

    cfg_k    = cfg_cn;   cfg_k.nt = n;   cfg_k.nx = n;
    cfg_k_e  = cfg_etd;  cfg_k_e.nt = n; cfg_k_e.nx = n;
    prob_k   = setup_problem(cfg_k, prob_def);

    if VAREPS > EPS_SINKHORN_THRESH
        r_sk    = sinkhorn_hopf_cole(prob_k, cfg_sink_tmpl);
        rho_ref = r_sk.rho(2:prob_k.nt, :);                                    % (ntm x nx)
        mx_ref  = 0.5*(r_sk.mx(1:prob_k.nt, :) + r_sk.mx(2:prob_k.nt+1, :));   % (nt  x nxm)
        fprintf('  Sinkhorn ref (Neumann)     iters=%d  err=%.2e\n', r_sk.iters, r_sk.error);
    else
        [rho_ref, mx_ref] = analytical_sb_gaussian(prob_k, VAREPS);
    end

    [err_cn(k),  iters_cn(k),  wall_cn(k),  conv_cn(k)]  = run_level(cfg_k,   prob_k, rho_ref, mx_ref);
    [err_etd(k), iters_etd(k), wall_etd(k), conv_etd(k)] = run_level(cfg_k_e, prob_k, rho_ref, mx_ref);
end

%% -----------------------------------------------------------------------
%% Figure: L2 error vs n
%% -----------------------------------------------------------------------
FS   = 11;   LW_D = 2.0;   LW_R = 1.4;   MS = 7;
FW   = 16.0; FH   = 10.5;

col_cn  = [0.216, 0.494, 0.722];   % blue
col_etd = [0.894, 0.102, 0.110];   % red
col_h1  = [0.40, 0.40, 0.40];
col_h2  = [0.10, 0.10, 0.10];

fig = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
             'PaperUnits','centimeters','PaperSize',[FW, FH]);
hold on;

loglog(n_vals, err_cn,  'o-', 'Color',col_cn,  'LineWidth',LW_D, ...
    'MarkerFaceColor',col_cn,  'MarkerSize',MS, 'DisplayName','CN');
loglog(n_vals, err_etd, '^-', 'Color',col_etd, 'LineWidth',LW_D, ...
    'MarkerFaceColor',col_etd, 'MarkerSize',MS, 'DisplayName','ETD1');

% Reference slopes anchored to the CN curve at n=32
loglog(n_vals, err_cn(1)*(n_vals(1)./n_vals).^1, '--', 'Color',col_h1, 'LineWidth',LW_R, ...
    'DisplayName','$\mathcal{O}(n^{-1})$');
loglog(n_vals, err_cn(1)*(n_vals(1)./n_vals).^2, ':',  'Color',col_h2, 'LineWidth',LW_R, ...
    'DisplayName','$\mathcal{O}(n^{-2})$');

set(gca, 'XScale','log', 'YScale','log', 'FontSize',FS, 'Box','on', 'TickDir','out', ...
    'XTick', n_vals);
xlabel('$n = n_t = n_x$', 'FontSize', FS);
ylabel('$\|(\tilde{\rho}_h,\tilde{m}_h) - (\tilde{\rho}^*\!,\tilde{m}^*)\|_{L^2}$', ...
    'FontSize', FS);
title(sprintf('Joint grid refinement  ($\\varepsilon = %.2g$)', VAREPS), 'FontSize', FS);
legend('Location','southwest', 'FontSize',FS, 'Box','off');
grid on;

eps_tag = strrep(sprintf('%.2g', VAREPS), '.', 'p');
save_fig(fig, fullfile(out_dir, sprintf('sb_grid_refinement_etd1_vs_cn_eps%s.pdf', eps_tag)));

%% -----------------------------------------------------------------------
%% Convergence table (terminal + LaTeX)
%% -----------------------------------------------------------------------
rate_fn = @(e) log(e(1:end-1) ./ e(2:end)) ./ log(2);   % n doubles each level

r_cn  = rate_fn(err_cn);
r_etd = rate_fn(err_etd);

fprintf('\n--- Convergence table (eps=%.2g) ---\n', VAREPS);
fprintf('  %4s   %10s  %5s  %5s     %10s  %5s  %5s\n', ...
    'n', 'err_CN', 'rate', 'iters', 'err_ETD1', 'rate', 'iters');
for k = 1:nl
    if k == 1
        rcs = '  ---';  res = '  ---';
    else
        rcs = sprintf('%5.2f', r_cn(k-1));
        res = sprintf('%5.2f', r_etd(k-1));
    end
    fprintf('  %4d   %10.3e  %s  %5d     %10.3e  %s  %5d\n', ...
        n_vals(k), err_cn(k), rcs, iters_cn(k), err_etd(k), res, iters_etd(k));
end

fprintf('\n%%%% ===== LaTeX table =====\n');
fprintf('\\begin{table}[htbp]\n');
fprintf('\\centering\n');
fprintf(['\\caption{Grid convergence for SB ($\\varepsilon=%.2g$), ' ...
         'Gaussian $N(\\frac{1}{3},0.05^2)\\to N(\\frac{2}{3},0.05^2)$, ' ...
         'joint refinement $n_t=n_x=n$.}\n'], VAREPS);
fprintf('\\label{tab:sb_grid_refinement}\n');
fprintf('\\begin{tabular}{r cc cc}\n');
fprintf('\\toprule\n');
fprintf(' & \\multicolumn{2}{c}{CN} & \\multicolumn{2}{c}{ETD1} \\\\\n');
fprintf('\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n');
fprintf('$n$ & $\\|e\\|_{L^2}$ & rate & $\\|e\\|_{L^2}$ & rate \\\\\n');
fprintf('\\midrule\n');
for k = 1:nl
    if k == 1
        rcs = '---';  res = '---';
    else
        rcs = sprintf('%.2f', r_cn(k-1));
        res = sprintf('%.2f', r_etd(k-1));
    end
    fprintf('%4d & %.2e & %s & %.2e & %s \\\\\n', ...
        n_vals(k), err_cn(k), rcs, err_etd(k), res);
end
fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n');

%% -----------------------------------------------------------------------
%% Local functions
%% -----------------------------------------------------------------------
function [err_tot, iters, walltime, converged] = run_level(cfg, prob, rho_ref, mx_ref)
    r   = cfg.pipeline(cfg, prob);
    dt  = prob.dt;   dx = prob.dx;
    e_rho = norm(r.rho_stag(:) - rho_ref(:)) * sqrt(dt * dx);
    e_mx  = norm(r.mx_stag(:)  - mx_ref(:))  * sqrt(dt * dx);
    err_tot  = sqrt(e_rho^2 + e_mx^2);
    iters    = r.iters;
    walltime = r.walltime;
    converged = r.converged;
    fprintf('  %-25s  nt=%3d  nx=%3d  err=%.2e  iters=%d  %.1fs  conv=%d\n', ...
        cfg.name, cfg.nt, cfg.nx, err_tot, iters, walltime, converged);
end

function save_fig(fig, fpath)
    try
        exportgraphics(fig, fpath, 'ContentType','vector', 'BackgroundColor','none');
    catch
        print(fig, fpath, '-dpdf', '-painters', '-r0');
    end
    fprintf('Saved: %s\n', fpath);
end
