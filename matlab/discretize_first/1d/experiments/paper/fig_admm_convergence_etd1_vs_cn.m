% FIG_ADMM_CONVERGENCE_ETD1_VS_CN
%   ADMM convergence to the analytical SB solution for ETD1 and CN.
%
%   Tracks per iteration:
%     - ||x^k - x*||_{L2}        (error to analytical SB)
%     - ||x^{k+1} - x^k||_{L2}  (primal step)
%   Plus a figure showing the density evolution at several times.
%
%   Output (results/paper/):
%     admm_error_to_analytical_nt<NT>_nx<NX>_eps1e-08.pdf
%     admm_primal_step_nt<NT>_nx<NX>_eps1e-08.pdf
%     admm_density_nt<NT>_nx<NX>_eps1e-08.pdf

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'setup_paths.m'));

set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultAxesTickLabelInterpreter',     'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

%% -----------------------------------------------------------------------
%% Parameters
%% -----------------------------------------------------------------------
VAREPS   = 1e-8;
NT       = 256;
NX       = 256;
GAMMA    = 100;
TAU      = 101;
MAX_ITER = 10000;
TOL      = 1e-10;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results', 'paper');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -----------------------------------------------------------------------
%% Problem setup (shared grid)
%% -----------------------------------------------------------------------
cfg_base          = cfg_ladmm_gaussian();
cfg_base.vareps   = VAREPS;
cfg_base.nt       = NT;
cfg_base.nx       = NX;
cfg_base.gamma    = GAMMA;
cfg_base.tau      = TAU;
cfg_base.max_iter = MAX_ITER;
cfg_base.tol      = TOL;

prob_def  = prob_gaussian();
prob_base = setup_problem(cfg_base, prob_def);

[rho_ana, mx_ana] = analytical_sb_gaussian(prob_base, VAREPS);

dt   = prob_base.dt;    dx  = prob_base.dx;
nt   = prob_base.nt;    ntm = nt - 1;
nx   = prob_base.nx;    nxm = nx - 1;
rho0 = prob_base.rho0;
rho1 = prob_base.rho1;
ops  = prob_base.ops;
xx   = prob_base.xx;

%% -----------------------------------------------------------------------
%% ADMM wiring (shared by both methods)
%% -----------------------------------------------------------------------
t_stag = linspace(0, 1, ntm)';
x0.rho = (1 - t_stag) .* rho0 + t_stag .* rho1;
x0.mx  = zeros(nt, nxm);

t_cc   = ((1:nt)' - 0.5) * dt;
y0.rho = (1 - t_cc) .* rho0 + t_cc .* rho1;
y0.mx  = zeros(nt, nx);

b_zero   = struct('rho', zeros(nt, nx), 'mx', zeros(nt, nx));
sigma    = 1 / GAMMA;
zeros_nt = zeros(nt, 1);

A_fn  = @(x) struct('rho', ops.interp_t_at_phi(x.rho, rho0, rho1), ...
                    'mx',  ops.interp_x_at_phi(x.mx, zeros_nt, zeros_nt));
At_fn = @(v) struct('rho', ops.interp_t_at_rho(v.rho), ...
                    'mx',  ops.interp_x_at_m(v.mx));
B_fn  = @(y) struct('rho', -y.rho, 'mx', -y.mx);

norm_fn = @(v) sqrt(dt * dx * (sum(v.rho(:).^2) + sum(v.mx(:).^2)));

admm_opts.gamma    = GAMMA;
admm_opts.tau      = TAU;
admm_opts.alpha    = cfg_base.alpha;
admm_opts.max_iter = MAX_ITER;
admm_opts.tol      = TOL;
admm_opts.norm_fn  = norm_fn;

iter_fn = @(x, ~) sqrt(dt * dx * ( ...
    norm(x.rho(:) - rho_ana(:))^2 + ...
    norm(x.mx(:)  - mx_ana(:))^2));

%% -----------------------------------------------------------------------
%% Run CN (Crank-Nicolson)
%% -----------------------------------------------------------------------
prob_cn             = prob_base;
prob_cn.banded_proj = precomp_banded_proj(prob_base, VAREPS);

prox_f1_cn = @(v, ~) proj_fokker_planck_banded(v, prob_cn, cfg_base);
solve_y_cn = @(delta, z_hat) prox_ke_cc( ...
    struct('rho', z_hat.rho - sigma*delta.rho, ...
           'mx',  z_hat.mx  - sigma*delta.mx), sigma, prob_base);

fprintf('Running CN-ADMM    (eps=%.1e, nt=%d, nx=%d)...\n', VAREPS, NT, NX);
[iter_err_cn, primal_step_cn, x_cn, info_cn] = run_admm_paper( ...
    prox_f1_cn, solve_y_cn, A_fn, At_fn, B_fn, b_zero, x0, y0, admm_opts, iter_fn, norm_fn);
fprintf('  converged=%d  iters=%d  final_err=%.2e  wall=%.1fs\n', ...
    info_cn.converged, info_cn.iters, iter_err_cn(end), info_cn.walltime);

%% -----------------------------------------------------------------------
%% Run ETD1 (exact semigroup)
%% -----------------------------------------------------------------------
cfg_etd              = cfg_base;
cfg_etd.name         = 'ladmm_gaussian_expsemi';
cfg_etd.projection   = @proj_fokker_planck_expsemi;
prob_etd             = prob_base;
prob_etd.expsemi_proj = precomp_expsemi_proj(prob_base, VAREPS);

prox_f1_etd = @(v, ~) proj_fokker_planck_expsemi(v, prob_etd, cfg_etd);
solve_y_etd = @(delta, z_hat) prox_ke_cc( ...
    struct('rho', z_hat.rho - sigma*delta.rho, ...
           'mx',  z_hat.mx  - sigma*delta.mx), sigma, prob_base);

fprintf('Running ETD1-ADMM  (eps=%.1e, nt=%d, nx=%d)...\n', VAREPS, NT, NX);
[iter_err_etd, primal_step_etd, x_etd, info_etd] = run_admm_paper( ...
    prox_f1_etd, solve_y_etd, A_fn, At_fn, B_fn, b_zero, x0, y0, admm_opts, iter_fn, norm_fn);
fprintf('  converged=%d  iters=%d  final_err=%.2e  wall=%.1fs\n', ...
    info_etd.converged, info_etd.iters, iter_err_etd(end), info_etd.walltime);

%% -----------------------------------------------------------------------
%% Figure 1: Error to analytical SB  ||x^k - x*||
%% -----------------------------------------------------------------------
FS  = 11;   LW  = 2.0;   FW  = 16.0;   FH  = 10.5;
col_cn  = [0.216, 0.494, 0.722];   % blue
col_etd = [0.894, 0.102, 0.110];   % red
ftag = sprintf('nt%d_nx%d_eps%.0e', NT, NX, VAREPS);

fig1 = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
              'PaperUnits','centimeters','PaperSize',[FW, FH]);
hold on;
plot(1:numel(iter_err_cn),  iter_err_cn,  '-',  'Color', col_cn,  'LineWidth', LW, 'DisplayName', 'CN');
plot(1:numel(iter_err_etd), iter_err_etd, '--', 'Color', col_etd, 'LineWidth', LW, 'DisplayName', 'ETD1');
set(gca, 'XScale','log', 'YScale','log', 'FontSize',FS, 'Box','on', 'TickDir','out');
xlabel('Iteration $k$', 'FontSize', FS);
ylabel('$\|(\tilde{\rho}^k,\tilde{m}^k) - (\tilde{\rho}^*\!,\tilde{m}^*)\|_{L^2}$', 'FontSize', FS);
title(sprintf('ADMM convergence to analytical SB  ($\\varepsilon = 10^{%d}$)', ...
    round(log10(VAREPS))), 'FontSize', FS);
legend('Location','northeast', 'FontSize', FS, 'Box','off');
grid on;
save_fig(fig1, fullfile(out_dir, sprintf('admm_error_to_analytical_%s.pdf', ftag)));

%% -----------------------------------------------------------------------
%% Figure 2: Primal step  ||x^{k+1} - x^k||
%% -----------------------------------------------------------------------
fig2 = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
              'PaperUnits','centimeters','PaperSize',[FW, FH]);
hold on;
plot(1:numel(primal_step_cn),  primal_step_cn,  '-',  'Color', col_cn,  'LineWidth', LW, 'DisplayName', 'CN');
plot(1:numel(primal_step_etd), primal_step_etd, '--', 'Color', col_etd, 'LineWidth', LW, 'DisplayName', 'ETD1');
set(gca, 'XScale','log', 'YScale','log', 'FontSize',FS, 'Box','on', 'TickDir','out');
xlabel('Iteration $k$', 'FontSize', FS);
ylabel('$\|x^{k+1} - x^k\|_{L^2}$', 'FontSize', FS);
title(sprintf('ADMM primal step  ($\\varepsilon = 10^{%d}$)', round(log10(VAREPS))), 'FontSize', FS);
legend('Location','northeast', 'FontSize', FS, 'Box','off');
grid on;
save_fig(fig2, fullfile(out_dir, sprintf('admm_primal_step_%s.pdf', ftag)));

%% -----------------------------------------------------------------------
%% Figure 3: Density evolution
%%   Analytical = solid lines;  CN = circles;  ETD1 = triangles
%% -----------------------------------------------------------------------
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
n_t     = numel(t_fracs);
cmap    = lines(n_t);
stride  = max(1, floor(nx / 40));
MS      = 5;   LW_M = 1.0;

t_rho = (1:ntm)' * dt;   % staggered time vector

fig3 = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
              'PaperUnits','centimeters','PaperSize',[FW, FH]);
hold on;

for p = 1:n_t
    [~, k] = min(abs(t_rho - t_fracs(p)));
    idx    = 1:stride:nx;
    col    = cmap(p,:);

    plot(xx,      rho_ana(k,:),        '-',  'Color', col, 'LineWidth', LW,   'HandleVisibility','off');
    plot(xx(idx), x_cn.rho(k, idx),    'o',  'Color', col, 'LineWidth', LW_M, 'MarkerSize', MS, 'MarkerFaceColor','none', 'HandleVisibility','off');
    plot(xx(idx), x_etd.rho(k, idx),   '^',  'Color', col, 'LineWidth', LW_M, 'MarkerSize', MS, 'MarkerFaceColor','none', 'HandleVisibility','off');
end

% Method legend entries (black proxy lines)
h_ana = plot(nan,nan, 'k-',  'LineWidth', LW,   'DisplayName', 'Analytical');
h_cn  = plot(nan,nan, 'ko',  'LineWidth', LW_M, 'MarkerSize', MS, 'MarkerFaceColor','none', 'DisplayName', 'CN');
h_etd = plot(nan,nan, 'k^',  'LineWidth', LW_M, 'MarkerSize', MS, 'MarkerFaceColor','none', 'DisplayName', 'ETD1');

% Time legend entries
h_t = gobjects(n_t, 1);
for p = 1:n_t
    [~, k] = min(abs(t_rho - t_fracs(p)));
    h_t(p) = plot(nan,nan, '-', 'Color', cmap(p,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$t=%.2f$', t_rho(k)));
end

xlabel('$x$', 'FontSize', FS);
ylabel('$\tilde{\rho}(t,x)$', 'FontSize', FS);
title(sprintf('Density evolution  ($\\varepsilon = 10^{%d}$)', round(log10(VAREPS))), 'FontSize', FS);
legend([h_ana; h_cn; h_etd; h_t], 'Location','best', 'FontSize', FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;
save_fig(fig3, fullfile(out_dir, sprintf('admm_density_%s.pdf', ftag)));

%% -----------------------------------------------------------------------
%% Local functions
%% -----------------------------------------------------------------------

function [iter_err, primal_step, x_final, info] = run_admm_paper( ...
        prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts, iter_fn, norm_fn)
% ADMM loop tracking both ||x^k - x*|| and ||x^{k+1} - x^k|| in one pass.
    gamma    = opts.gamma;
    tau      = opts.tau;
    alpha    = opts.alpha;
    max_iter = opts.max_iter;
    tol      = opts.tol;

    x     = x0;
    y     = y0;
    delta = szero(b);
    By    = B_fn(y0);

    iter_err    = zeros(max_iter, 1);
    primal_step = zeros(max_iter, 1);
    residual    = zeros(max_iter, 1);

    tic;
    for k = 1:max_iter
        y_prev  = y;
        By_prev = By;

        % x-subproblem (linearised)
        x_old = x;
        r     = sadd(A_fn(x), ssub(By_prev, b));
        g     = At_fn(ssub(sscl(gamma, r), delta));
        x     = prox_f1(ssub(x, sscl(1/tau, g)), 1/tau);

        iter_err(k)    = iter_fn(x, k);
        primal_step(k) = norm_fn(ssub(x, x_old));

        % over-relaxation
        Ax    = A_fn(x);
        z_hat = sadd(sscl(alpha, Ax), sscl(1 - alpha, ssub(b, By_prev)));

        % y-subproblem (exact)
        y  = solve_y(delta, z_hat);
        By = B_fn(y);

        % dual update
        delta = ssub(delta, sscl(gamma, ssub(sadd(z_hat, By), b)));

        % convergence on ||y^{k+1} - y^k||
        residual(k) = norm_fn(ssub(y, y_prev));
        if residual(k) < tol
            iter_err    = iter_err(1:k);
            primal_step = primal_step(1:k);
            break;
        end
    end

    x_final        = x;
    info.iters     = k;
    info.converged = residual(k) < tol;
    info.walltime  = toc;
end

function c = sadd(a, b)
    c.rho = a.rho + b.rho;   c.mx = a.mx + b.mx;
end
function c = ssub(a, b)
    c.rho = a.rho - b.rho;   c.mx = a.mx - b.mx;
end
function c = sscl(s, a)
    c.rho = s * a.rho;        c.mx = s * a.mx;
end
function c = szero(a)
    c.rho = zeros(size(a.rho));   c.mx = zeros(size(a.mx));
end

function save_fig(fig, fpath)
    try
        exportgraphics(fig, fpath, 'ContentType','vector', 'BackgroundColor','none');
    catch
        print(fig, fpath, '-dpdf', '-painters', '-r0');
    end
    fprintf('Saved: %s\n', fpath);
end
