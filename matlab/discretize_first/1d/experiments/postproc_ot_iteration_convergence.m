% POSTPROC_OT_ITERATION_CONVERGENCE  Iteration-level convergence to exact solution.
%
%   Problem  : Gaussian N(1/3, 0.05^2) -> N(2/3, 0.05^2),  eps = 0 (pure OT)
%   Reference: analytical_gaussian  (staggered-grid exact solution)
%
%   Tracks  ||x^k - x*||  at every LADMM iteration, where x = (rho_stag, mx_stag)
%   and x* = (rho_ana, mx_ana).  Curves are shown for several gamma values.
%
%   Output (results/paper/):
%     ot_iteration_convergence.pdf

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% -----------------------------------------------------------------------
%% Global rendering
%% -----------------------------------------------------------------------
set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultAxesTickLabelInterpreter',     'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

%% -----------------------------------------------------------------------
%% Parameters
%% -----------------------------------------------------------------------
NT = 128;
NX = 128;

gamma_vals = [0.1, 1, 10, 100];

cfg_base         = cfg_ladmm_gaussian();
cfg_base.vareps  = 0;
cfg_base.nt      = NT;
cfg_base.nx      = NX;
cfg_base.max_iter = 10000;
cfg_base.tol     = 1e-10;   % low tolerance so curves run long enough to show convergence

prob_def = prob_gaussian();

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'paper');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -----------------------------------------------------------------------
%% Run monitored solves
%% -----------------------------------------------------------------------
ng = numel(gamma_vals);
iter_errs = cell(ng, 1);   % iter_errs{i} = vector of ||x^k - x*|| values

for i = 1:ng
    cfg        = cfg_base;
    cfg.gamma  = gamma_vals(i);
    cfg.tau    = gamma_vals(i) + 1;   % tau > gamma (||A||^2 <= 1)

    prob = setup_problem(cfg, prob_def);
    [rho_ana, mx_ana] = analytical_gaussian(prob);

    dt = prob.dt;  dx = prob.dx;
    nt = prob.nt;  ntm = nt - 1;
    nx = prob.nx;  nxm = nx - 1;
    rho0 = prob.rho0;  rho1 = prob.rho1;
    ops  = prob.ops;

    % Precompute banded projection
    prob.banded_proj = precomp_banded_proj(prob, cfg.vareps);

    % Initial guesses
    t_stag = linspace(0, 1, ntm)';
    x0.rho = (1 - t_stag) .* rho0 + t_stag .* rho1;
    x0.mx  = zeros(nt, nxm);

    t_cc   = ((1:nt)' - 0.5) * dt;
    y0.rho = (1 - t_cc) .* rho0 + t_cc .* rho1;
    y0.mx  = zeros(nt, nx);

    b_zero = struct('rho', zeros(nt, nx), 'mx', zeros(nt, nx));

    % Operators (mirrors discretize_then_optimize)
    sigma    = 1 / cfg.gamma;
    zeros_nt = zeros(nt, 1);

    A_fn  = @(x) struct('rho', ops.interp_t_at_phi(x.rho, rho0, rho1), ...
                        'mx',  ops.interp_x_at_phi(x.mx, zeros_nt, zeros_nt));
    At_fn = @(v) struct('rho', ops.interp_t_at_rho(v.rho), ...
                        'mx',  ops.interp_x_at_m(v.mx));
    B_fn  = @(y) struct('rho', -y.rho, 'mx', -y.mx);

    prox_f1 = @(v, step) cfg.projection(v, prob, cfg);
    solve_y = @(delta, z_hat) cfg.prox_ke( ...
        struct('rho', z_hat.rho - sigma*delta.rho, ...
               'mx',  z_hat.mx  - sigma*delta.mx), sigma, prob);

    norm_fn = @(v) sqrt(dt * dx * (sum(v.rho(:).^2) + sum(v.mx(:).^2)));

    admm_opts.gamma    = cfg.gamma;
    admm_opts.tau      = cfg.tau;
    admm_opts.alpha    = cfg.alpha;
    admm_opts.max_iter = cfg.max_iter;
    admm_opts.tol      = cfg.tol;
    admm_opts.norm_fn  = norm_fn;

    % Error w.r.t. analytical solution on the staggered grid
    iter_fn = @(x, k) sqrt(dt * dx * ( ...
        norm(x.rho(:) - rho_ana(:))^2 + ...
        norm(x.mx(:)  - mx_ana(:))^2));

    fprintf('gamma=%-6g ... ', gamma_vals(i));
    [~, ~, ~, info] = ladmm_solve_monitored( ...
        prox_f1, solve_y, A_fn, At_fn, B_fn, b_zero, x0, y0, admm_opts, iter_fn);

    iter_errs{i} = info.iter_vals;
    fprintf('converged=%d  iters=%d  final_err=%.2e\n', ...
        info.converged, info.iters, info.iter_vals(end));
end

%% -----------------------------------------------------------------------
%% Figure: ||x^k - x*|| vs iteration
%% -----------------------------------------------------------------------
FS   = 11;
LW_D = 2.0;
MS   = 6;
FW   = 16.0;
FH   = 10.5;

cmap = lines(ng);

fig = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
             'PaperUnits','centimeters','PaperSize',[FW, FH]);
hold on;

for i = 1:ng
    semilogy(1:numel(iter_errs{i}), iter_errs{i}, ...
        'Color', cmap(i,:), 'LineWidth', LW_D);
end

set(gca, 'FontSize',FS, 'Box','on', 'TickDir','out', 'YScale','log');
xlabel('Iteration $k$',                                        'FontSize',FS);
ylabel('$\|(\tilde{\rho}^k,b^k) - (\tilde{\rho}^*\!,b^*)\|_{L^2}$', 'FontSize',FS);
grid on;

leg_str = arrayfun(@(g) sprintf('$\\gamma = %g$', g), gamma_vals, 'UniformOutput',false);
legend(leg_str, 'Location','northeast', 'FontSize',FS, 'Box','off');

save_fig(fig, fullfile(out_dir, sprintf('ot_iteration_convergence_nt%d_nx%d.pdf', NT, NX)));

%% -----------------------------------------------------------------------
%% Local functions
%% -----------------------------------------------------------------------
function save_fig(fig, fpath)
    try
        exportgraphics(fig, fpath, 'ContentType','vector', 'BackgroundColor','none');
    catch
        print(fig, fpath, '-dpdf', '-painters', '-r0');
    end
    fprintf('Saved: %s\n', fpath);
end
