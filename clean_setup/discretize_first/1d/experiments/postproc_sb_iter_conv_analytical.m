% POSTPROC_SB_ITER_CONV_ANALYTICAL  LADMM iterates vs exact SB solution.
%
%   Tracks  ||x^k - x*||_{L2}  (staggered grid) at every ADMM iteration
%   for several epsilon values.  Valid while mass escape from [0,1] is negligible.
%
%   Output (results/paper/):
%     sb_iter_conv_analytical_nt<NT>_nx<NX>.pdf

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultAxesTickLabelInterpreter',     'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

%% -----------------------------------------------------------------------
%% Parameters
%% -----------------------------------------------------------------------
NT = 256;   NX = 256;
eps_vals = [1e-10, 1e-4, 1e-3, 1e-2];

cfg_base          = cfg_ladmm_gaussian();
cfg_base.vareps   = 0;      % overridden per run
cfg_base.nt       = NT;
cfg_base.nx       = NX;
cfg_base.gamma    = 10;
cfg_base.tau      = 11;
cfg_base.max_iter = 20000;
cfg_base.tol      = 1e-10;

prob_def = prob_gaussian();
out_dir  = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'paper');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -----------------------------------------------------------------------
%% Run monitored solves
%% -----------------------------------------------------------------------
ne        = numel(eps_vals);
iter_errs = cell(ne, 1);

for i = 1:ne
    cfg        = cfg_base;
    cfg.vareps = eps_vals(i);

    prob = setup_problem(cfg, prob_def);
    [rho_ana, mx_ana] = analytical_sb_gaussian(prob, eps_vals(i));

    dt = prob.dt;  dx = prob.dx;
    nt = prob.nt;  ntm = nt - 1;
    nx = prob.nx;  nxm = nx - 1;
    rho0 = prob.rho0;  rho1 = prob.rho1;
    ops  = prob.ops;

    prob.banded_proj = precomp_banded_proj(prob, cfg.vareps);

    t_stag = linspace(0, 1, ntm)';
    x0.rho = (1 - t_stag) .* rho0 + t_stag .* rho1;
    x0.mx  = zeros(nt, nxm);

    t_cc   = ((1:nt)' - 0.5) * dt;
    y0.rho = (1 - t_cc) .* rho0 + t_cc .* rho1;
    y0.mx  = zeros(nt, nx);

    b_zero   = struct('rho', zeros(nt, nx), 'mx', zeros(nt, nx));
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

    iter_fn = @(x, k) sqrt(dt * dx * ( ...
        norm(x.rho(:) - rho_ana(:))^2 + ...
        norm(x.mx(:)  - mx_ana(:))^2));

    fprintf('eps=%-10g ... ', eps_vals(i));
    [~, ~, ~, info] = ladmm_solve_monitored( ...
        prox_f1, solve_y, A_fn, At_fn, B_fn, b_zero, x0, y0, admm_opts, iter_fn);

    iter_errs{i} = info.iter_vals;
    fprintf('converged=%d  iters=%d  final_err=%.2e\n', ...
        info.converged, info.iters, info.iter_vals(end));
end

%% -----------------------------------------------------------------------
%% Figure
%% -----------------------------------------------------------------------
FS   = 11;   LW_D = 2.0;   FW = 16.0;   FH = 10.5;
cmap = lines(ne);

fig = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
             'PaperUnits','centimeters','PaperSize',[FW, FH]);
hold on;

for i = 1:ne
    plot(1:numel(iter_errs{i}), iter_errs{i}, ...
        'Color', cmap(i,:), 'LineWidth', LW_D);
end

set(gca, 'XScale','log', 'YScale','log', 'FontSize',FS, 'Box','on', 'TickDir','out');
xlabel('Iteration $k$', 'FontSize',FS);
ylabel('$\|(\tilde{\rho}^k,b^k) - (\tilde{\rho}^*\!,b^*)\|_{L^2}$', 'FontSize',FS);
grid on;

leg_str = arrayfun(@(e) sprintf('$\\varepsilon = 10^{%d}$', round(log10(e))), ...
    eps_vals, 'UniformOutput', false);
legend(leg_str, 'Location','northeast', 'FontSize',FS, 'Box','off');

save_fig(fig, fullfile(out_dir, ...
    sprintf('sb_iter_conv_analytical_nt%d_nx%d.pdf', NT, NX)));

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
