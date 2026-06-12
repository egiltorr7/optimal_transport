% POSTPROC_SB_ITER_CONV_SINKHORN  LADMM iterates vs converged Sinkhorn (Neumann BCs).
%
%   For each epsilon, Sinkhorn is run to tight tolerance on the same bounded
%   domain [0,1] with reflecting (Neumann) BCs.  The converged Sinkhorn
%   solution is interpolated to the ADMM staggered grid and used as reference.
%
%   Grid alignment:
%     Sinkhorn rho:  (nt+1 x nx) at t = 0, dt, ..., T  (edge times)
%     ADMM x.rho:   (ntm  x nx) at t = dt, ..., (nt-1)*dt
%     -> rho_ref = result.rho(2:nt, :)
%
%     Sinkhorn mx:   (nt+1 x nxm) at edge times
%     ADMM x.mx:    (nt   x nxm) at t = 0.5*dt, 1.5*dt, ..., (nt-0.5)*dt
%     -> mx_ref(k,:) = 0.5*(result.mx(k,:) + result.mx(k+1,:))
%
%   Output (results/paper/):
%     sb_iter_conv_sinkhorn_nt<NT>_nx<NX>.pdf

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultAxesTickLabelInterpreter',     'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

%% -----------------------------------------------------------------------
%% Parameters
%% -----------------------------------------------------------------------
NT = 128;   NX = 128;
eps_vals = [1e-2, 1e-1, 1];

cfg_base          = cfg_ladmm_gaussian();
cfg_base.vareps   = 0;      % overridden per run
cfg_base.nt       = NT;
cfg_base.nx       = NX;
cfg_base.gamma    = 10;
cfg_base.tau      = 101;
cfg_base.max_iter = 10000;
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

    dt = prob.dt;  dx = prob.dx;
    nt = prob.nt;  ntm = nt - 1;
    nx = prob.nx;  nxm = nx - 1;
    rho0 = prob.rho0;  rho1 = prob.rho1;
    ops  = prob.ops;

    % --- Sinkhorn reference (tight tolerance, Neumann BCs) ---
    fprintf('eps=%-6g  Sinkhorn ... ', eps_vals(i));
    cfg_sink.vareps       = eps_vals(i);
    cfg_sink.max_iter     = 2000;
    cfg_sink.tol          = 1e-12;
    cfg_sink.precomp_heat = @precomp_heat_neumann;
    res_sink = sinkhorn_hopf_cole(prob, cfg_sink);
    fprintf('iters=%d  err=%.2e\n', res_sink.iters, res_sink.error);

    % Interpolate Sinkhorn solution to ADMM staggered grid
    rho_ref = res_sink.rho(2:nt, :);                              % (ntm x nx)
    mx_ref  = 0.5*(res_sink.mx(1:nt,:) + res_sink.mx(2:nt+1,:)); % (nt  x nxm)

    % --- ADMM monitored solve ---
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
        norm(x.rho(:) - rho_ref(:))^2 + ...
        norm(x.mx(:)  - mx_ref(:))^2));

    fprintf('         ADMM  ... ');
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
ylabel('$\|(\tilde{\rho}^k,b^k) - (\tilde{\rho}^*_\mathrm{Sink},b^*_\mathrm{Sink})\|_{L^2}$', ...
    'FontSize',FS);
grid on;

leg_str = arrayfun(@(e) sprintf('$\\varepsilon = 10^{%d}$', round(log10(e))), ...
    eps_vals, 'UniformOutput', false);
legend(leg_str, 'Location','northeast', 'FontSize',FS, 'Box','off');

save_fig(fig, fullfile(out_dir, ...
    sprintf('sb_iter_conv_sinkhorn_nt%d_nx%d.pdf', NT, NX)));

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
