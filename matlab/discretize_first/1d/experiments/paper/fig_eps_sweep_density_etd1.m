% FIG_EPS_SWEEP_DENSITY_ETD1
%   2x2 density evolution figure for ETD1-ADMM across four epsilon values.
%   Each panel = one eps; all time snapshots for that eps overlaid.
%   Reference: analytical SB  for eps <= ANA_EPS
%              Sinkhorn (Neumann BCs) for eps >  ANA_EPS
%
%   Note on eps=100: the diffusion timescale sigma^2/(2*eps)~1e-5 is far
%   below dt=1/NT~0.008, so even the first staggered time step shows a
%   nearly uniform density.  Increase NT if you need to resolve the
%   peaked initial/final states for large eps.
%
%   Output (results/paper/):
%     sb_eps_sweep_density_etd1_nt<NT>_nx<NX>.pdf

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'setup_paths.m'));

set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultAxesTickLabelInterpreter',     'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

%% -----------------------------------------------------------------------
%% Parameters
%% -----------------------------------------------------------------------
eps_vals = [1e-4,  1e-2,  1,    100];
ANA_EPS  = 1e-3;   % use analytical SB reference for eps <= this

% Time fractions per eps panel.  For eps=1 and eps=100 add times within
% 2-5 grid points of the boundary (expressed as multiples of dt = 1/NT).
t_slices = {
    [0.10, 0.30, 0.70, 0.90],                                      % eps = 1e-4
    [0.10, 0.30, 0.70, 0.90],                                      % eps = 1e-2
    [2/NT, 5/NT, 0.05, 0.15,  0.85, 0.95,  1-5/NT, 1-2/NT],      % eps = 1
    [1/NT, 2/NT, 0.01, 0.03,  0.97, 0.99,  1-2/NT, 1-1/NT]       % eps = 100
};

NT       = 256;
NX       = 256;
GAMMA    = 100;
TAU      = 101;
MAX_ITER = 10000;
TOL      = 1e-10;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results', 'paper');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -----------------------------------------------------------------------
%% Solve and collect reference for each eps
%% -----------------------------------------------------------------------
ne       = numel(eps_vals);
prob_def = prob_gaussian();

cfg_base          = cfg_ladmm_gaussian_expsemi();
cfg_base.nt       = NT;
cfg_base.nx       = NX;
cfg_base.gamma    = GAMMA;
cfg_base.tau      = TAU;
cfg_base.max_iter = MAX_ITER;
cfg_base.tol      = TOL;

rho_num    = cell(ne, 1);
rho_ref    = cell(ne, 1);
prob_arr   = cell(ne, 1);
ref_labels = cell(ne, 1);

for i = 1:ne
    eps_i        = eps_vals(i);
    cfg_i        = cfg_base;
    cfg_i.vareps = eps_i;
    prob_i       = setup_problem(cfg_i, prob_def);
    prob_arr{i}  = prob_i;

    fprintf('Running ETD1-ADMM  eps=%.1e ... ', eps_i);
    res          = discretize_then_optimize(cfg_i, prob_i);
    rho_num{i}   = res.rho_stag;
    fprintf('iters=%d  conv=%d  wall=%.1fs\n', res.iters, res.converged, res.walltime);

    if eps_i <= ANA_EPS
        [rho_ana_i, ~] = analytical_sb_gaussian(prob_i, eps_i);
        rho_ref{i}     = rho_ana_i;
        ref_labels{i}  = 'Analytical';
    else
        fprintf('  Sinkhorn reference    eps=%.1e ... ', eps_i);
        cfg_sk.vareps       = eps_i;
        cfg_sk.max_iter     = 500;
        cfg_sk.tol          = 1e-10;
        cfg_sk.precomp_heat = @precomp_heat_neumann;
        res_sk             = sinkhorn_hopf_cole(prob_i, cfg_sk);
        rho_ref{i}         = res_sk.rho(2:prob_i.nt, :);
        ref_labels{i}      = 'Sinkhorn';
        fprintf('iters=%d  wall=%.1fs\n', res_sk.iters, res_sk.walltime);
    end
end

%% -----------------------------------------------------------------------
%% Common y-axis: probability density (rho / dx), anchored to reference peak
%% -----------------------------------------------------------------------
dx   = prob_arr{1}.dx;
ymax = 0;
for i = 1:ne
    ymax = max(ymax, max(rho_ref{i}(:)) / dx);
end
ylims = [0, ymax * 1.12];

%% -----------------------------------------------------------------------
%% Figure: 2 rows x 2 cols, one eps per panel
%% -----------------------------------------------------------------------
FS   = 11;   LW = 1.5;   LW_M = 0.9;   MS = 3;
FW   = 18.0; FH = 15.0;

fig = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
             'PaperUnits','centimeters','PaperSize',[FW, FH]);
tl  = tiledlayout(2, 2, 'TileSpacing','compact', 'Padding','loose');

for i = 1:ne
    ax     = nexttile;
    hold on;

    prob_i  = prob_arr{i};
    ntm_i   = prob_i.nt - 1;
    dt_i    = prob_i.dt;
    dx_i    = prob_i.dx;
    t_rho_i = (1:ntm_i)' * dt_i;
    xx_i    = prob_i.xx;
    nx_i    = prob_i.nx;
    stride  = max(1, floor(nx_i / 40));
    t_req   = t_slices{i};

    ns_i = numel(t_req);
    cmap = lines(ns_i);
    h_t  = gobjects(ns_i, 1);
    for j = 1:ns_i
        [~, k] = min(abs(t_rho_i - t_req(j)));
        t_act  = t_rho_i(k);
        col    = cmap(j, :);
        idx    = 1:stride:nx_i;

        % Reference (solid line)
        plot(xx_i, rho_ref{i}(k,:) / dx_i, '-', ...
            'Color', col, 'LineWidth', LW);
        % ETD1 (open circles, subsampled)
        plot(xx_i(idx), rho_num{i}(k, idx) / dx_i, 'o', ...
            'Color', col, 'MarkerSize', MS, ...
            'MarkerFaceColor', 'none', 'LineWidth', LW_M);

        % Colored proxy line for time legend
        h_t(j) = plot(nan, nan, '-', 'Color', col, 'LineWidth', LW, ...
            'DisplayName', sprintf('$t=%.3g$', t_act));
    end

    % Method proxy lines
    h_ref = plot(nan, nan, 'k-',  'LineWidth', LW,  'DisplayName', ref_labels{i});
    h_etd = plot(nan, nan, 'ko',  'MarkerSize', MS, 'MarkerFaceColor', 'none', ...
        'LineWidth', LW_M, 'DisplayName', 'ETD1');

    legend([h_ref; h_etd; h_t], 'Location', 'best', 'FontSize', FS-2, 'Box', 'off');

    ylim(ylims);
    xlim([0, 1]);
    set(ax, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');

    % Panel title: eps value + reference type
    eps_str = eps_label(eps_vals(i));
    title({eps_str, ['(' ref_labels{i} ')']}, 'FontSize', FS);

    % Axis labels: left column and bottom row only
    row = ceil(i / 2);   col_idx = mod(i-1, 2) + 1;
    if col_idx == 1
        ylabel('$\tilde{\rho}(t,x)$', 'FontSize', FS);
    end
    if row == 2
        xlabel('$x$', 'FontSize', FS);
    end
end

save_fig(fig, fullfile(out_dir, sprintf('sb_eps_sweep_density_etd1_nt%d_nx%d.pdf', NT, NX)));

%% -----------------------------------------------------------------------
%% Local helpers
%% -----------------------------------------------------------------------
function s = eps_label(eps)
    e = round(log10(eps));
    if abs(log10(eps) - e) < 0.01
        s = sprintf('$\\varepsilon = 10^{%d}$', e);
    else
        s = sprintf('$\\varepsilon = %g$', eps);
    end
end

function save_fig(fig, fpath)
    try
        exportgraphics(fig, fpath, 'ContentType', 'vector', 'BackgroundColor', 'none');
    catch
        print(fig, fpath, '-dpdf', '-painters', '-r0');
    end
    fprintf('Saved: %s\n', fpath);
end
