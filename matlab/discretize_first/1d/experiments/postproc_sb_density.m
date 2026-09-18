% POSTPROC_SB_DENSITY  Density evolution figures for the SB problem.
%
%   Produces one figure per epsilon value showing the full time evolution.
%   Each figure mirrors the OT density figure style:
%     - Solid coloured lines  : reference solution
%     - Open coloured circles : LADMM (staggered grid)
%
%   eps = 0    : analytical_gaussian
%   eps = 1e-4 : analytical_sb_gaussian
%   eps = 0.1  : sinkhorn_hopf_cole (Neumann BCs)
%   eps = 1    : sinkhorn_hopf_cole (Neumann BCs)
%
%   Output (results/paper/):
%     sb_density_eps<tag>_nt<NT>_nx<NX>.pdf   (one file per epsilon)

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
eps_vals = [1e-2, 0.1, 1, 4];
t_show   = [0.10, 0.25, 0.50, 0.75, 0.90];

ne    = numel(eps_vals);
nshow = numel(t_show);

cfg_base          = cfg_ladmm_gaussian();
cfg_base.nt       = NT;
cfg_base.nx       = NX;
cfg_base.max_iter = 10000;
cfg_base.tol      = 1e-10;
cfg_base.gamma = 100;
cfg_base.tau   = 101;

prob_def = prob_gaussian();
out_dir  = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'paper');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% Style (matches OT density figure)
FS   = 11;
LW_D = 2.0;
MS   = 6;
FW   = 16.0;
FH   = 10.5;
cmap   = lines(nshow);
stride = max(1, floor(NX / 60));

%% -----------------------------------------------------------------------
%% Compute: ADMM solves + reference solutions
%% -----------------------------------------------------------------------
res_all  = cell(ne, 1);
ref_rho  = cell(ne, 1);
prob_all = cell(ne, 1);

for i = 1:ne
    cfg        = cfg_base;
    cfg.vareps = eps_vals(i);

    prob = setup_problem(cfg, prob_def);
    fprintf('\n=== eps=%-6g ===\n', eps_vals(i));

    res_all{i}  = discretize_then_optimize(cfg, prob);
    prob_all{i} = prob;
    fprintf('  ADMM: iters=%d  converged=%d  wall=%.1fs\n', ...
        res_all{i}.iters, res_all{i}.converged, res_all{i}.walltime);

    if eps_vals(i) == 0
        [rho_ref{i}, ~] = analytical_gaussian(prob);
    elseif eps_vals(i) <= 1e-4
        [rho_ref{i}, ~] = analytical_sb_gaussian(prob, eps_vals(i));
    else
        fprintf('  Sinkhorn ... ');
        cfg_sink.vareps       = eps_vals(i);
        cfg_sink.max_iter     = 2000;
        cfg_sink.tol          = 1e-12;
        cfg_sink.precomp_heat = @precomp_heat_neumann;
        res_sink   = sinkhorn_hopf_cole(prob, cfg_sink);
        fprintf('iters=%d  err=%.2e\n', res_sink.iters, res_sink.error);
        rho_ref{i} = res_sink.rho(2:prob.nt, :);
    end
end

%% -----------------------------------------------------------------------
%% Plot: 2x2 figure, one panel per epsilon
%% -----------------------------------------------------------------------
FW2 = 30.0;   FH2 = 18.0;

fig = figure('Units','centimeters','Position',[2, 2, FW2, FH2], ...
             'PaperUnits','centimeters','PaperSize',[FW2, FH2]);
tl  = tiledlayout(2, 2, 'TileSpacing','compact', 'Padding','loose');

panel_labels = {'(a)','(b)','(c)','(d)'};
leg_handles  = [];   % filled on first tile, reused for shared legend

for i = 1:ne
    prob = prob_all{i};
    res  = res_all{i};
    ntm  = prob.nt - 1;
    dt   = prob.dt;   dx = prob.dx;   xx = prob.xx;

    nexttile;
    hold on;

    for j = 1:nshow
        ti = max(1, min(ntm, round(t_show(j) / dt)));

        rho_exact = rho_ref{i}(ti, :)   / dx;
        rho_num   = res.rho_stag(ti, :) / dx;

        plot(xx, rho_exact, '-', 'Color',cmap(j,:), ...
            'LineWidth',LW_D, 'HandleVisibility','off');

        idx = 1:stride:NX;
        plot(xx(idx), rho_num(idx), 'o', 'Color',cmap(j,:), ...
            'MarkerSize',MS-1, 'MarkerFaceColor','none', 'LineWidth',0.9, ...
            'HandleVisibility','off');
    end

    % Build legend handles once (first panel) for the shared legend
    if i == 1
        h_ref  = plot(nan,nan, 'k-', 'LineWidth',LW_D);
        h_num  = plot(nan,nan, 'ko', 'MarkerSize',MS-1, 'MarkerFaceColor','none', ...
                      'LineWidth',0.9, 'LineStyle','none');
        h_time = gobjects(nshow,1);
        for j = 1:nshow
            h_time(j) = plot(nan,nan, '-', 'Color',cmap(j,:), 'LineWidth',LW_D);
        end
        leg_handles = [h_ref; h_num; h_time];
    end

    xlabel('$x$',                 'FontSize',FS);
    ylabel('$\tilde{\rho}(t,x)$', 'FontSize',FS);
    title(sprintf('%s\\quad $\\varepsilon = %s$', panel_labels{i}, eps_str(eps_vals(i))), ...
        'FontSize',FS+1);
    set(gca, 'FontSize',FS, 'Box','on', 'TickDir','out');
    grid on;
end

% Shared horizontal legend at the bottom of the layout
leg_str = [{'Reference', 'LADMM'}, ...
           arrayfun(@(t) sprintf('$t=%.2f$',t), t_show, 'UniformOutput',false)];
lgd = legend(leg_handles, leg_str, 'Orientation','horizontal', ...
             'FontSize',FS, 'Box','off');
lgd.Layout.Tile = 'south';

fname = sprintf('sb_density_nt%d_nx%d.pdf', NT, NX);
save_fig(fig, fullfile(out_dir, fname));

%% -----------------------------------------------------------------------
%% Local functions
%% -----------------------------------------------------------------------
function s = eps_str(e)
    if e == 0,   s = '0';
    elseif e==1, s = '1';
    else,        s = sprintf('10^{%d}', round(log10(e)));
    end
end

function s = eps_tag(e)
    if e == 0,   s = '0';
    else,        s = sprintf('1e%d', round(log10(e)));
    end
end

function save_fig(fig, fpath)
    try
        exportgraphics(fig, fpath, 'ContentType','vector', 'BackgroundColor','none');
    catch
        print(fig, fpath, '-dpdf', '-painters', '-r0');
    end
    fprintf('Saved: %s\n', fpath);
end
