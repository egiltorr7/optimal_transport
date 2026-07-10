% POSTPROCESS_BANDED  Post-process data from test_banded.m.
%
%   Mirrors postprocess_expsemi.m but reads banded2d_gaussian_*.mat files
%   produced by the Crank-Nicolson banded FP projection.
%
%   Run this locally after copying results/data/ from the server.
%
%   Per-run figures  (one per (NT,NX,eps)):
%     admm_conv_*      -- res_x / res_y / res_primal vs iteration
%     err_vs_t_*       -- relative L2 error vs time (stag + cc)
%     density_*        -- rho snapshots at t=0.1, 0.5, 0.9 (Banded vs ref)
%
%   Per-grid figures  (one per (NT,NX)):
%     sweep_err_*      -- max rel L2 error vs eps  (stag + cc)
%     sweep_timing_*   -- wall time + iters vs eps
%
%   Cross-grid figures  (all grids, per eps):
%     refine_err_*     -- max rel error vs NX  (convergence rate)
%     refine_time_*    -- wall time vs NX

clear; close all;

base_dir = fileparts(mfilename('fullpath'));
dat_dir  = fullfile(base_dir, '..', 'results', 'data');
fig_dir  = fullfile(base_dir, '..', 'results', 'figures','paper');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

FS = 11;   LW = 1.5;   MS = 6;

%% -------------------------------------------------------------------------
%  Discover available files
% -------------------------------------------------------------------------
files = dir(fullfile(dat_dir, 'banded2d_gaussian_eps*_nt*_nx*.mat'));
if isempty(files)
    error('No data files found in %s', dat_dir);
end

% Parse (eps, NT, NX) from filenames
NF    = numel(files);
f_eps = nan(NF,1);
f_NT  = nan(NF,1);
f_NX  = nan(NF,1);
for k = 1:NF
    tok = regexp(files(k).name, ...
        'banded2d_gaussian_eps([\d.e+\-]+)_nt(\d+)_nx(\d+)', 'tokens');
    if ~isempty(tok)
        t = tok{1};
        f_eps(k) = str2double(t{1});
        f_NT(k)  = str2double(t{2});
        f_NX(k)  = str2double(t{3});
    end
end
valid = ~isnan(f_eps);
files = files(valid);   f_eps = f_eps(valid);   f_NT = f_NT(valid);   f_NX = f_NX(valid);

%% -------------------------------------------------------------------------
%  Optional filter — set to [] to process all files
% -------------------------------------------------------------------------
FILTER_NT  = [64];    % e.g. 64
FILTER_NX  = [128];    % e.g. 32
FILTER_EPS = [0.01];    % e.g. 1.0

mask = true(numel(files), 1);
if ~isempty(FILTER_NT),  mask = mask & (f_NT == FILTER_NT);                    end
if ~isempty(FILTER_NX),  mask = mask & (f_NX == FILTER_NX);                    end
if ~isempty(FILTER_EPS), mask = mask & (abs(f_eps - FILTER_EPS) < 1e-12);      end
files = files(mask);   f_eps = f_eps(mask);   f_NT = f_NT(mask);   f_NX = f_NX(mask);

EPS_VALS  = unique(f_eps);
GRID_VALS = unique(f_NX);   % NX values; NT determined by file
fprintf('Found %d files:  %d eps values,  %d grid levels\n', ...
    numel(files), numel(EPS_VALS), numel(GRID_VALS));

cmap_eps  = lines(numel(EPS_VALS));
cmap_grid = lines(numel(GRID_VALS));

%% =========================================================================
%  Per-run figures
%% =========================================================================
for k = 1:numel(files)
    d   = load(fullfile(dat_dir, files(k).name));
    tag = sprintf('nt%d_nx%d_eps%g', d.NT, d.NX, d.eps_i);
    fprintf('Processing %s ...\n', tag);

    %% --- (1) ADMM convergence history -----------------------------------
    iters_vec = (1:d.iters_bd)';

    fig = figure('Units','centimeters','Position',[2 2 20 7]);
    tl  = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

    nexttile;
    semilogy(iters_vec, d.res_x, 'LineWidth', LW);
    xlabel('Iteration', 'FontSize', FS);
    ylabel('$\|x^{k+1}-x^k\|$', 'FontSize', FS);
    title('(a) $x$-variable change', 'FontSize', FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    nexttile;
    semilogy(iters_vec, d.res_y, 'Color',[0.84 0.15 0.16], 'LineWidth', LW);
    xlabel('Iteration', 'FontSize', FS);
    ylabel('$\|y^{k+1}-y^k\|$', 'FontSize', FS);
    title('(b) $y$-variable change', 'FontSize', FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    nexttile;
    semilogy(iters_vec, d.res_primal, 'Color',[0.2 0.6 0.2], 'LineWidth', LW);
    xlabel('Iteration', 'FontSize', FS);
    ylabel('$\|Ax-y\|$', 'FontSize', FS);
    title('(c) Primal feasibility', 'FontSize', FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    sgtitle(sprintf('ADMM convergence  ($\\varepsilon=%.4g$,  $N_T=%d$,  $N_x=%d$)', ...
        d.eps_i, d.NT, d.NX), 'FontSize', FS+1, 'Interpreter', 'latex');
    savefig_both(fig, fig_dir, ['admm_conv_banded_' tag]);

    %% --- (2) Error vs time ----------------------------------------------
    fig = figure('Units','centimeters','Position',[2 2 16 7]);
    tl  = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile;
    semilogy(d.t_stag, d.rel_err_rho_stag_t, '-',  'LineWidth', LW, ...
        'DisplayName', 'Stag ($x$-var)');
    hold on;
    semilogy(d.t_cc,   d.rel_err_rho_cc_t,   '--', 'LineWidth', LW, ...
        'DisplayName', 'CC ($y$-var)');
    xlabel('$t$','FontSize',FS);
    ylabel('Relative $L^2$ error','FontSize',FS);
    title('(a) Rel. error vs time','FontSize',FS);
    legend('Location','best','FontSize',FS-1,'Box','off');
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    nexttile;
    semilogy(d.t_stag, d.err_rho_stag_t, '-',  'LineWidth', LW, ...
        'DisplayName', 'Stag ($x$-var)');
    hold on;
    semilogy(d.t_cc,   d.err_rho_cc_t,   '--', 'LineWidth', LW, ...
        'DisplayName', 'CC ($y$-var)');
    xlabel('$t$','FontSize',FS);
    ylabel('$L^2$ error','FontSize',FS);
    title('(b) Abs. error vs time','FontSize',FS);
    legend('Location','best','FontSize',FS-1,'Box','off');
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    ref_label = d.ref_type;
    sgtitle(sprintf('Error vs time  ($\\varepsilon=%.4g$,  $N_T=%d$,  $N_x=%d$,  ref: %s)', ...
        d.eps_i, d.NT, d.NX, ref_label), 'FontSize', FS+1, 'Interpreter', 'latex');
    savefig_both(fig, fig_dir, ['err_vs_t_banded_' tag]);

    %% --- (3) Density snapshots ------------------------------------------
    t_fracs = [0.1, 0.5, 0.9];
    n_snap  = numel(t_fracs);
    ntm_d   = size(d.rho_bd_stag, 1);

    fig = figure('Units','centimeters','Position',[2 2 22 10]);
    tl  = tiledlayout(2, n_snap, 'TileSpacing','compact','Padding','compact');
    clim_max = max(d.rho_ref_cc(:));

    for s = 1:n_snap
        k_idx = max(1, round(t_fracs(s) * d.nt));
        t_k   = (k_idx - 0.5) * d.dt;

        ax = nexttile(tl, s);
        imagesc(d.xx, d.yy, squeeze(d.rho_ref_cc(k_idx,:,:))');
        axis xy; colorbar; clim([0, clim_max]);
        title(sprintf('Ref (%s),  $t=%.2f$', ref_label, t_k), 'FontSize', FS);
        if s==1, ylabel('$y$','FontSize',FS); end
        set(ax,'FontSize',FS-1,'TickDir','out');

        ax = nexttile(tl, n_snap + s);
        imagesc(d.xx, d.yy, squeeze(d.rho_bd_cc(k_idx,:,:))');
        axis xy; colorbar; clim([0, clim_max]);
        title(sprintf('Banded,  $t=%.2f$', t_k), 'FontSize', FS);
        xlabel('$x$','FontSize',FS);
        if s==1, ylabel('$y$','FontSize',FS); end
        set(ax,'FontSize',FS-1,'TickDir','out');
    end
    sgtitle(sprintf('Density ($\\varepsilon=%.4g$,  $N_T=%d$,  $N_x=%d$)', ...
        d.eps_i, d.NT, d.NX), 'FontSize', FS+1, 'Interpreter', 'latex');
    savefig_both(fig, fig_dir, ['density_banded_' tag]);

    %% --- (4) Iter time histogram ----------------------------------------
    it = d.iter_times(1:d.iters_bd);
    fig = figure('Units','centimeters','Position',[2 2 12 6]);
    histogram(it * 1e3, 40, 'FaceColor',[0.13 0.47 0.71], 'EdgeColor','none');
    xlabel('Time per iteration (ms)', 'FontSize', FS);
    ylabel('Count', 'FontSize', FS);
    title(sprintf('Iter time  ($N_T=%d$,  $N_x=%d$,  $\\varepsilon=%.4g$,  median=%.1fms)', ...
        d.NT, d.NX, d.eps_i, median(it)*1e3), 'FontSize', FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;
    savefig_both(fig, fig_dir, ['iter_time_banded_' tag]);

    %% --- (5) Density along diagonal x=y ------------------------------------
    nx_d = d.nx;   ny_d = d.ny;   nt_d = d.nt;
    nd   = min(nx_d, ny_d);
    x_diag = ((1:nd) - 0.5) * d.dx;   % spatial coord along x=y

    rho_bd_diag  = zeros(nt_d, nd);
    rho_ref_diag = zeros(nt_d, nd);
    for ii = 1:nd
        rho_bd_diag(:,  ii) = d.rho_bd_cc(:,  ii, ii);
        rho_ref_diag(:, ii) = d.rho_ref_cc(:, ii, ii);
    end

    t_fracs_d = [0.1, 0.25, 0.5, 0.75, 0.9];
    ns_d      = numel(t_fracs_d);
    t_cc_d    = ((1:nt_d)' - 0.5) * d.dt;
    clrs      = cool(ns_d);

    fig = figure('Units','centimeters','Position',[2 2 14 8]);
    hold on;
    for s = 1:ns_d
        k_idx = max(1, round(t_fracs_d(s) * nt_d));
        t_k   = t_cc_d(k_idx);
        plot(x_diag, rho_ref_diag(k_idx, :), '--', 'Color', clrs(s,:), ...
            'LineWidth', LW, 'HandleVisibility', 'off');
        plot(x_diag, rho_bd_diag(k_idx, :),  '-',  'Color', clrs(s,:), ...
            'LineWidth', LW, 'DisplayName', sprintf('$t=%.2f$', t_k));
    end
    plot(nan, nan, 'k-',  'LineWidth', LW, 'DisplayName', 'Banded');
    plot(nan, nan, 'k--', 'LineWidth', LW, 'DisplayName', sprintf('Ref (%s)', ref_label));
    xlabel('$x$  (along $x=y$)', 'FontSize', FS);
    ylabel('$\rho$', 'FontSize', FS);
    title(sprintf('Slice $x=y$  ($\\varepsilon=%.4g$,  $N_T=%d$,  $N_x=%d$)', ...
        d.eps_i, d.NT, d.NX), 'FontSize', FS, 'Interpreter', 'latex');
    legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
    set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');  grid on;
    savefig_both(fig, fig_dir, ['diag_slice_banded_' tag]);

    close all;
end

%% =========================================================================
%  Per-grid sweep figures  (load sweep summary files)
%% =========================================================================
sfiles = dir(fullfile(dat_dir, 'banded2d_sweep_gaussian_nt*_nx*.mat'));

for k = 1:numel(sfiles)
    s = load(fullfile(dat_dir, sfiles(k).name));
    tag = sprintf('nt%d_nx%d', s.NT, s.NX);
    fprintf('Sweep figure: %s\n', tag);

    fig = figure('Units','centimeters','Position',[2 2 22 7]);
    tl  = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

    nexttile;
    loglog(s.sw_eps, s.sw_max_rel_err_rho_stag, '-o', 'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', [0.13 0.47 0.71], ...
        'DisplayName', 'Stag ($x$-var)');
    hold on;
    loglog(s.sw_eps, s.sw_max_rel_err_rho_cc,   '--^', 'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', [0.84 0.15 0.16], ...
        'DisplayName', 'CC ($y$-var)');
    xlabel('$\varepsilon$','FontSize',FS);
    ylabel('$\max_t$ rel. $L^2$ error','FontSize',FS);
    title('(a) Max error vs $\varepsilon$','FontSize',FS);
    legend('Location','best','FontSize',FS-1,'Box','off');
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    nexttile;
    loglog(s.sw_eps, s.sw_wall_bd,  '-o',  'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', [0.13 0.47 0.71], ...
        'DisplayName', 'Banded');
    hold on;
    loglog(s.sw_eps, s.sw_wall_ref, '--s', 'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', [0.5 0.5 0.5], ...
        'DisplayName', 'Reference');
    xlabel('$\varepsilon$','FontSize',FS);
    ylabel('Wall time (s)','FontSize',FS);
    title('(b) Wall time vs $\varepsilon$','FontSize',FS);
    legend('Location','best','FontSize',FS-1,'Box','off');
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    nexttile;
    semilogx(s.sw_eps, s.sw_iters_bd, '-o', 'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', [0.2 0.6 0.2]);
    xlabel('$\varepsilon$','FontSize',FS);
    ylabel('ADMM iterations','FontSize',FS);
    title('(c) Iterations vs $\varepsilon$','FontSize',FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    sgtitle(sprintf('$N_T=%d$,  $N_x=N_y=%d$', s.NT, s.NX), 'FontSize', FS+1, 'Interpreter', 'latex');
    savefig_both(fig, fig_dir, ['sweep_banded_' tag]);
    close all;
end

%% =========================================================================
%  Cross-grid refinement figures  (fixed eps, vary NX)
%% =========================================================================
for ei = 1:numel(EPS_VALS)
    eps_i = EPS_VALS(ei);
    mask  = abs(f_eps - eps_i) < 1e-12;
    if sum(mask) < 2, continue; end

    nxv   = f_NX(mask);
    ntv   = f_NT(mask);
    fv    = files(mask);
    [nxv_s, ord] = sort(nxv);
    fv    = fv(ord);   ntv = ntv(ord);

    err_stag = nan(numel(fv), 1);
    err_cc   = nan(numel(fv), 1);
    wall_v   = nan(numel(fv), 1);
    iter_v   = nan(numel(fv), 1);

    for j = 1:numel(fv)
        fpath = fullfile(dat_dir, fv(j).name);
        err_stag(j) = max(load_field(fpath, 'rel_err_rho_stag_t'));
        err_cc(j)   = max(load_field(fpath, 'rel_err_rho_cc_t'));
        wall_v(j)   = load_scalar(fpath, 'bd_wall');
        iter_v(j)   = load_scalar(fpath, 'iters_bd');
    end

    hv = 1 ./ nxv_s;   % grid spacing

    fig = figure('Units','centimeters','Position',[2 2 16 7]);
    tl  = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile;
    loglog(hv, err_stag, '-o', 'LineWidth', LW, 'MarkerSize', MS, ...
        'MarkerFaceColor', [0.13 0.47 0.71], 'DisplayName', 'Stag ($x$-var)');
    hold on;
    loglog(hv, err_cc,   '--^', 'LineWidth', LW, 'MarkerSize', MS, ...
        'MarkerFaceColor', [0.84 0.15 0.16], 'DisplayName', 'CC ($y$-var)');
    add_slope_triangle(hv, err_stag, 1, FS);
    add_slope_triangle(hv, err_stag, 2, FS);
    xlabel('$h = 1/N_x$','FontSize',FS);
    ylabel('$\max_t$ rel. $L^2$ error','FontSize',FS);
    title('(a) Grid refinement','FontSize',FS);
    legend('Location','best','FontSize',FS-1,'Box','off');
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    nexttile;
    loglog(hv, wall_v, '-o', 'LineWidth', LW, 'MarkerSize', MS, ...
        'MarkerFaceColor', [0.2 0.6 0.2]);
    xlabel('$h = 1/N_x$','FontSize',FS);
    ylabel('Wall time (s)','FontSize',FS);
    title('(b) Cost vs grid size','FontSize',FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    sgtitle(sprintf('Grid refinement  ($\\varepsilon=%.4g$)', eps_i), 'FontSize', FS+1, 'Interpreter', 'latex');
    savefig_both(fig, fig_dir, sprintf('refine_banded_eps%g', eps_i));
    close all;
end

fprintf('\nDone.  Figures saved to %s\n', fig_dir);

%% =========================================================================
%  Helper functions
%% =========================================================================

function savefig_both(fig, fig_dir, name)
    print(fig, fullfile(fig_dir, [name '.pdf']), '-dpdf', '-painters', '-bestfit');
    saveas(fig,  fullfile(fig_dir, [name '.png']));
end

function val = load_field(fpath, field)
    d   = load(fpath, field);
    val = d.(field);
end

function val = load_scalar(fpath, field)
    d   = load(fpath, field);
    val = d.(field);
end

function add_slope_triangle(hv, ev, p, FS)
% Draw a reference slope triangle for order p on a loglog plot.
    valid = ~isnan(ev) & ev > 0;
    if sum(valid) < 2, return; end
    hv_v = hv(valid);   ev_v = ev(valid);
    % Place triangle in middle of data range
    i1 = max(1, floor(sum(valid)/2));
    i2 = min(sum(valid), i1+1);
    h1 = hv_v(i1);   h2 = hv_v(i2);
    e1 = ev_v(i1);
    e2_ref = e1 * (h2/h1)^p;
    patch([h1, h2, h2, h1], [e1, e1, e2_ref, e1], [0.8 0.8 0.8], ...
        'FaceAlpha', 0.4, 'EdgeColor', [0.4 0.4 0.4], 'LineWidth', 0.8, ...
        'HandleVisibility','off');
    text(sqrt(h1*h2), e1 * 0.7, sprintf('$O(h^%d)$', p), ...
        'FontSize', FS-2, 'HorizontalAlignment','center', ...
        'Interpreter','latex', 'HandleVisibility','off');
end
