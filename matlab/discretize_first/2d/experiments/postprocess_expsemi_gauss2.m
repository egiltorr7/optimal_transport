% POSTPROCESS_EXPSEMI_GAUSS2  Post-process data from test_expsemi_gauss2.m.
%
%   Gaussian-to-Gaussian SB with DIFFERENT means AND different standard
%   deviations at the two endpoints (sigma0 != sigma1). Mirrors
%   postprocess_expsemi.m exactly; see that file for the full figure list.
%
%   Run this locally after copying results/data/ from the server.
%   Reads all expsemi2d_gauss2_*.mat files found in dat_dir and produces:
%
%   Per-run figures  (one per (NT,NX,eps)):
%     admm_conv_*      -- res_x / res_y / res_primal vs iteration
%     err_vs_t_*       -- relative L2 error vs time (stag + cc)
%     density_*        -- ExpSemi rho at 6 times, t=0 (rho0) to t=1 (rho1)
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
dat_dir  = fullfile(base_dir, '..', 'results', 'data', 'gauss2');
fig_dir  = fullfile(base_dir, '..', 'results', 'figures','paper', 'gauss2');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

FS = 11;   LW = 1.5;   MS = 6;

%% -------------------------------------------------------------------------
%  Discover available files
% -------------------------------------------------------------------------
files = dir(fullfile(dat_dir, 'expsemi2d_gauss2_eps*_nt*_nx*.mat'));
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
        'expsemi2d_gauss2_eps([\d.e+\-]+)_nt(\d+)_nx(\d+)', 'tokens');
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
FILTER_NT  = [];    % e.g. 64
FILTER_NX  = [];    % e.g. 32
FILTER_EPS = [];    % e.g. 1.0

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
    iters_vec = (1:d.iters_es)';

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
    savefig_both(fig, fig_dir, ['admm_conv_gauss2_' tag]);

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
    savefig_both(fig, fig_dir, ['err_vs_t_gauss2_' tag]);

    %% --- (3) Density snapshots: ExpSemi only, t=0 (rho0) to t=1 (rho1) ---
    rho0_2d = normal2d_density(d.xx, d.yy, d.MU0(1), d.MU0(2), d.SIGMA0);
    rho0_2d = rho0_2d / (sum(rho0_2d(:)) * d.dx * d.dy);
    rho1_2d = normal2d_density(d.xx, d.yy, d.MU1(1), d.MU1(2), d.SIGMA1);
    rho1_2d = rho1_2d / (sum(rho1_2d(:)) * d.dx * d.dy);

    % Build cell indices via exact integer mirroring (k_hi = nt+1-k_lo) rather
    % than independently rounding both t_frac*nt and (1-t_frac)*nt -- the two
    % roundings are NOT guaranteed to be mirror images of each other, which
    % previously caused a spurious few-cell time asymmetry near the boundaries.
    f_edge     = edge_time_frac(d.eps_i, d.nt);
    k_edge_lo  = max(1, round(f_edge * d.nt));
    k_edge_hi  = d.nt + 1 - k_edge_lo;
    k_third_lo = max(1, round((1/3) * d.nt));
    k_third_hi = d.nt + 1 - k_third_lo;
    idxs       = [NaN, k_edge_lo, k_third_lo, k_third_hi, k_edge_hi, NaN];
    n_snap     = numel(idxs);

    imgs = cell(n_snap, 1);
    t_ks = zeros(n_snap, 1);
    for s = 1:n_snap
        if s == 1
            imgs{s} = rho0_2d;   t_ks(s) = 0;
        elseif s == n_snap
            imgs{s} = rho1_2d;   t_ks(s) = 1;
        else
            k_idx   = idxs(s);
            imgs{s} = squeeze(d.rho_es_cc(k_idx,:,:));
            t_ks(s) = (k_idx - 0.5) * d.dt;
        end
    end
    all_vals = cell2mat(cellfun(@(im) im(:), imgs, 'UniformOutput', false));
    clim_lo  = min(all_vals);
    clim_hi  = max(all_vals);

    fig = figure('Units','centimeters','Position',[2 2 22 10]);
    tl  = tiledlayout(2, 3, 'TileSpacing','compact','Padding','compact');

    for s = 1:n_snap
        ax = nexttile(tl, s);
        imagesc(d.xx, d.yy, imgs{s}');
        axis xy; colorbar; clim([clim_lo, clim_hi]);
        title(sprintf('ExpSemi,  %s', fmt_time(t_ks(s))), 'FontSize', FS);
        if s > 3, xlabel('$x$','FontSize',FS); end
        if mod(s-1,3)==0, ylabel('$y$','FontSize',FS); end
        set(ax,'FontSize',FS-1,'TickDir','out');
    end
    sgtitle(sprintf('Density ($\\varepsilon=%.4g$,  $N_T=%d$,  $N_x=%d$)', ...
        d.eps_i, d.NT, d.NX), 'FontSize', FS+1, 'Interpreter', 'latex');
    savefig_both(fig, fig_dir, ['density_gauss2_' tag]);

    %% --- (4) Iter time histogram ----------------------------------------
    it = d.iter_times(1:d.iters_es);
    fig = figure('Units','centimeters','Position',[2 2 12 6]);
    histogram(it * 1e3, 40, 'FaceColor',[0.13 0.47 0.71], 'EdgeColor','none');
    xlabel('Time per iteration (ms)', 'FontSize', FS);
    ylabel('Count', 'FontSize', FS);
    title(sprintf('Iter time  ($N_T=%d$,  $N_x=%d$,  $\\varepsilon=%.4g$,  median=%.1fms)', ...
        d.NT, d.NX, d.eps_i, median(it)*1e3), 'FontSize', FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;
    savefig_both(fig, fig_dir, ['iter_time_gauss2_' tag]);

    %% --- (5) Density along diagonal x=y ------------------------------------
    nx_d = d.nx;   ny_d = d.ny;   nt_d = d.nt;
    nd   = min(nx_d, ny_d);
    x_diag = ((1:nd) - 0.5) * d.dx;   % spatial coord along x=y

    rho_es_diag  = zeros(nt_d, nd);
    rho_ref_diag = zeros(nt_d, nd);
    for ii = 1:nd
        rho_es_diag(:,  ii) = d.rho_es_cc(:,  ii, ii);
        rho_ref_diag(:, ii) = d.rho_ref_cc(:, ii, ii);
    end
    rho0_diag = diag(rho0_2d)';   % (1 x nd), rho0_2d computed in section (3)
    rho1_diag = diag(rho1_2d)';

    % Build cell indices via exact integer mirroring (k_hi = nt+1-k_lo) rather
    % than independently rounding both t_frac*nt and (1-t_frac)*nt -- see note
    % in section (3) above.
    f_edge_d = edge_time_frac(d.eps_i, nt_d);
    k_edge_lo_d = max(1, round(f_edge_d * nt_d));
    k_edge_hi_d = nt_d + 1 - k_edge_lo_d;
    k_q_lo_d    = max(1, round(0.25 * nt_d));
    k_q_hi_d    = nt_d + 1 - k_q_lo_d;
    k_mid_d     = max(1, round(0.5 * nt_d));
    idxs_d      = [NaN, k_edge_lo_d, k_q_lo_d, k_mid_d, k_q_hi_d, k_edge_hi_d, NaN];
    ns_d        = numel(idxs_d);
    t_cc_d      = ((1:nt_d)' - 0.5) * d.dt;
    clrs        = cool(ns_d);

    % Subsample markers so the ADMM solution reads as discrete points rather
    % than a solid dotted line at high spatial resolution.
    n_markers = min(nd, 25);
    mk_idx    = round(linspace(1, nd, n_markers));

    fig = figure('Units','centimeters','Position',[2 2 18 8]);
    hold on;
    for s = 1:ns_d
        if s == 1
            % Exact rho0 -- no ADMM iterate exists at t=0, line only.
            plot(x_diag, rho0_diag, '-', 'Color', clrs(s,:), ...
                'LineWidth', LW, 'DisplayName', fmt_time(0));
        elseif s == ns_d
            % Exact rho1 -- no ADMM iterate exists at t=1, line only.
            plot(x_diag, rho1_diag, '-', 'Color', clrs(s,:), ...
                'LineWidth', LW, 'DisplayName', fmt_time(1));
        else
            k_idx = idxs_d(s);
            t_k   = t_cc_d(k_idx);
            plot(x_diag, rho_ref_diag(k_idx, :), '-', 'Color', clrs(s,:), ...
                'LineWidth', LW, 'HandleVisibility', 'off');
            plot(x_diag(mk_idx), rho_es_diag(k_idx, mk_idx),  'o', 'Color', clrs(s,:), ...
                'MarkerSize', MS, 'MarkerFaceColor', clrs(s,:), 'LineStyle', 'none', ...
                'DisplayName', fmt_time(t_k));
        end
    end
    plot(nan, nan, 'ko', 'MarkerSize', MS, 'MarkerFaceColor', 'k', 'DisplayName', 'ExpSemi');
    plot(nan, nan, 'k-', 'LineWidth', LW, 'DisplayName', sprintf('Ref (%s)', ref_label));
    xlabel('$x$  (along $x=y$)', 'FontSize', FS);
    ylabel('$\rho$', 'FontSize', FS);
    title(sprintf('Slice $x=y$  ($\\varepsilon=%.4g$,  $N_T=%d$,  $N_x=%d$)', ...
        d.eps_i, d.NT, d.NX), 'FontSize', FS, 'Interpreter', 'latex');
    legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
    set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');  grid on;
    savefig_both(fig, fig_dir, ['diag_slice_gauss2_' tag]);

    close all;
end

%% =========================================================================
%  Per-grid sweep figures  (load sweep summary files)
%% =========================================================================
sfiles = dir(fullfile(dat_dir, 'expsemi2d_sweep_gauss2_nt*_nx*.mat'));

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
    loglog(s.sw_eps, s.sw_wall_es,  '-o',  'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', [0.13 0.47 0.71], ...
        'DisplayName', 'ExpSemi');
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
    semilogx(s.sw_eps, s.sw_iters_es, '-o', 'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', [0.2 0.6 0.2]);
    xlabel('$\varepsilon$','FontSize',FS);
    ylabel('ADMM iterations','FontSize',FS);
    title('(c) Iterations vs $\varepsilon$','FontSize',FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    sgtitle(sprintf('$N_T=%d$,  $N_x=N_y=%d$', s.NT, s.NX), 'FontSize', FS+1, 'Interpreter', 'latex');
    savefig_both(fig, fig_dir, ['sweep_gauss2_' tag]);
    close all;
end

%% =========================================================================
%  Cross-grid refinement figures  (fixed eps, vary NX)
%% =========================================================================
for ei = 1:numel(EPS_VALS)
    eps_i = EPS_VALS(ei);
    mask  = abs(f_eps - eps_i) < 1e-12;
    % Restrict to the consistent lockstep-refinement family (fixed nt = 2*nx
    % ratio); grids off that ratio (e.g. nt=nx) sit on a different
    % refinement path and would corrupt the observed convergence-rate slope.
    mask  = mask & (f_NT == 2*f_NX);
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
        err_stag(j) = max(load_field(fpath, 'err_rho_stag_t'));
        err_cc(j)   = max(load_field(fpath, 'err_rho_cc_t'));
        wall_v(j)   = load_scalar(fpath, 'es_wall');
        iter_v(j)   = load_scalar(fpath, 'iters_es');
    end

    Nv = nxv_s;   % N = Nx = Ny, ascending (Nt = 2N held fixed across this family)

    fig = figure('Units','centimeters','Position',[2 2 16 7]);
    tl  = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile;
    loglog(Nv, err_stag, '-o', 'LineWidth', LW, 'MarkerSize', MS, ...
        'MarkerFaceColor', [0.13 0.47 0.71], 'DisplayName', 'Stag ($x$-var)');
    hold on;
    loglog(Nv, err_cc,   '--^', 'LineWidth', LW, 'MarkerSize', MS, ...
        'MarkerFaceColor', [0.84 0.15 0.16], 'DisplayName', 'CC ($y$-var)');
    add_slope_triangle(Nv, err_stag, 1);
    add_slope_triangle(Nv, err_stag, 2);
    xlabel('$N \equiv N_x=N_y$  ($N_t=2N$)','FontSize',FS);
    ylabel('$\max_t$ $L^2$ error','FontSize',FS);
    title('(a) Grid refinement','FontSize',FS);
    legend('Location','best','FontSize',FS-1,'Box','off');
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    nexttile;
    loglog(Nv, wall_v, '-o', 'LineWidth', LW, 'MarkerSize', MS, ...
        'MarkerFaceColor', [0.2 0.6 0.2]);
    xlabel('$N \equiv N_x=N_y$  ($N_t=2N$)','FontSize',FS);
    ylabel('Wall time (s)','FontSize',FS);
    title('(b) Cost vs grid size','FontSize',FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    sgtitle(sprintf('Grid refinement  ($\\varepsilon=%.4g$)', eps_i), 'FontSize', FS+1, 'Interpreter', 'latex');
    savefig_both(fig, fig_dir, sprintf('refine_gauss2_eps%g', eps_i));
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

function add_slope_triangle(Nv, ev, p)
% Draw a dashed O(N^-p) reference line spanning the data range on a
% loglog(N, error) plot (error DECREASES as N increases, opposite convention
% from plotting vs h=1/N). Anchored through the middle data point; labeled
% via the legend rather than an on-plot text annotation.
    valid = ~isnan(ev) & ev > 0;
    if sum(valid) < 2, return; end
    Nv_v = Nv(valid);   ev_v = ev(valid);
    i1 = max(1, floor(sum(valid)/2));
    N1 = Nv_v(i1);   e1 = ev_v(i1);

    N_range = [min(Nv_v), max(Nv_v)];
    e_range = e1 * (N1 ./ N_range).^p;
    plot(N_range, e_range, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1, ...
        'DisplayName', sprintf('$O(N^{-%d})$', p));
end

function frac = edge_time_frac(eps_i, nt)
% As eps grows, push the first/last sampled snapshot time closer to the
% temporal boundary -- interesting dynamics concentrate near t=0,1 for large
% eps since diffusion homogenizes the bulk quickly. Interpolates linearly in
% log10(eps) between 0.1 (eps<=0.01) and the first interior grid point 1/nt
% (eps>=10, "just off the boundary").
    log_lo = log10(0.01);
    log_hi = log10(10);
    t = min(max((log10(eps_i) - log_lo) / (log_hi - log_lo), 0), 1);
    frac = 0.1 + t * (1/nt - 0.1);
end

function rho = normal2d_density(xx, yy, mux, muy, sig)
% Unnormalised isotropic 2D Gaussian density (same formula as test_expsemi.m's
% Normal2d), used to reconstruct the exact rho0/rho1 boundary densities from
% the saved MU0/MU1/SIGMA/xx/yy/dx/dy parameters (normalise after calling).
    rho = exp(-((xx - mux).^2 + (yy - muy).^2) / (2*sig^2)) / (2*pi*sig^2);
end

function s = fmt_time(t_k)
% Format a snapshot time for figure titles/legends. Uses scientific notation
% near the boundaries (t~0 or t~1) since such values can round to "0.00"/
% "1.00" under %.2f even when not exactly at the boundary -- misleading when
% an EXACT boundary panel (t=0 or t=1, using rho0/rho1) is shown alongside.
    if t_k == 0
        s = '$t=0$';
    elseif t_k == 1
        s = '$t=1$';
    elseif t_k < 0.01
        s = sprintf('$t=%s$', sci_str(t_k));
    elseif (1 - t_k) < 0.01
        s = sprintf('$t=1-%s$', sci_str(1 - t_k));
    else
        s = sprintf('$t=%.2f$', t_k);
    end
end

function s = sci_str(x)
% LaTeX-formatted scientific notation, e.g. 3.9\times10^{-3}
    e = floor(log10(x));
    m = x / 10^e;
    s = sprintf('%.1f\\times10^{%d}', m, e);
end
