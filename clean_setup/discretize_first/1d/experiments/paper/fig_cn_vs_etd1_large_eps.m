% FIG_CN_VS_ETD1_LARGE_EPS
%   Side-by-side density evolution at large eps: CN (left) vs ETD1 (right).
%
%   At large eps the CN Fokker-Planck projection uses the Padé(1,1) factor
%   c_k = (1 - alpha_k/2)/(1 + alpha_k/2) where alpha_k = eps*lambda_x(k)*dt.
%   For eps=4, NT=NX=128 the diffusion number r = eps*dt/dx^2 ~ 512, so
%   modes k >= 5 all have c_k ≈ -1: they FLIP SIGN at every time step,
%   producing temporal oscillations.  ETD1 uses c_k = exp(-alpha_k) > 0,
%   which correctly damps those modes and gives a smooth solution.
%
%   Time slices use CONSECUTIVE staggered indices near t=0 and t=1 so the
%   odd/even alternation in the CN profiles is visually apparent.
%
%   Reference: Sinkhorn (Neumann BCs).
%   Plot: probability density rho/dx.  Solid thin = Sinkhorn, dashed thick = solver.
%
%   Output (results/paper/):
%     sb_cn_vs_etd1_eps<VAREPS>_nt<NT>_nx<NX>.pdf

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'setup_paths.m'));

set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultAxesTickLabelInterpreter',     'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

%% -----------------------------------------------------------------------
%% Parameters
%% -----------------------------------------------------------------------
VAREPS   = 4;
NT       = 256;
NX       = 256;
GAMMA    = 100;
TAU      = 101;
MAX_ITER = 10000;
TOL      = 1e-10;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results', 'paper');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -----------------------------------------------------------------------
%% Problem setup (single shared grid)
%% -----------------------------------------------------------------------
prob_def = prob_gaussian();

cfg_cn          = cfg_ladmm_gaussian();
cfg_cn.vareps   = VAREPS;
cfg_cn.nt       = NT;     cfg_cn.nx  = NX;
cfg_cn.gamma    = GAMMA;  cfg_cn.tau = TAU;
cfg_cn.max_iter = MAX_ITER;
cfg_cn.tol      = TOL;

cfg_etd          = cfg_ladmm_gaussian_expsemi();
cfg_etd.vareps   = VAREPS;
cfg_etd.nt       = NT;     cfg_etd.nx  = NX;
cfg_etd.gamma    = GAMMA;  cfg_etd.tau = TAU;
cfg_etd.max_iter = MAX_ITER;
cfg_etd.tol      = TOL;

problem = setup_problem(cfg_cn, prob_def);

nt  = problem.nt;   ntm = nt - 1;
dt  = problem.dt;   dx  = problem.dx;
xx  = problem.xx;

fprintf('Grid: NT=%d  NX=%d  dt=%.4g  dx=%.4g\n', NT, NX, dt, dx);

%% -----------------------------------------------------------------------
%% Sinkhorn reference
%% -----------------------------------------------------------------------
cfg_sk.vareps       = VAREPS;
cfg_sk.max_iter     = 500;
cfg_sk.tol          = 1e-10;
cfg_sk.precomp_heat = @precomp_heat_neumann;

fprintf('\nRunning Sinkhorn  (eps=%g)...\n', VAREPS);
res_sk = sinkhorn_hopf_cole(problem, cfg_sk);
rho_sk = res_sk.rho(2:nt, :);   % (ntm x nx)
fprintf('  iters=%d  wall=%.1fs\n', res_sk.iters, res_sk.walltime);

%% -----------------------------------------------------------------------
%% CN and ETD1 solves
%% -----------------------------------------------------------------------
fprintf('Running CN-ADMM   (eps=%g)...\n', VAREPS);
res_cn = discretize_then_optimize(cfg_cn, problem);
fprintf('  iters=%d  conv=%d  wall=%.1fs\n', res_cn.iters, res_cn.converged, res_cn.walltime);

fprintf('Running ETD1-ADMM (eps=%g)...\n', VAREPS);
res_etd = discretize_then_optimize(cfg_etd, problem);
fprintf('  iters=%d  conv=%d  wall=%.1fs\n', res_etd.iters, res_etd.converged, res_etd.walltime);

rho_cn  = res_cn.rho_stag;    % (ntm x nx)
rho_etd = res_etd.rho_stag;   % (ntm x nx)

%% -----------------------------------------------------------------------
%% Time slices: consecutive indices near t=0 and t=1 + one in the middle.
%% Choosing both odd and even k exposes the alternating (odd=wiggly,
%% even=less-wiggly) pattern in the CN solution.
%% -----------------------------------------------------------------------
t_rho  = (1:ntm)' * dt;

k_show = [1, 2, 3, 4, 5, ...             % k=1..5  near t=0
          round(ntm/2), ...              % t ≈ 0.5
          ntm-4, ntm-3, ntm-2, ntm-1, ntm];  % k=ntm-4..ntm near t=1
ns     = numel(k_show);

%% -----------------------------------------------------------------------
%% Y-axis: base on Sinkhorn peak; cap CN excursions at 2x that peak.
%% Allow negative values (CN oscillations can go below zero).
%% -----------------------------------------------------------------------
pk_ref = max(rho_sk(:)) / dx;
ymax   = pk_ref * 2.0;
ymin   = min(0, min(rho_cn(:)) / dx);
ylims  = [ymin * 1.1 - 0.05, ymax];

%% -----------------------------------------------------------------------
%% Figure
%% -----------------------------------------------------------------------
FS     = 11;   LW_R = 1.2;   LW_M = 0.9;   MS = 3;
FW     = 38.0; FH   = 11.5;
cmap   = lines(ns);
stride = max(1, floor(NX / 40));

methods      = {'CN',  'ETD1'};
rho_nums     = {rho_cn, rho_etd};
marker_syms  = {'o',   '^'};

fig = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
             'PaperUnits','centimeters','PaperSize',[FW, FH]);
tl  = tiledlayout(1, 2, 'TileSpacing','compact', 'Padding','loose');

for panel = 1:2
    ax = nexttile;
    hold on;

    msym = marker_syms{panel};
    idx  = 1:stride:numel(xx);
    for j = 1:ns
        k   = k_show(j);
        col = cmap(j, :);
        % Sinkhorn reference (thin solid)
        plot(xx, rho_sk(k,:) / dx, '-', 'Color', col, 'LineWidth', LW_R);
        % Solver (open markers, stride-sampled)
        plot(xx(idx), rho_nums{panel}(k, idx) / dx, msym, ...
            'Color', col, 'MarkerSize', MS, ...
            'MarkerFaceColor', 'none', 'LineWidth', LW_M);
    end

    % Method proxy legend
    h_ref = plot(nan, nan, 'k-',  'LineWidth', LW_R, 'DisplayName', 'Sinkhorn');
    h_num = plot(nan, nan, ['k' msym], 'MarkerSize', MS, ...
        'MarkerFaceColor', 'none', 'LineWidth', LW_M, 'DisplayName', methods{panel});

    % Time proxy legend (compact: show only first 3, middle, last 3)
    idx_leg = [1, 2, 3, round(ns/2), ns-2, ns-1, ns];
    h_t = gobjects(numel(idx_leg), 1);
    for ii = 1:numel(idx_leg)
        j        = idx_leg(ii);
        k        = k_show(j);
        h_t(ii)  = plot(nan, nan, '-', 'Color', cmap(j,:), 'LineWidth', LW_R, ...
            'DisplayName', sprintf('$t=%.3f$', t_rho(k)));
    end

    legend([h_ref; h_num; h_t], 'Location', 'northeast', 'FontSize', FS-2, 'Box', 'off');

    xlim([0, 1]);
    ylim(ylims);
    yline(0, 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    xlabel('$x$',                 'FontSize', FS);
    ylabel('$\tilde{\rho}(t,x)$', 'FontSize', FS);
    title(sprintf('%s  ($\\varepsilon=%g$)', methods{panel}, VAREPS), 'FontSize', FS);
    set(ax, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
    grid on;
end

sgtitle(sprintf('Density evolution: CN vs ETD1  ($\\varepsilon=%g$,  $N_T=N_x=%d$)', ...
    VAREPS, NT), 'FontSize', FS, 'Interpreter', 'latex');

fname = sprintf('sb_cn_vs_etd1_eps%g_nt%d_nx%d.pdf', VAREPS, NT, NX);
save_fig(fig, fullfile(out_dir, fname));

%% -----------------------------------------------------------------------
%% Local helpers
%% -----------------------------------------------------------------------
function save_fig(fig, fpath)
    try
        exportgraphics(fig, fpath, 'ContentType', 'vector', 'BackgroundColor', 'none');
    catch
        print(fig, fpath, '-dpdf', '-painters', '-r0');
    end
    fprintf('Saved: %s\n', fpath);
end
