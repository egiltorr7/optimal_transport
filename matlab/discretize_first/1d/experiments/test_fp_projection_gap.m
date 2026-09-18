% TEST_FP_PROJECTION_GAP
%
%   Measures  ||P(x) - x||  where x is the Sinkhorn solution projected onto
%   the ADMM grid and P is the projection onto the discrete FP constraint set
%   (either ETD or CN/banded).
%
%   A large gap ||P(x) - x|| means the Sinkhorn solution lies far from the
%   discrete constraint manifold — the ADMM projection step must do more work
%   to push it back.  This directly reveals which eps regime is "hard" for
%   each discretisation.
%
%   For each epsilon:
%     1. Solve via log-domain Sinkhorn -> (log_phi0, log_psiT).
%     2. Reconstruct the Sinkhorn solution on the ADMM staggered grid:
%          mu_sk   (ntm x nx)  : rho at interior integer times
%          psi_sk  (nt  x nxm) : mx at half-integer times
%     3. Apply P_ETD and P_CN to x_sk = {mu_sk, psi_sk}.
%     4. Report ||P(x)-x|| decomposed into rho and mx components, as
%        functions of t and globally.
%
%   Figures (saved to results/figures/):
%     fp_gap_vs_t_<tag>.pdf    -- ||P(x)-x||_2(t) vs time, ETD and CN, one panel each
%     fp_gap_vs_eps_<tag>.pdf  -- global gap vs eps, ETD vs CN overlaid

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% -------------------------------------------------------------------------
%  Parameters
% -------------------------------------------------------------------------
NT = 128;
NX = 128;

EPS_VALS = [5e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0];
NE       = numel(EPS_VALS);

TOL_SINK = 1e-10;

%% -------------------------------------------------------------------------
%  Base config
% -------------------------------------------------------------------------
cfg_base          = cfg_ladmm_gaussian_expsemi();
cfg_base.nt       = NT;
cfg_base.nx       = NX;
cfg_base.max_iter = 0;

prob_def = prob_gaussian();

%% -------------------------------------------------------------------------
%  Storage
%   gap_rho_t{i}(j) = sqrt(dx * ||P(x).rho(j,:) - mu(j,:)||^2)  at time j
%   gap_mx_t{i}(j)  = sqrt(dx * ||P(x).mx(j,:)  - psi(j,:)||^2)
%   Gap_rho(i), Gap_mx(i) = max over j of the above
% -------------------------------------------------------------------------
gap_rho_t_etd = cell(NE, 1);
gap_mx_t_etd  = cell(NE, 1);
Gap_rho_etd   = nan(NE, 1);
Gap_mx_etd    = nan(NE, 1);

gap_rho_t_cn  = cell(NE, 1);
gap_mx_t_cn   = cell(NE, 1);
Gap_rho_cn    = nan(NE, 1);
Gap_mx_cn     = nan(NE, 1);

fprintf('\n%-8s  %-12s  %-12s  %-12s  %-12s  %-8s\n', ...
    'eps', 'ETD|drho|', 'ETD|dmx|', 'CN|drho|', 'CN|dmx|', 'iters');
fprintf('%s\n', repmat('-', 1, 68));

for i = 1:NE
    eps_i = EPS_VALS(i);

    cfg_i        = cfg_base;
    cfg_i.vareps = eps_i;
    problem_i    = setup_problem(cfg_i, prob_def);

    nt  = problem_i.nt;
    ntm = nt - 1;
    nx  = problem_i.nx;
    nxm = nx - 1;
    dt  = problem_i.dt;
    dx  = problem_i.dx;

    % ---- Precompute projections ----
    problem_i.expsemi_proj = precomp_expsemi_proj(problem_i, eps_i);
    problem_i.banded_proj  = precomp_banded_proj(problem_i, eps_i);

    % ---- Sinkhorn (log-domain) ----
    cfg_sk              = struct();
    cfg_sk.vareps       = eps_i;
    cfg_sk.max_iter     = 10000;
    cfg_sk.tol          = TOL_SINK;
    cfg_sk.precomp_heat = @precomp_heat_neumann_log;
    if eps_i < 0.05
        cfg_sk.eps_init      = min(1.0, eps_i * 500);
        cfg_sk.anneal_factor = 4.0;
    end

    res_sk   = sinkhorn_hopf_cole_logdomain(problem_i, cfg_sk);
    log_phi0 = res_sk.log_phi0;
    log_psiT = res_sk.log_psiT;
    heat     = precomp_heat_neumann_log(problem_i, cfg_sk);

    % ---- Reconstruct Sinkhorn solution on ADMM staggered grid ----
    % mu_sk: rho at interior integer times t_1 .. t_{nt-1}  (ntm x nx)
    mu_sk = zeros(ntm, nx);
    for k = 1:ntm
        phi_k = exp(heat.apply_time(log_phi0, k * dt));   % (nx x 1)
        psi_k = exp(heat.apply_time(log_psiT, 1 - k*dt));
        mu_sk(k, :) = phi_k(:)' .* psi_k(:)';
    end

    % psi_sk: mx at half-integer times t_{1/2} .. t_{nt-1/2}  (nt x nxm)
    psi_sk = zeros(nt, nxm);
    for j = 1:nt
        t_half   = (j - 0.5) * dt;
        phi_h    = exp(heat.apply_time(log_phi0, t_half));
        psi_h    = exp(heat.apply_time(log_psiT, 1 - t_half));
        phi_stag = 0.5 * (phi_h(1:nxm) + phi_h(2:nx));
        dpsi_dx  = (psi_h(2:nx) - psi_h(1:nxm)) / dx;
        psi_sk(j, :) = (2 * eps_i * phi_stag .* dpsi_dx)';
    end

    x_sk.rho = mu_sk;
    x_sk.mx  = psi_sk;

    % ---- Apply ETD projection ----
    x_etd = proj_fokker_planck_expsemi(x_sk, problem_i, cfg_i);

    drho_etd = x_etd.rho - mu_sk;   % (ntm x nx)
    dmx_etd  = x_etd.mx  - psi_sk;  % (nt  x nxm)

    gap_rho_t_etd{i} = sqrt(dx * sum(drho_etd.^2, 2));   % (ntm x 1)
    gap_mx_t_etd{i}  = sqrt(dx * sum(dmx_etd.^2,  2));   % (nt  x 1)
    Gap_rho_etd(i)   = max(gap_rho_t_etd{i});
    Gap_mx_etd(i)    = max(gap_mx_t_etd{i});

    % ---- Apply CN (banded) projection ----
    x_cn = proj_fokker_planck_banded(x_sk, problem_i, cfg_i);

    drho_cn = x_cn.rho - mu_sk;
    dmx_cn  = x_cn.mx  - psi_sk;

    gap_rho_t_cn{i} = sqrt(dx * sum(drho_cn.^2, 2));
    gap_mx_t_cn{i}  = sqrt(dx * sum(dmx_cn.^2,  2));
    Gap_rho_cn(i)   = max(gap_rho_t_cn{i});
    Gap_mx_cn(i)    = max(gap_mx_t_cn{i});

    fprintf('%-8.2e  %-12.3e  %-12.3e  %-12.3e  %-12.3e  %-8d\n', ...
        eps_i, Gap_rho_etd(i), Gap_mx_etd(i), Gap_rho_cn(i), Gap_mx_cn(i), ...
        res_sk.iters);
end

fprintf('%s\n\n', repmat('-', 1, 68));

%% -------------------------------------------------------------------------
%  Figure 1: gap vs time — 2x2: rows = rho/mx, cols = ETD/CN
% -------------------------------------------------------------------------
FS   = 11;   LW = 1.4;
cmap = cool(NE);
step = max(1, floor(NE / 7));
ftag = sprintf('nt%d_nx%d', NT, NX);

% Time axes
t_int  = ((1:NT-1)') * (1/NT);        % interior integer times (ntm x 1)
t_half = ((1:NT)' - 0.5) * (1/NT);   % half-integer times     (nt  x 1)

fig1 = figure('Units', 'centimeters', 'Position', [2 2 28 18]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel (a): ETD gap in rho
nexttile;
hold on;
h_etd = gobjects(NE, 1);
for i = 1:NE
    h_etd(i) = semilogy(t_int, gap_rho_t_etd{i}, '-', ...
        'Color', cmap(i,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$\\varepsilon = %s$', sci2latex(EPS_VALS(i))));
end
xlabel('$t$', 'FontSize', FS);
ylabel('$\Vert\Delta\rho(t)\Vert_{L^2_x}$', 'FontSize', FS);
title('(a) ETD: gap in $\rho$', 'FontSize', FS);
legend(h_etd(1:step:NE), 'Location', 'best', 'FontSize', FS-2, 'Box', 'off');
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

% Panel (b): CN gap in rho
nexttile;
hold on;
h_cn = gobjects(NE, 1);
for i = 1:NE
    h_cn(i) = semilogy(t_int, gap_rho_t_cn{i}, '-', ...
        'Color', cmap(i,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$\\varepsilon = %s$', sci2latex(EPS_VALS(i))));
end
xlabel('$t$', 'FontSize', FS);
ylabel('$\Vert\Delta\rho(t)\Vert_{L^2_x}$', 'FontSize', FS);
title('(b) CN: gap in $\rho$', 'FontSize', FS);
legend(h_cn(1:step:NE), 'Location', 'best', 'FontSize', FS-2, 'Box', 'off');
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

% Panel (c): ETD gap in mx
nexttile;
hold on;
for i = 1:NE
    semilogy(t_half, gap_mx_t_etd{i}, '-', ...
        'Color', cmap(i,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$\\varepsilon = %s$', sci2latex(EPS_VALS(i))));
end
xlabel('$t$', 'FontSize', FS);
ylabel('$\Vert\Delta m_x(t)\Vert_{L^2_x}$', 'FontSize', FS);
title('(c) ETD: gap in $m_x$', 'FontSize', FS);
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

% Panel (d): CN gap in mx
nexttile;
hold on;
for i = 1:NE
    semilogy(t_half, gap_mx_t_cn{i}, '-', ...
        'Color', cmap(i,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$\\varepsilon = %s$', sci2latex(EPS_VALS(i))));
end
xlabel('$t$', 'FontSize', FS);
ylabel('$\Vert\Delta m_x(t)\Vert_{L^2_x}$', 'FontSize', FS);
title('(d) CN: gap in $m_x$', 'FontSize', FS);
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

sgtitle('Projection gap ||P(x) - x||: Sinkhorn solution onto discrete FP set', ...
    'FontSize', FS+1, 'Interpreter', 'none');

exportgraphics(fig1, fullfile(fig_dir, sprintf('fp_gap_vs_t_%s.pdf', ftag)), ...
    'ContentType', 'vector');
saveas(fig1, fullfile(fig_dir, sprintf('fp_gap_vs_t_%s.png', ftag)));

%% -------------------------------------------------------------------------
%  Figure 2: global gap vs eps — ETD vs CN, rho and mx panels
% -------------------------------------------------------------------------
col_etd = [0.13 0.47 0.71];
col_cn  = [0.84 0.15 0.16];
LW2 = LW + 0.2;   MS2 = 6;

fig2 = figure('Units', 'centimeters', 'Position', [2 2 22 8]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel (a): rho gap
nexttile;
loglog(EPS_VALS, Gap_rho_etd, '-o', 'Color', col_etd, 'LineWidth', LW2, ...
    'MarkerSize', MS2, 'MarkerFaceColor', col_etd, 'DisplayName', 'ETD');
hold on;
loglog(EPS_VALS, Gap_rho_cn, '-s', 'Color', col_cn, 'LineWidth', LW2, ...
    'MarkerSize', MS2, 'MarkerFaceColor', col_cn, 'DisplayName', 'CN');
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('$\max_t \Vert\Delta\rho(t)\Vert_{L^2_x}$', 'FontSize', FS);
title('(a) Gap in $\rho$', 'FontSize', FS);
legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

% Panel (b): mx gap
nexttile;
loglog(EPS_VALS, Gap_mx_etd, '-o', 'Color', col_etd, 'LineWidth', LW2, ...
    'MarkerSize', MS2, 'MarkerFaceColor', col_etd, 'DisplayName', 'ETD');
hold on;
loglog(EPS_VALS, Gap_mx_cn, '-s', 'Color', col_cn, 'LineWidth', LW2, ...
    'MarkerSize', MS2, 'MarkerFaceColor', col_cn, 'DisplayName', 'CN');
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('$\max_t \Vert\Delta m_x(t)\Vert_{L^2_x}$', 'FontSize', FS);
title('(b) Gap in $m_x$', 'FontSize', FS);
legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

sgtitle(sprintf('||P(x)-x|| vs eps: ETD vs CN  (NT=%d, Nx=%d)', NT, NX), ...
    'FontSize', FS+1, 'Interpreter', 'none');

exportgraphics(fig2, fullfile(fig_dir, sprintf('fp_gap_vs_eps_%s.pdf', ftag)), ...
    'ContentType', 'vector');
saveas(fig2, fullfile(fig_dir, sprintf('fp_gap_vs_eps_%s.png', ftag)));

fprintf('Figures saved:\n  fp_gap_vs_t_%s.{pdf,png}\n  fp_gap_vs_eps_%s.{pdf,png}\n', ...
    ftag, ftag);

%% -------------------------------------------------------------------------
%  Save results
% -------------------------------------------------------------------------
res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
save(fullfile(res_dir, sprintf('fp_gap_sinkhorn_%s.mat', ftag)), ...
    'EPS_VALS', 'NT', 'NX', 't_int', 't_half', ...
    'gap_rho_t_etd', 'gap_mx_t_etd', 'Gap_rho_etd', 'Gap_mx_etd', ...
    'gap_rho_t_cn',  'gap_mx_t_cn',  'Gap_rho_cn',  'Gap_mx_cn');

%% -------------------------------------------------------------------------
%  Helper
% -------------------------------------------------------------------------
function s = sci2latex(x)
    e = floor(log10(x));
    m = x / 10^e;
    if abs(m - round(m)) < 0.05
        if round(m) == 1
            s = sprintf('10^{%d}', e);
        else
            s = sprintf('%d \\times 10^{%d}', round(m), e);
        end
    else
        s = sprintf('%.1f \\times 10^{%d}', m, e);
    end
end
