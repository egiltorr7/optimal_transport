% TEST_PROJECTION_VS_EPS  Diagnose FP projection quality as epsilon grows.
%
% Uses the Sinkhorn-Neumann solution as x_in (instead of the analytical
% infinite-domain solution) so that boundary effects at large eps do not
% contaminate the results.  Both ADMM and Sinkhorn-Neumann solve the same
% reflected-BM problem, so the Sinkhorn solution is the correct reference.
%
% For each epsilon in a sweep this script measures:
%
%   1. FP residual of the Sinkhorn solution BEFORE projection
%      Does the Sinkhorn solution satisfy the *discrete* FP equation?
%      If this grows with eps, the two discretizations (Sinkhorn vs ADMM)
%      differ at large eps.
%
%   2. How far projection moves the input: ||x_out - x_in|| / ||x_in||
%      Large = the Sinkhorn and ADMM solutions live on different FP manifolds.
%
%   3. FP residual AFTER projection (sanity: should be ~machine eps always).
%
%   4. Spectral content of rho before vs after projection.
%      red > blue at high k -> projection amplifies oscillations (bad).
%      red < blue at high k -> projection damps them (anti-oscillatory).
%
%   5. Condition numbers of the tridiagonal T_k systems for k=2..nx.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config ---
cfg       = cfg_ladmm_gaussian();
cfg.nt    = 256;
cfg.nx    = 256;

% Sinkhorn config (Neumann BCs, same physics as ADMM)
cfg_sink.max_iter     = 2000;
cfg_sink.tol          = 1e-10;
cfg_sink.precomp_heat = @precomp_heat_neumann;

prob_def  = prob_gaussian();
eps_vals  = [0.01, 0.1, 0.5, 1, 2, 4, 8, 100, 1000];
n_eps     = numel(eps_vals);

%% --- Allocate result arrays ---
fp_res_before = zeros(n_eps, 1);
fp_res_after  = zeros(n_eps, 1);
proj_dist     = zeros(n_eps, 1);
cond_max      = zeros(n_eps, 1);
nx_           = cfg.nx;
cond_at_k     = zeros(n_eps, nx_);

%% --- Main sweep ---
fprintf('%8s  %13s  %13s  %13s  %13s\n', ...
    'eps', 'FP_res_before', 'FP_res_after', 'proj_dist', 'max_cond(Tk_sc)');

for i = 1:n_eps
    vareps     = eps_vals(i);
    cfg.vareps = vareps;

    problem = setup_problem(cfg, prob_def);
    bp      = precomp_banded_proj(problem, vareps);
    problem.banded_proj = bp;

    % Sinkhorn-Neumann solution (same BCs as ADMM)
    cfg_sink.vareps = vareps;
    res_sink = sinkhorn_hopf_cole(problem, cfg_sink);

    % Resample from Sinkhorn edge-time grid (k=0..nt) to ADMM staggered grid:
    %   rho: interior times k*dt, k=1..nt-1  -> rows 2..nt of res_sink.rho
    %   mx:  half-times (k-0.5)*dt, k=1..nt  -> average adjacent edge rows
    nt = problem.nt;
    x_in.rho = res_sink.rho(2:nt,   :);                                   % (nt-1 x nx)
    x_in.mx  = 0.5*(res_sink.mx(1:nt,:) + res_sink.mx(2:nt+1,:));         % (nt x nxm)

    % FP residual before
    fp_res_before(i) = compute_fp_res(x_in, problem, vareps);

    % Project
    x_out = proj_fokker_planck_banded(x_in, problem, cfg);

    % FP residual after
    fp_res_after(i) = compute_fp_res(x_out, problem, vareps);

    % Relative distance projection moved the input
    dt = problem.dt;   dx = problem.dx;
    drho = x_out.rho - x_in.rho;
    dmx  = x_out.mx  - x_in.mx;
    norm_in = sqrt(dt*dx*(sum(x_in.rho(:).^2) + sum(x_in.mx(:).^2)));
    norm_d  = sqrt(dt*dx*(sum(drho(:).^2)     + sum(dmx(:).^2)));
    proj_dist(i) = norm_d / norm_in;

    % Condition numbers of T_k for k = 2..nx
    cond_at_k(i, :) = compute_cond_Tk(problem, vareps);
    cond_max(i)      = max(cond_at_k(i, 2:end));

    fprintf('%8.4g  %13.3e  %13.3e  %13.3e  %13.3e\n', ...
        vareps, fp_res_before(i), fp_res_after(i), proj_dist(i), cond_max(i));
end

%% --- Figure 1: FP residuals and projection distance ---
figure('Name', 'Projection quality vs eps', 'Position', [50 50 950 320]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
loglog(eps_vals, fp_res_before, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('eps');
ylabel('L2 norm');
title('FP residual of analytical sol (before)');
grid on;

nexttile;
semilogy(eps_vals, proj_dist, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('eps');
ylabel('||x_out - x_in|| / ||x_in||');
title('Relative distance projection moves input');
grid on;

nexttile;
semilogy(eps_vals, fp_res_after, 'k-^', 'LineWidth', 1.5, 'MarkerSize', 6);
yline(1e-12, 'r--', 'machine eps', 'LabelHorizontalAlignment', 'left');
xlabel('eps');
ylabel('L2 norm');
title('FP residual after projection (sanity)');
grid on;

sgtitle(sprintf('FP projection diagnostics  (nt=%d, nx=%d)', cfg.nt, cfg.nx));
saveas(gcf, fullfile(fig_dir, sprintf('proj_quality_nt%d_nx%d.png', cfg.nt, cfg.nx)));

%% --- Figure 2: Condition numbers of T_k ---
figure('Name', 'Condition numbers of T_k', 'Position', [50 420 700 320]);
colors = parula(n_eps);
hold on;
k_vals = 2:nx_;
for i = 1:n_eps
    semilogy(k_vals, cond_at_k(i, k_vals), '-', 'Color', colors(i,:), 'LineWidth', 1.2, ...
        'DisplayName', sprintf('eps=%.4g', eps_vals(i)));
end
legend('Location', 'best', 'FontSize', 8);
xlabel('Spatial mode k');
ylabel('cond(Tk)');
title(sprintf('Condition numbers of tridiagonal Tk  (nt=%d, nx=%d)', cfg.nt, cfg.nx));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('proj_condnums_nt%d_nx%d.png', cfg.nt, cfg.nx)));

%% --- Figure 3: Spectral power before vs after projection ---
figure('Name', 'Spectral content', 'Position', [800 50 700 520]);
nc = 2;   nr = ceil(n_eps / nc);
tiledlayout(nr, nc, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:n_eps
    vareps     = eps_vals(i);
    cfg.vareps = vareps;
    problem    = setup_problem(cfg, prob_def);
    bp         = precomp_banded_proj(problem, vareps);
    problem.banded_proj = bp;

    cfg_sink.vareps = vareps;
    res_sink_sp     = sinkhorn_hopf_cole(problem, cfg_sink);
    nt_sp           = problem.nt;
    x_in.rho = res_sink_sp.rho(2:nt_sp,   :);
    x_in.mx  = 0.5*(res_sink_sp.mx(1:nt_sp,:) + res_sink_sp.mx(2:nt_sp+1,:));
    x_out    = proj_fokker_planck_banded(x_in, problem, cfg);

    % Mean DCT power spectrum over time
    pow_in  = mean(dct(x_in.rho').^2, 2);    % (nx x 1)
    pow_out = mean(dct(x_out.rho').^2, 2);

    nexttile;
    semilogy(1:nx_, pow_in,  'b-',  'LineWidth', 1.2, 'DisplayName', 'before proj');
    hold on;
    semilogy(1:nx_, pow_out, 'r--', 'LineWidth', 1.2, 'DisplayName', 'after proj');
    xlabel('DCT mode k');
    ylabel('Power');
    title(sprintf('eps = %.4g', vareps));
    legend('Location', 'northeast', 'FontSize', 7);
    grid on;
end
sgtitle('Spectral power of rho: before vs after projection (red>blue at high k = amplification)');
saveas(gcf, fullfile(fig_dir, sprintf('proj_spectral_nt%d_nx%d.png', cfg.nt, cfg.nx)));

fprintf('\nFigures saved to %s\n', fig_dir);
fprintf('\nKey: what to look for\n');
fprintf('  FP res before  : grows with eps  -> analytical sol inconsistent with discrete FP\n');
fprintf('  proj_dist      : grows with eps  -> projection working harder\n');
fprintf('  FP res after   : ~machine eps    -> linear solve accurate (sanity)\n');
fprintf('  cond(Tk)       : blows up at k   -> ill-conditioned mode\n');
fprintf('  spectral red>blue at high k      -> projection introduces oscillations\n');
fprintf('  spectral red<blue at high k      -> projection is smoothing (anti-oscillatory)\n');

%% =========================================================
%  Local functions
%% =========================================================

function res = compute_fp_res(x, problem, vareps)
    ops     = problem.ops;
    rho0    = problem.rho0;
    rho1    = problem.rho1;
    nt      = problem.nt;
    dt      = problem.dt;
    dx      = problem.dx;
    zeros_x = zeros(nt, 1);

    mu  = x.rho;
    psi = x.mx;

    lap_mu = ops.deriv_x_at_phi( ...
                 ops.deriv_x_at_m( ...
                     ops.interp_t_at_phi(mu, rho0, rho1)), ...
                 zeros_x, zeros_x);

    f = ops.deriv_t_at_phi(mu, rho0, rho1) ...
      + ops.deriv_x_at_phi(psi, zeros_x, zeros_x) ...
      - vareps * lap_mu;

    res = sqrt(dt * dx) * norm(f(:));
end

function conds = compute_cond_Tk(problem, vareps)
    nt  = problem.nt;    ntm = nt - 1;
    nx  = problem.nx;
    dt  = problem.dt;
    lx  = problem.lambda_x;

    It_phi = 0.5 * toeplitz([1 1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]);
    Dt_phi =       toeplitz([1 -1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]) / dt;
    It_rho = 0.5 * toeplitz([1 zeros(1,ntm-1)], [1 1 zeros(1,ntm-1)]);
    Dt_rho =       toeplitz([-1 zeros(1,ntm-1)], [-1 1 zeros(1,ntm-1)]) / dt;

    M0  = -Dt_phi * Dt_rho;
    M1d = ones(nt, 1);
    M1d(1)  = 1 + vareps / dt;
    M1d(nt) = 1 - vareps / dt;
    M2  = vareps^2 * (It_phi * It_rho);

    conds = zeros(1, nx);
    for k = 2:nx
        Tk  = M0 + diag(lx(k) * M1d) + lx(k)^2 * M2;
        dk  = sqrt(abs(diag(Tk)));
        Tks = diag(1./dk) * Tk * diag(1./dk);   % scaled system
        conds(k) = condest(sparse(Tks));
    end
end
