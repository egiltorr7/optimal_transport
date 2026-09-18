% TEST_SINKHORN_WARMSTART_ADMM
%
%   Warm-starts ExpSemi-ADMM from the converged Sinkhorn solution on the
%   Gaussian-to-Gaussian SB problem and compares convergence against the
%   default cold start (linear interpolation).
%
%   The warm start evaluates rho and m at each staggered grid point exactly
%   via the Sinkhorn semigroup (phi_0, psi_T) -- no spatial/temporal
%   interpolation is needed.
%
%   Warm-start grids:
%     x0.rho  (ntm x nx)   rho at t = k*dt,         k = 1..ntm
%     x0.mx   (nt  x nxm)  m   at t = (k-0.5)*dt,   k = 1..nt
%     y0.rho  (nt  x nx)   rho at t = (k-0.5)*dt,   k = 1..nt
%     y0.mx   (nt  x nx)   zeros (cell-centre m; updated on first ADMM step)
%
%   Figures (saved to results/figures/):
%     warmstart_residual_<tag>.png   -- ADMM residual: cold vs warm
%     warmstart_density_<tag>.png    -- final density: cold vs warm vs Sinkhorn

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% --- Config ---
VAREPS = 100.0;
NT     = 128;
NX     = 128;
FS = 11;   LW = 1.5;

cfg_es        = cfg_ladmm_gaussian_expsemi();
cfg_es.vareps = VAREPS;
cfg_es.nt     = NT;
cfg_es.nx     = NX;

cfg_sink.vareps       = VAREPS;
cfg_sink.max_iter     = 500;
cfg_sink.tol          = 1e-10;
cfg_sink.precomp_heat = @precomp_heat_neumann;

prob_def = prob_gaussian();
problem  = setup_problem(cfg_es, prob_def);

nt     = problem.nt;    ntm = nt - 1;
dt     = problem.dt;    dx  = problem.dx;
nx     = problem.nx;    nxm = nx - 1;
vareps = cfg_es.vareps;
ftag   = sprintf('nt%d_nx%d_eps%g', NT, NX, VAREPS);

%% --- Step 1: Run Sinkhorn ---
fprintf('Running Sinkhorn (eps=%.4g)...\n', vareps);
res_sink = sinkhorn_hopf_cole(problem, cfg_sink);
fprintf('  iters=%d  converged=%d  err=%.2e  wall=%.2fs\n', ...
    res_sink.iters, res_sink.converged, res_sink.error, res_sink.walltime);

%% --- Step 2: Build warm-start initial conditions ---
% Converged Sinkhorn potentials
phi_0 = res_sink.phi(1,   :)';   % (nx x 1)  forward factor at t=0
psi_T = res_sink.psi(end, :)';   % (nx x 1)  backward factor at t=T

% Heat kernel for arbitrary-time evaluation
heat = cfg_sink.precomp_heat(problem, cfg_sink);

% Precompute DCT of potentials once
lambda_x = problem.lambda_x(:);   % (nx x 1)
phi_0_hat = dct(phi_0);
psi_T_hat = dct(psi_T);

apply_t = @(f_hat, t) max(idct(exp(-vareps * lambda_x * t) .* f_hat), 0);

% x0.rho at t = k*dt, k = 1..ntm
x0_warm.rho = zeros(ntm, nx);
for k = 1:ntm
    t_k = k * dt;
    phi_k = apply_t(phi_0_hat, t_k);
    psi_k = apply_t(psi_T_hat, 1 - t_k);
    x0_warm.rho(k, :) = (phi_k .* psi_k)';
end

% x0.mx at t = (k-0.5)*dt, k = 1..nt   (staggered-x momentum: nxm wide)
% m = 2*eps * phi_stag * d_x(psi)
x0_warm.mx = zeros(nt, nxm);
for k = 1:nt
    t_k = (k - 0.5) * dt;
    phi_k = apply_t(phi_0_hat, t_k);
    psi_k = apply_t(psi_T_hat, 1 - t_k);
    phi_stag  = 0.5 * (phi_k(1:nxm) + phi_k(2:nx));
    dpsi_stag = (psi_k(2:nx) - psi_k(1:nxm)) / dx;
    x0_warm.mx(k, :) = (2 * vareps * phi_stag .* dpsi_stag)';
end

% y0.rho at t = (k-0.5)*dt, k = 1..nt  (cell-centre density)
y0_warm.rho = zeros(nt, nx);
for k = 1:nt
    t_k = (k - 0.5) * dt;
    phi_k = apply_t(phi_0_hat, t_k);
    psi_k = apply_t(psi_T_hat, 1 - t_k);
    y0_warm.rho(k, :) = (phi_k .* psi_k)';
end

% y0.mx: cell-centre momentum -- leave at zero (updated on first ADMM step)
y0_warm.mx = zeros(nt, nx);

%% --- Step 3: Cold-start ADMM (default linear interpolation) ---
fprintf('Running cold-start ADMM...\n');
res_cold = discretize_then_optimize(cfg_es, problem);
fprintf('  iters=%d  converged=%d  err=%.2e  wall=%.2fs\n', ...
    res_cold.iters, res_cold.converged, res_cold.error, res_cold.walltime);

%% --- Step 4: Warm-start ADMM (initialised from Sinkhorn) ---
fprintf('Running warm-start ADMM...\n');
cfg_warm     = cfg_es;
cfg_warm.x0  = x0_warm;
cfg_warm.y0  = y0_warm;
res_warm = discretize_then_optimize(cfg_warm, problem);
fprintf('  iters=%d  converged=%d  err=%.2e  wall=%.2fs\n', ...
    res_warm.iters, res_warm.converged, res_warm.error, res_warm.walltime);

%% --- Figure 1: Residual comparison ---
fig1 = figure('Units','centimeters','Position',[2 2 16 9]);
hold on;
semilogy(res_cold.residual, 'b-',  'LineWidth', LW, 'DisplayName', 'Cold (linear interp)');
semilogy(res_warm.residual, 'r-',  'LineWidth', LW, 'DisplayName', 'Warm (Sinkhorn)');
yline(cfg_es.tol, 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');
xlabel('ADMM iteration', 'FontSize', FS);
ylabel('$\|y^{k+1} - y^k\|$', 'FontSize', FS);
title(sprintf('ADMM residual: cold vs warm start  ($\\varepsilon=%.4g$)', vareps), 'FontSize', FS);
legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;
saveas(fig1, fullfile(fig_dir, sprintf('warmstart_residual_%s.png', ftag)));

%% --- Figure 2: Final density comparison ---
% Compare all three at the staggered interior times t = k*dt, k=1..ntm
rho_sink_stag = res_sink.rho(2:nt, :);   % (ntm x nx)

n_plot  = 5;
t_fracs = linspace(0.1, 0.9, n_plot);
cmap    = lines(n_plot);
stride  = max(1, floor(nx / 60));

t_stag_vec = (1:ntm)' * dt;
err_cold   = sqrt(dx * sum((res_cold.rho_stag - rho_sink_stag).^2, 2));
err_warm   = sqrt(dx * sum((res_warm.rho_stag - rho_sink_stag).^2, 2));
norm_ref   = sqrt(dx * sum(rho_sink_stag.^2, 2));
rel_cold   = err_cold ./ norm_ref;
rel_warm   = err_warm ./ norm_ref;

fig2 = figure('Units','centimeters','Position',[2 2 30 9]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile; hold on;
for p = 1:n_plot
    [~, k] = min(abs((1:ntm)*dt - t_fracs(p)));
    idx = 1:stride:nx;
    plot(problem.xx, rho_sink_stag(k,:),           '-', 'Color', cmap(p,:), 'LineWidth', LW, 'HandleVisibility', 'off');
    plot(problem.xx(idx), res_cold.rho_stag(k,idx),'o', 'Color', cmap(p,:), 'MarkerSize', 3, 'HandleVisibility', 'off');
    plot(problem.xx(idx), res_warm.rho_stag(k,idx),'s', 'Color', cmap(p,:), 'MarkerSize', 3, 'HandleVisibility', 'off');
end
h1 = plot(nan,nan,'k-', 'LineWidth', LW, 'DisplayName', 'Sinkhorn');
h2 = plot(nan,nan,'ko', 'MarkerSize', 3, 'DisplayName', sprintf('Cold  (%d iters)', res_cold.iters));
h3 = plot(nan,nan,'ks', 'MarkerSize', 3, 'DisplayName', sprintf('Warm  (%d iters)', res_warm.iters));
legend([h1 h2 h3], 'Location', 'best', 'FontSize', FS-1, 'Box', 'off');
xlabel('$x$', 'FontSize', FS);
ylabel('$\rho$', 'FontSize', FS);
title(sprintf('Final density  ($\\varepsilon=%.4g$)', vareps), 'FontSize', FS);
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

mk_stride = max(1, floor(ntm / 30));
mk_idx    = 1:mk_stride:ntm;

nexttile; hold on;
semilogy(t_stag_vec, err_cold, 'b-',  'LineWidth', LW, 'HandleVisibility', 'off');
semilogy(t_stag_vec(mk_idx), err_cold(mk_idx), 'bo', 'MarkerSize', 4, 'LineWidth', 0.8, 'DisplayName', 'Cold');
semilogy(t_stag_vec, err_warm, 'r-',  'LineWidth', LW, 'DisplayName', 'Warm');
xlabel('$t$', 'FontSize', FS);
ylabel('$\|\rho - \rho_\mathrm{Sink}\|_{L^2(x)}$', 'FontSize', FS);
title('Absolute $L^2$ error vs Sinkhorn', 'FontSize', FS);
legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

nexttile; hold on;
semilogy(t_stag_vec, rel_cold, 'b-',  'LineWidth', LW, 'HandleVisibility', 'off');
semilogy(t_stag_vec(mk_idx), rel_cold(mk_idx), 'bo', 'MarkerSize', 4, 'LineWidth', 0.8, 'DisplayName', 'Cold');
semilogy(t_stag_vec, rel_warm, 'r-',  'LineWidth', LW, 'DisplayName', 'Warm');
xlabel('$t$', 'FontSize', FS);
ylabel('$\|\rho - \rho_\mathrm{Sink}\|_{L^2} / \|\rho_\mathrm{Sink}\|_{L^2}$', 'FontSize', FS);
title('Relative $L^2$ error vs Sinkhorn', 'FontSize', FS);
legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
grid on;

saveas(fig2, fullfile(fig_dir, sprintf('warmstart_density_%s.png', ftag)));

%% --- Summary ---
fprintf('\n--- Summary (eps=%.4g, nt=%d, nx=%d) ---\n', vareps, NT, NX);
fprintf('  Sinkhorn:    wall=%.2fs  iters=%d\n', res_sink.walltime, res_sink.iters);
fprintf('  Cold ADMM:   wall=%.2fs  iters=%d  converged=%d\n', ...
    res_cold.walltime, res_cold.iters, res_cold.converged);
fprintf('  Warm ADMM:   wall=%.2fs  iters=%d  converged=%d\n', ...
    res_warm.walltime, res_warm.iters, res_warm.converged);
fprintf('  Speedup (iters): %.1fx\n', res_cold.iters / max(res_warm.iters, 1));
fprintf('  Max abs  err (cold/warm): %.3e / %.3e\n', max(err_cold), max(err_warm));
fprintf('  Max rel  err (cold/warm): %.3e / %.3e\n', max(rel_cold), max(rel_warm));
