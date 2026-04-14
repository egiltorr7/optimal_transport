% TEST_SB_GAUSSIAN_SINKHORN  Compare ADMM vs dynamic Sinkhorn (Hopf-Cole) on Gaussian SB.
%
%   Runs both solvers on N(1/3, 0.05²) -> N(2/3, 0.05²) with the same ε,
%   compares density and momentum against the analytical SB solution, and
%   plots density evolution, L2 error histories, and convergence.
%
%   Run setup_paths first, or call from the repo root.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Shared config ---
cfg_admm       = cfg_ladmm_gaussian();
cfg_admm.vareps = 1.0;   % set ε explicitly (must match between both solvers)

cfg_sink.vareps   = cfg_admm.vareps;
cfg_sink.max_iter = 500;
cfg_sink.tol      = 1e-10;

prob_def = prob_gaussian();
problem  = setup_problem(cfg_admm, prob_def);

ftag = sprintf('nt%d_nx%d_eps%g', cfg_admm.nt, cfg_admm.nx, cfg_admm.vareps);

%% --- Run ADMM ---
fprintf('Running ADMM  (nt=%d, nx=%d, eps=%.4g)...\n', ...
    cfg_admm.nt, cfg_admm.nx, cfg_admm.vareps);
res_admm = discretize_then_optimize(cfg_admm, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    res_admm.iters, res_admm.converged, res_admm.error, res_admm.walltime);

%% --- Run Sinkhorn ---
fprintf('Running Sinkhorn-Hopf-Cole  (eps=%.4g)...\n', cfg_sink.vareps);
res_sink = sinkhorn_hopf_cole(problem, cfg_sink);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    res_sink.iters, res_sink.converged, res_sink.error, res_sink.walltime);

%% --- Analytical solution ---
% On cell-centre times t = (k-0.5)*dt (for ADMM comparison)
[rho_ana_stag, ~] = analytical_sb_gaussian(problem, cfg_admm.vareps);
ops     = problem.ops;
rho_ana_cc = ops.interp_t_at_phi(rho_ana_stag, problem.rho0, problem.rho1);   % (nt x nx)

% On Sinkhorn edge-time grid t = k*dt, k=0..nt
nt   = problem.nt;   dt = problem.dt;   nx = problem.nx;   dx = problem.dx;
xx   = problem.xx;

mu0 = 1/3;   mu1 = 2/3;   sigma = 0.05;
vareps = cfg_admm.vareps;
alpha  = sqrt(sigma^4 + vareps^2) - sigma^2;
Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

t_edge = res_sink.t_grid;   % (nt+1 x 1)
rho_ana_edge = zeros(nt+1, nx);
for k = 1:(nt+1)
    t_k   = t_edge(k);
    mu_k  = (1-t_k)*mu0 + t_k*mu1;
    sig_k = sqrt(sigma^2 + 2*alpha*t_k*(1-t_k));
    row   = Normal(xx, mu_k, sig_k);
    rho_ana_edge(k,:) = row / sum(row);
end

%% --- Figure 1: Density evolution comparison ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
n_t     = numel(t_fracs);
colors  = parula(n_t);

figure('Name', sprintf('SB Gaussian density  eps=%.4g', vareps), ...
    'Position', [50 50 900 420]);

tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Left panel: ADMM vs analytical
nexttile;
hold on;
for p = 1:n_t
    k     = max(1, round(t_fracs(p) * nt));
    t_val = (k - 0.5) * dt;
    stride = max(1, floor(nx / 60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana_cc(k,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), res_admm.rho_cc(k,idx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 4, 'MarkerFaceColor', colors(p,:));
end
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('ADMM  ($\\varepsilon$=%.4g)', vareps), 'Interpreter', 'latex');
grid on;

% Right panel: Sinkhorn vs analytical (at edge times closest to t_fracs)
nexttile;
hold on;
for p = 1:n_t
    k     = max(1, round(t_fracs(p) * nt));
    t_val = k * dt;
    stride = max(1, floor(nx / 60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana_edge(k+1,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), res_sink.rho(k+1,idx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 4, 'MarkerFaceColor', colors(p,:));
end
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('Sinkhorn-HC  ($\\varepsilon$=%.4g)', vareps), 'Interpreter', 'latex');
grid on;

sgtitle(sprintf('Density evolution   $\\varepsilon$=%.4g  (solid=analytical, circles=numerical)', ...
    vareps), 'Interpreter', 'latex');
saveas(gcf, fullfile(fig_dir, sprintf('sinkhorn_density_%s.png', ftag)));

%% --- Figure 2: L2 error in rho vs analytical ---
% ADMM: cell-centre times
err_admm = sqrt(dx * sum((res_admm.rho_cc - rho_ana_cc).^2, 2));   % (nt x 1)
t_cc     = ((1:nt)' - 0.5) * dt;

% Sinkhorn: edge times (skip t=0 and t=T boundary)
err_sink = sqrt(dx * sum((res_sink.rho(2:nt,:) - rho_ana_edge(2:nt,:)).^2, 2));   % (nt-1 x 1)
t_edge_inner = t_edge(2:nt);

figure('Name', sprintf('L2 error  eps=%.4g', vareps), 'Position', [50 550 700 300]);
hold on;
plot(t_cc,          err_admm,  'b-',  'LineWidth', 1.5, 'DisplayName', 'ADMM');
plot(t_edge_inner,  err_sink,  'r--', 'LineWidth', 1.5, 'DisplayName', 'Sinkhorn-HC');
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\|\rho_\mathrm{num} - \rho_\mathrm{ana}\|_{L^2(x)}$', 'Interpreter', 'latex');
title(sprintf('$L^2$ error in $\\rho$   $\\varepsilon$=%.4g', vareps), 'Interpreter', 'latex');
legend('Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('sinkhorn_l2error_%s.png', ftag)));

%% --- Figure 3: Sinkhorn convergence ---
figure('Name', 'Sinkhorn convergence', 'Position', [780 50 500 300]);
semilogy(res_sink.errors, 'b-', 'LineWidth', 1.5);
yline(cfg_sink.tol, 'r--', sprintf('tol = %.1e', cfg_sink.tol));
xlabel('Sinkhorn iteration');
ylabel('Left-marginal $L^2$ error', 'Interpreter', 'latex');
title(sprintf('Sinkhorn convergence   iters=%d', res_sink.iters));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('sinkhorn_convergence_%s.png', ftag)));

%% --- Figure 4: ADMM residual ---
figure('Name', 'ADMM residual', 'Position', [780 400 500 300]);
semilogy(res_admm.residual, 'b-', 'LineWidth', 1.5);
yline(cfg_admm.tol, 'r--', sprintf('tol = %.1e', cfg_admm.tol));
xlabel('ADMM iteration');
ylabel('$\|y^{k+1} - y^k\|$', 'Interpreter', 'latex');
title(sprintf('ADMM residual   iters=%d,  converged=%d', ...
    res_admm.iters, res_admm.converged));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('sinkhorn_admm_residual_%s.png', ftag)));

%% --- Summary ---
fprintf('\n--- Summary (eps=%.4g, nt=%d, nx=%d) ---\n', vareps, nt, nx);
fprintf('  ADMM:     wall=%.2fs,  iters=%d,  max L2 rho err=%.3e\n', ...
    res_admm.walltime, res_admm.iters, max(err_admm));
fprintf('  Sinkhorn: wall=%.2fs,  iters=%d,  max L2 rho err=%.3e\n', ...
    res_sink.walltime, res_sink.iters, max(err_sink));
