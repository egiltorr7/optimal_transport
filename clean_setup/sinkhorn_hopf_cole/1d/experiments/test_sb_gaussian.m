% TEST_SB_GAUSSIAN  Quick test: Schrödinger bridge between two Gaussians.
%
%   Runs sinkhorn_hopf_cole on the Gaussian problem, compares density
%   evolution against the analytical SB solution, and plots the Sinkhorn
%   convergence history and L2 error vs time.
%
%   Run setup_paths first, or call this script from the repo root after
%   adding the required folders to the MATLAB path.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

% Output directory for figures
fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
cfg      = cfg_sinkhorn_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

%% --- Run ---
fprintf('Running sinkhorn_hopf_cole  (nt=%d, nx=%d, eps=%.4g)...\n', ...
    cfg.nt, cfg.nx, cfg.vareps);

result = sinkhorn_hopf_cole(problem, cfg);

fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

% Shared filename tag encoding all key parameters
ftag = sprintf('nt%d_nx%d_eps%g', cfg.nt, cfg.nx, cfg.vareps);

%% --- Analytical solution on edge-time grid t = 0, dt, ..., T ---
% Gaussian SB analytical formula:
%   mu(t)  = (1-t)*mu0 + t*mu1
%   sig(t) = sqrt(sig0^2 + 2*alpha*t*(1-t)),  alpha = sqrt(sig0^4 + eps^2) - sig0^2
nt  = problem.nt;
nx  = problem.nx;
dt  = problem.dt;
dx  = problem.dx;
xx  = problem.xx;

mu0   = 1/3;   mu1 = 2/3;   sigma = 0.05;
alpha = sqrt(sigma^4 + cfg.vareps^2) - sigma^2;
Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

t_grid = result.t_grid;          % (nt+1 x 1)
rho_ana = zeros(nt+1, nx);
for k = 1:(nt+1)
    t_k   = t_grid(k);
    mu_k  = (1-t_k)*mu0 + t_k*mu1;
    sig_k = sqrt(sigma^2 + 2*alpha*t_k*(1-t_k));
    row   = Normal(xx, mu_k, sig_k);
    rho_ana(k,:) = row / sum(row);
end

%% --- Figure 1: density evolution ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
n_t     = numel(t_fracs);
colors  = parula(n_t);

figure('Name', sprintf('SB Gaussian density  eps=%.4g', cfg.vareps), ...
    'Position', [100 100 700 400]);
hold on;

for p = 1:n_t
    k      = max(1, round(t_fracs(p) * nt));
    stride = max(1, floor(nx / 60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana(k+1,:), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result.rho(k+1,idx), 'o', 'Color', colors(p,:), ...
        'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end

leg_str = cell(2*n_t, 1);
for p = 1:n_t
    k = max(1, round(t_fracs(p) * nt));
    leg_str{2*p-1} = sprintf('Ana       t=%.2f', k*dt);
    leg_str{2*p}   = sprintf('Sinkhorn  t=%.2f', k*dt);
end
legend(leg_str, 'Location', 'best', 'FontSize', 7);

xlabel('$x$',    'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('Density evolution   $\\varepsilon$=%.4g', cfg.vareps), ...
    'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('density_%s.png', ftag)));

%% --- Figure 2: Sinkhorn convergence ---
figure('Name', sprintf('Sinkhorn convergence  eps=%.4g', cfg.vareps), ...
    'Position', [100 550 600 300]);

semilogy(result.errors, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol = %.1e', cfg.tol));
xlabel('Sinkhorn iteration');
ylabel('Left-marginal $L^2$ error', 'Interpreter', 'latex');
title(sprintf('Sinkhorn convergence   iters=%d,  converged=%d', ...
    result.iters, result.converged));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('convergence_%s.png', ftag)));

%% --- Figure 3: L2 error in rho vs analytical ---
% Skip t=0 and t=T (boundary conditions are exact by construction)
inner = 2:nt;
err   = sqrt(dx * sum((result.rho(inner,:) - rho_ana(inner,:)).^2, 2));

figure('Name', sprintf('L2 error  eps=%.4g', cfg.vareps), ...
    'Position', [720 550 600 300]);

plot(t_grid(inner), err, 'b-', 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\|\rho_\mathrm{Sinkhorn} - \rho_\mathrm{ana}\|_{L^2(x)}$', 'Interpreter', 'latex');
title('$L^2$ error in $\rho$ vs analytical SB', 'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('l2error_%s.png', ftag)));
