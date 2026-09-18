% TEST_SB_GAUSSIAN  Quick test: Schrödinger bridge between two Gaussians.
%
%   Runs sinkhorn_hopf_cole on the Gaussian problem, compares density
%   evolution against the analytical SB solution, and plots the Sinkhorn
%   convergence history and L2 error vs time.
%
%   The analytical comparison adapts to the kernel and marginal convention:
%     neumann     — rho0/rho1 normalised over [0,1], compared to domain-normalised Gaussian
%     free_space  — rho0/rho1 as raw PDF (use_pdf_marginals=true), compared to
%                   analytical_sb_gaussian_R (raw PDF values, no domain renormalisation)
%                   an extra figure shows total probability mass vs time to visualise leakage
%
%   Run setup_paths first, or call this script from the repo root after
%   adding the required folders to the MATLAB path.

clear; close all;
setup_paths_base = fullfile(fileparts(mfilename('fullpath')), '..');  %#ok<NASGU>
run(fullfile(setup_paths_base, 'setup_paths.m'));

% Output directory for figures
fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
cfg      = cfg_sinkhorn_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

%% --- Run ---
fprintf('Running sinkhorn_hopf_cole  (nt=%d, nx=%d, eps=%.4g, kernel=%s)...\n', ...
    cfg.nt, cfg.nx, cfg.vareps, func2str(cfg.precomp_heat));

result = sinkhorn_hopf_cole(problem, cfg);

fprintf('  kernel=%s,  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.heat_name, result.iters, result.converged, result.error, result.walltime);

% Shared filename tag encoding all key parameters
ftag = sprintf('nt%d_nx%d_eps%g_%s', cfg.nt, cfg.nx, cfg.vareps, result.heat_name);

nt = problem.nt;
nx = problem.nx;
dt = problem.dt;
dx = problem.dx;
xx = problem.xx;

%% --- Analytical solutions ---
% Neumann reference: domain-normalised Gaussian (correct for reflecting walls)
% Computed on edge-time grid to match Sinkhorn output
mu0 = 1/3;   mu1 = 2/3;   sigma = 0.05;
alpha  = sqrt(sigma^4 + cfg.vareps^2) - sigma^2;
Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

t_grid = result.t_grid;   % (nt+1 x 1)
rho_ana_neu = zeros(nt+1, nx);
for k = 1:(nt+1)
    t_k   = t_grid(k);
    mu_k  = (1 - t_k)*mu0 + t_k*mu1;
    sig_k = sqrt(sigma^2 + 2*alpha*t_k*(1 - t_k));
    row   = Normal(xx, mu_k, sig_k);
    rho_ana_neu(k,:) = row / sum(row);   % normalised over [0,1]
end

% Free-space reference: raw PDF values (no *dx, no domain renormalisation).
% Matches result.rho when cfg.use_pdf_marginals = true.
rho_ana_R = analytical_sb_gaussian_R(problem, cfg.vareps);

% Select the appropriate comparison for the kernel / marginal convention in use
use_pdf = isfield(cfg, 'use_pdf_marginals') && cfg.use_pdf_marginals;
if use_pdf
    rho_ana       = rho_ana_R;
    ana_label_tex = 'Analytical ($\mathbf{R}$)';   % for LaTeX titles/axes
    ana_label_leg = 'Analytical (R)';               % for legend entries (no special chars)
else
    rho_ana       = rho_ana_neu;
    ana_label_tex = 'Analytical (Neumann)';
    ana_label_leg = 'Analytical (Neumann)';
end

% Safe version of heat_name for LaTeX titles (replace _ with space)
heat_name_tex = strrep(result.heat_name, '_', ' ');

%% --- Figure 1: density evolution ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
n_t     = numel(t_fracs);
colors  = parula(n_t);

figure('Name', sprintf('SB Gaussian density  eps=%.4g  kernel=%s', ...
    cfg.vareps, result.heat_name), 'Position', [100 100 700 400]);
hold on;

for p = 1:n_t
    k      = max(1, round(t_fracs(p) * nt));
    stride = max(1, floor(nx / 60));
    idx    = 1:stride:nx;
    plot(xx, rho_ana(k+1,:),            '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    plot(xx(idx), result.rho(k+1,idx),  'o',  'Color', colors(p,:), ...
        'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
end

leg_str = cell(2*n_t, 1);
for p = 1:n_t
    k = max(1, round(t_fracs(p) * nt));
    leg_str{2*p-1} = sprintf('%s  t=%.2f', ana_label_leg, k*dt);
    leg_str{2*p}   = sprintf('Sinkhorn  t=%.2f', k*dt);
end
legend(leg_str, 'Location', 'best', 'FontSize', 7);

xlabel('$x$',    'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('Density evolution   $\\varepsilon$=%.4g,  kernel: %s', ...
    cfg.vareps, heat_name_tex), 'Interpreter', 'latex');
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

%% --- Figure 3: L2 error in rho vs appropriate analytical ---
% Skip t=0 and t=T (boundary conditions are exact by construction)
inner = 2:nt;
err   = sqrt(dx * sum((result.rho(inner,:) - rho_ana(inner,:)).^2, 2));

figure('Name', sprintf('L2 error  eps=%.4g  kernel=%s', cfg.vareps, result.heat_name), ...
    'Position', [720 550 600 300]);

plot(t_grid(inner), err, 'b-', 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\|\rho_\mathrm{num} - \rho_\mathrm{ana}\|_{L^2(x)}$', 'Interpreter', 'latex');
title(sprintf('$L^2$ error vs %s   $\\varepsilon$=%.4g', ...
    ana_label_leg, cfg.vareps), 'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('l2error_%s.png', ftag)));

%% --- Figure 4 (free_space only): total probability mass on [0,1] ---
% sum(rho * dx) gives total probability; drops below 1 when the Gaussian
% spreads outside [0,1].  Both curves should track each other if the
% free-space Sinkhorn is correctly reproducing the R solution.
if strcmp(result.heat_name, 'free_space')
    % multiply by dx to convert from PDF sum to total probability mass
    mass_num = sum(result.rho, 2) * dx;    % (nt+1 x 1)
    mass_ana = sum(rho_ana_R,  2) * dx;    % (nt+1 x 1): P(X in [0,1]) under N(mu_t, sig_t^2)

    figure('Name', sprintf('Total mass  eps=%.4g', cfg.vareps), ...
        'Position', [100 950 600 280]);
    hold on;
    plot(t_grid, mass_num, 'b-',  'LineWidth', 1.5, 'DisplayName', 'Sinkhorn (free space)');
    plot(t_grid, mass_ana, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Analytical ($\mathbf{R}$)');
    xlabel('$t$', 'Interpreter', 'latex');
    ylabel('$\sum_i \rho_i \, \Delta x$', 'Interpreter', 'latex');
    title(sprintf('Total probability on $[0,1]$   $\\varepsilon$=%.4g', cfg.vareps), ...
        'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex');
    grid on;
    saveas(gcf, fullfile(fig_dir, sprintf('mass_%s.png', ftag)));
end
