% SWEEP_GRID  Grid refinement study for Gaussian-to-Gaussian SB (discretize-first).
%
% Grids: (nt,nx) = (16,32) -> (32,64) -> (64,128) -> (128,256)
%
% Plots:
%   1. Density evolution at selected time slices vs analytical SB
%   2. L2 error ||(rho,m) - analytical|| vs n_x  (log-log)
%   3. Kinetic energy vs grid level vs analytical SB
%   4. Total mass in [0,1] vs time (BC artefact)

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% --- Parameters ---
nt_vals = [16,  32,  64,  128];
nx_vals = [32,  64,  128, 256];
n_grids = numel(nt_vals);

cfg_base        = cfg_ladmm_gaussian();
cfg_base.vareps = 1e-1;
prob_def        = prob_gaussian();

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
ftag = sprintf('sweep_grid_eps%g_gam%g_tau%g', cfg_base.vareps, cfg_base.gamma, cfg_base.tau);

%% --- Run ---
results  = cell(n_grids, 1);
problems = cell(n_grids, 1);
rho_ana  = cell(n_grids, 1);
mx_ana   = cell(n_grids, 1);
ke_admm  = zeros(n_grids, 1);
ke_ana   = zeros(n_grids, 1);
err_rho  = zeros(n_grids, 1);
err_mx   = zeros(n_grids, 1);

for k = 1:n_grids
    cfg_k    = cfg_base;
    cfg_k.nt = nt_vals(k);
    cfg_k.nx = nx_vals(k);
    prob_k   = setup_problem(cfg_k, prob_def);

    fprintf('[%d/%d]  nt=%d  nx=%d ... ', k, n_grids, nt_vals(k), nx_vals(k));
    r = cfg_k.pipeline(cfg_k, prob_k);
    fprintf('iters=%d  error=%.2e  time=%.1fs\n', r.iters, r.error, r.walltime);

    [rho_a, mx_a] = analytical_sb_gaussian(prob_k, cfg_base.vareps);

    % KE on staggered grid (x is FP-feasible staggered variable)
    ke_admm(k) = compute_objective(r.rho_stag, r.mx_stag, prob_k) / prob_k.dx;
    ke_ana(k)  = compute_objective(rho_a,      mx_a,      prob_k) / prob_k.dx;

    % L2 error on staggered grid
    dt = prob_k.dt;   dx = prob_k.dx;
    err_rho(k) = norm(r.rho_stag(:) - rho_a(:)) * sqrt(dt * dx);
    err_mx(k)  = norm(r.mx_stag(:)  - mx_a(:))  * sqrt(dt * dx);

    results{k}  = r;
    problems{k} = prob_k;
    rho_ana{k}  = rho_a;
    mx_ana{k}   = mx_a;
end

%% -----------------------------------------------------------------------
%% Plot 1: Density evolution per grid level
%% -----------------------------------------------------------------------
t_fracs    = [0.1, 0.25, 0.5, 0.75, 0.9];
colors_t   = lines(numel(t_fracs));
ncols      = ceil((n_grids + 1) / 2);   % +1 reserves a slot for the legend

figure('Position', [100 100 300*ncols 500]);
for k = 1:n_grids
    subplot(2, ncols, k);
    prob_k = problems{k};
    ntm_k  = prob_k.nt - 1;
    nx_k   = prob_k.nx;
    xx_k   = prob_k.xx;
    dt_k   = prob_k.dt;
    stride = max(1, floor(nx_k / 40));
    idx    = 1:stride:nx_k;

    hold on;
    for j = 1:numel(t_fracs)
        ti = max(1, min(ntm_k, round(t_fracs(j) / dt_k)));
        plot(xx_k, rho_ana{k}(ti,:), '-', 'Color', colors_t(j,:), 'LineWidth', 1.5);
        plot(xx_k(idx), results{k}.rho_stag(ti,idx), 'o', 'Color', colors_t(j,:), ...
             'MarkerSize', 5, 'MarkerFaceColor', colors_t(j,:));
    end
    title(sprintf('$n_t=%d,\\, n_x=%d$', nt_vals(k), nx_vals(k)), ...
          'Interpreter', 'latex');
    xlabel('$x$', 'Interpreter', 'latex');
    if mod(k-1, ncols) == 0
        ylabel('$\rho$', 'Interpreter', 'latex');
    end
    grid on;
end

% Legend in last subplot slot
ax_leg = subplot(2, ncols, n_grids + 1);
hold on; axis off;
leg = {};
for j = 1:numel(t_fracs)
    plot(nan, nan, '-',  'Color', colors_t(j,:), 'LineWidth', 1.5);
    plot(nan, nan, 'o',  'Color', colors_t(j,:), 'MarkerFaceColor', colors_t(j,:), 'MarkerSize', 5);
    leg{end+1} = sprintf('Ana $t=%.2f$',   t_fracs(j)); %#ok<AGROW>
    leg{end+1} = sprintf('LADMM $t=%.2f$', t_fracs(j)); %#ok<AGROW>
end
legend(leg, 'Interpreter', 'latex', 'FontSize', 8, 'Location', 'best', 'NumColumns', 2);
saveas(gcf, fullfile(fig_dir, sprintf('density_%s.png', ftag)));

%% -----------------------------------------------------------------------
%% Plot 2: L2 error vs n_x (log-log)
%% -----------------------------------------------------------------------
err_total = sqrt(err_rho.^2 + err_mx.^2);

figure('Position', [100 600 600 350]);
loglog(nx_vals, err_total, 'k^-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
loglog(nx_vals, err_total(1)*(nx_vals(1)./nx_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$L^2$ error', 'Interpreter', 'latex', 'FontSize', 13);
title('Grid refinement: $\|(\rho,m) - (\rho,m)_\mathrm{ana}\|$ vs $n_x$', ...
      'Interpreter', 'latex');
legend('$\|(\rho,m) - (\rho,m)_\mathrm{ana}\|$', '$\mathcal{O}(h^2)$', ...
       'Interpreter', 'latex', 'Location', 'southwest');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('l2error_%s.png', ftag)));

%% -----------------------------------------------------------------------
%% Plot 3: Kinetic energy vs grid level
%% -----------------------------------------------------------------------
figure('Position', [720 600 600 350]);
semilogx(nx_vals, ke_admm, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
semilogx(nx_vals, ke_ana,  'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$\iint m^2/\rho\,dt\,dx$', 'Interpreter', 'latex', 'FontSize', 13);
title(sprintf('Kinetic energy vs grid   $\\varepsilon=%.1g,\\, \\gamma=%g$', ...
    cfg_base.vareps, cfg_base.gamma), 'Interpreter', 'latex');
legend('LADMM', 'Analytical SB', 'Interpreter', 'latex', 'Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('ke_%s.png', ftag)));

%% -----------------------------------------------------------------------
%% Plot 4: Total mass in [0,1] vs time
%% -----------------------------------------------------------------------
vareps   = cfg_base.vareps;
sigma    = 0.05;
alpha_sb = sqrt(sigma^4 + vareps^2) - sigma^2;
Phi      = @(z) 0.5*(1 + erf(z/sqrt(2)));

colors_g = lines(n_grids);

figure('Position', [100 100 700 400]);
hold on;
for k = 1:n_grids
    prob_k = problems{k};
    ntm_k  = prob_k.nt - 1;
    dt_k   = prob_k.dt;
    t_int  = (1:ntm_k)' * dt_k;
    t_full = [0; t_int; 1];
    mass   = [sum(prob_k.rho0); sum(results{k}.rho_stag, 2); sum(prob_k.rho1)];
    plot(t_full, mass, '-', 'Color', colors_g(k,:), 'LineWidth', 1.2, ...
         'DisplayName', sprintf('LADMM $n_x=%d$', nx_vals(k)));
end

t_cont   = linspace(0, 1, 500)';
mu_cont  = (1-t_cont)/3 + t_cont*2/3;
sig_cont = sqrt(sigma^2 + 2*alpha_sb * t_cont .* (1 - t_cont));
mass_ana = Phi((1-mu_cont)./sig_cont) - Phi(-mu_cont./sig_cont);
plot(t_cont, mass_ana, 'k--', 'LineWidth', 2, 'DisplayName', 'Analytical SB (full domain)');

xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Total mass in $[0,1]$', 'Interpreter', 'latex', 'FontSize', 13);
title(sprintf('Total mass vs time   $\\varepsilon=%.1g,\\, \\gamma=%g$', ...
    vareps, cfg_base.gamma), 'Interpreter', 'latex');
legend('Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 9);
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('mass_%s.png', ftag)));

%% --- Summary table ---
fprintf('\n%-12s  %-8s  %-12s  %-12s  %-10s  %-10s\n', ...
    'nt x nx', 'iters', 'KE (LADMM)', 'KE (ana)', 'err_rho', 'err_mx');
fprintf('%s\n', repmat('-', 1, 72));
for k = 1:n_grids
    fprintf('%3d x %-6d  %-8d  %-12.6f  %-12.6f  %-10.2e  %-10.2e\n', ...
        nt_vals(k), nx_vals(k), results{k}.iters, ke_admm(k), ke_ana(k), ...
        err_rho(k), err_mx(k));
end
