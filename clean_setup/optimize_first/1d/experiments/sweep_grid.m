%% sweep_grid.m
%
% Grid refinement study for Gaussian-to-Gaussian Schrödinger bridge (eps=0.1).
%
% Grids: (nt,nx) = (16,32) -> (32,64) -> (64,128) -> (128,256) -> (256,512)
%
% Plots:
%   1. Density evolution rho(t,x) at selected time slices vs analytical SB
%   2. L2 error ||(rho,m) - (rho,m)_exact|| vs n_x  (log-log)
%   3. ADMM kinetic energy vs grid level vs analytical SB kinetic energy

clear; clc; close all;

base = fileparts(mfilename('fullpath'));
addpath(fullfile(base, '..'));
addpath(fullfile(base, '..', 'config'));
addpath(fullfile(base, '..', 'problems'));
addpath(fullfile(base, '..', 'discretization'));
addpath(fullfile(base, '..', 'projection'));
addpath(fullfile(base, '..', 'prox'));
addpath(fullfile(base, '..', 'pipelines'));
addpath(fullfile(base, '..', 'utils'));

%% Grid levels (nt doubles, nx doubles each level)
nt_vals = [16,  32,  64,  128, 256];
nx_vals = [32,  64,  128, 256, 512];
nt_vals = [16,  32,  64];
nx_vals = [32,  64,  128];
n_grids = length(nt_vals);

%% Base config: eps=0.1 (Schrödinger bridge), banded projection (exact for eps>0)
cfg_base            = cfg_staggered_gaussian();
cfg_base.vareps     = 1e-1;
cfg_base.projection = @proj_fokker_planck_banded;
prob_def            = prob_gaussian();

%% Run ADMM on each grid
results   = cell(n_grids, 1);
problems  = cell(n_grids, 1);
rho_ana   = cell(n_grids, 1);
mx_ana    = cell(n_grids, 1);
ke_admm   = zeros(n_grids, 1);
ke_ana    = zeros(n_grids, 1);
err_rho   = zeros(n_grids, 1);
err_mx    = zeros(n_grids, 1);

for k = 1:n_grids
    cfg_k    = cfg_base;
    cfg_k.nt = nt_vals(k);
    cfg_k.nx = nx_vals(k);

    prob_k   = setup_problem(cfg_k, prob_def);

    fprintf('[%d/%d]  nt=%d  nx=%d ... ', k, n_grids, nt_vals(k), nx_vals(k));
    r = cfg_k.pipeline(cfg_k, prob_k);
    fprintf('iters=%d  error=%.2e  time=%.1fs\n', r.iters, r.error, r.walltime);

    [rho_a, mx_a] = analytical_sb_gaussian(prob_k, cfg_base.vareps);

    ke_admm(k) = compute_objective(r.rho, r.mx, prob_k) / prob_k.dx;
    ke_ana(k)  = compute_objective(rho_a, mx_a, prob_k) / prob_k.dx;

    dt = prob_k.dt;   dx = prob_k.dx;
    err_rho(k) = norm(r.rho(:) - rho_a(:)) * sqrt(dt * dx);
    err_mx(k)  = norm(r.mx(:)  - mx_a(:))  * sqrt(dt * dx);

    results{k}  = r;
    problems{k} = prob_k;
    rho_ana{k}  = rho_a;
    mx_ana{k}   = mx_a;
end

%% -----------------------------------------------------------------------
%% Plot 1: Density evolution per grid level (analytical overlaid)
%% -----------------------------------------------------------------------
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];   % time slices to show
colors   = lines(length(t_fracs));

ncols = ceil(n_grids / 2);
nrows = 2;
figure('Name', sprintf('Density evolution ADMM vs analytical gamma%g', cfg_base.gamma), ...
       'Position', [100, 100, 300*ncols, 500]);

for k = 1:n_grids
    subplot(nrows, ncols, k);

    prob_k = problems{k};
    ntm_k  = prob_k.nt - 1;
    xx_k   = prob_k.xx;
    dt_k   = prob_k.dt;

    rho_k  = results{k}.rho;
    rho_ak = rho_ana{k};

    hold on;
    for j = 1:length(t_fracs)
        idx = round(t_fracs(j) / dt_k);
        idx = max(1, min(ntm_k, idx));

        plot(xx_k, rho_k(idx, :),  'o', 'Color', colors(j,:), 'MarkerSize', 3, ...
             'MarkerFaceColor', colors(j,:), 'LineStyle', 'none');
        plot(xx_k, rho_ak(idx, :), '-', 'Color', colors(j,:), 'LineWidth', 1.5);
    end

    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 11);
    if mod(k-1, ncols) == 0
        ylabel('$\rho(t,x)$', 'Interpreter', 'latex', 'FontSize', 11);
    end
    title(sprintf('$n_t=%d,\\, n_x=%d$', nt_vals(k), nx_vals(k)), ...
          'Interpreter', 'latex', 'FontSize', 11);
    grid on;
end

% Shared legend in the empty 6th subplot slot
ax_leg = subplot(nrows, ncols, n_grids + 1);
hold on; axis off;
leg = {};
for j = 1:length(t_fracs)
    plot(nan, nan, 'o', 'Color', colors(j,:), 'MarkerFaceColor', colors(j,:), ...
         'MarkerSize', 5, 'LineStyle', 'none');
    plot(nan, nan, '-', 'Color', colors(j,:), 'LineWidth', 1.5);
    leg{end+1} = sprintf('ADMM $t=%.2f$', t_fracs(j)); %#ok<AGROW>
    leg{end+1} = sprintf('Exact $t=%.2f$', t_fracs(j)); %#ok<AGROW>
end
legend(leg, 'Interpreter', 'latex', 'FontSize', 8, ...
       'Location', 'best', 'NumColumns', 2);

%% -----------------------------------------------------------------------
%% Plot 2: L2 error vs n_x  (log-log)
%% -----------------------------------------------------------------------
err_total = sqrt(err_rho.^2 + err_mx.^2);

figure('Name', 'Grid refinement: L2 error vs nx');
loglog(nx_vals, err_total, 'k^-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;

% O(1/nx^2) reference line anchored at coarsest grid
loglog(nx_vals, err_total(1)*(nx_vals(1)./nx_vals).^2,  'k--',  'LineWidth', 1.2);

xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$L^2$ error', 'Interpreter', 'latex', 'FontSize', 14);
title('Grid refinement: $\|(\rho,m) - (\rho,m)_{\mathrm{exact}}\|$ vs $n_x$', ...
      'Interpreter', 'latex', 'FontSize', 13);
legend('$\|(\rho,m) - (\rho,m)_{\mathrm{exact}}\|$', ...
       '$\mathcal{O}(h^2)$', ...
       'Interpreter', 'latex', 'Location', 'southwest');
grid on;

%% -----------------------------------------------------------------------
%% Plot 3: Kinetic energy vs grid level
%% -----------------------------------------------------------------------
figure('Name', 'Grid refinement: kinetic energy vs grid level');
semilogx(nx_vals, ke_admm, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
semilogx(nx_vals, ke_ana,  'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);

xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Kinetic energy $\int\!\!\int m^2/\rho\,dt\,dx$', 'Interpreter', 'latex', 'FontSize', 14);
title(sprintf('Kinetic energy vs grid refinement  ($\\varepsilon=%.1g$)', cfg_base.vareps), ...
      'Interpreter', 'latex', 'FontSize', 13);
legend('ADMM', 'Analytical SB', 'Interpreter', 'latex', 'Location', 'best');
grid on;

%% -----------------------------------------------------------------------
%% Figure 4: Total mass in [0,1] vs time
%% -----------------------------------------------------------------------
% Continuous analytical mass in [0,1]: fraction of N(mu_t, sig_t^2) inside domain.
% For eps>0 sig_t grows at mid-time, so the Gaussian bleeds outside [0,1].
% ADMM enforces zero-flux BCs => mass is conserved at 1 for all t.
% The gap between the two reveals the BC modelling error.

vareps = cfg_base.vareps;
sigma  = 0.05;
alpha  = sqrt(sigma^4 + vareps^2) - sigma^2;

colors_grid = lines(n_grids);

figure('Name', sprintf('Total mass vs time gamma%g', cfg_base.gamma), ...
       'Position', [100, 100, 700, 420]);
hold on;

for k = 1:n_grids
    prob_k  = problems{k};
    rho_k   = results{k}.rho;
    ntm_k   = prob_k.nt - 1;
    dt_k    = prob_k.dt;

    % Interior time points + boundaries
    t_int  = (1:ntm_k)' * dt_k;
    t_full = [0; t_int; 1];

    % ADMM mass at each interior time + known boundaries
    mass_admm = [sum(prob_k.rho0); sum(rho_k, 2); sum(prob_k.rho1)];

    plot(t_full, mass_admm, '-', 'Color', colors_grid(k,:), 'LineWidth', 1.2, ...
         'DisplayName', sprintf('ADMM $n_x=%d$', nx_vals(k)));
end

% Continuous analytical mass: integrate N(mu_t, sig_t^2) over [0,1]
t_cont  = linspace(0, 1, 500)';
mu_cont = (1 - t_cont)/3 + t_cont*2/3;
sig_cont = sqrt(sigma^2 + 2*alpha * t_cont .* (1 - t_cont));
Phi = @(z) 0.5 * (1 + erf(z / sqrt(2)));
mass_cont = Phi((1 - mu_cont)./sig_cont) - Phi(-mu_cont./sig_cont);

plot(t_cont, mass_cont, 'k--', 'LineWidth', 2.0, ...
     'DisplayName', 'Analytical SB (full domain)');

xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\sum_j \rho_j(t)$', 'Interpreter', 'latex', 'FontSize', 14);
title(sprintf('Total mass in $[0,1]$ vs time  ($\\varepsilon=%.1g,\\, \\gamma=%g$)', ...
    vareps, cfg_base.gamma), 'Interpreter', 'latex', 'FontSize', 13);
legend('Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 9);
grid on;

%% -----------------------------------------------------------------------
%% Save figures
%% -----------------------------------------------------------------------
fig_dir = fullfile(base, '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

test_tag = 'sweep_grid_gaussian';
cfg_tag  = sprintf('eps%g_gam%g', cfg_base.vareps, cfg_base.gamma);
figs = findall(0, 'Type', 'figure');
for fi = 1:numel(figs)
    raw   = figs(fi).Name;
    clean = regexprep(raw, '[^\w]', '_');
    clean = regexprep(clean, '_+', '_');
    clean = strtrim(clean);
    fname = sprintf('%s__%s__%s', test_tag, clean, cfg_tag);
    saveas(figs(fi), fullfile(fig_dir, [fname '.png']));
end
fprintf('\nFigures saved to: %s\n', fig_dir);

%% -----------------------------------------------------------------------
%% Print summary table
%% -----------------------------------------------------------------------
fprintf('\n%-12s  %-10s  %-12s  %-12s  %-12s  %-12s\n', ...
        'nt x nx', 'iters', 'KE (ADMM)', 'KE (exact disc)', 'err_rho', 'err_m');
fprintf('%s\n', repmat('-', 1, 76));
for k = 1:n_grids
    fprintf('%3d x %-6d  %-10d  %-12.6f  %-12.6f  %-12.2e  %-12.2e\n', ...
            nt_vals(k), nx_vals(k), results{k}.iters, ...
            ke_admm(k), ke_ana(k), err_rho(k), err_mx(k));
end
fprintf('\nAnalytical SB KE (finest grid) = %.6f\n', ke_ana(end));
