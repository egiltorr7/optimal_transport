%% sweep_eps.m
%
% Sweep over epsilon (Schrodinger Bridge regularization) and plot
% convergence and kinetic energy as a function of eps.
%
% Usage:
%   Run this script directly from MATLAB.
%   Override any base config field before calling run_sweep, e.g.:
%     cfg.gamma = 20;

clear; clc;

base = fileparts(mfilename('fullpath'));
addpath(fullfile(base, '..'));
addpath(fullfile(base, '..', 'config'));
addpath(fullfile(base, '..', 'problems'));
addpath(fullfile(base, '..', 'discretization'));
addpath(fullfile(base, '..', 'projection'));
addpath(fullfile(base, '..', 'prox'));
addpath(fullfile(base, '..', 'pipelines'));
addpath(fullfile(base, '..', 'utils'));

%% Setup
cfg      = cfg_staggered_gaussian();
prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

%% Define sweep
eps_values = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4];

sweep = run_sweep(cfg, problem, 'vareps', eps_values);

%% Extract quantities of interest
n          = length(sweep);
obj_vals   = zeros(n, 1);
iters      = zeros(n, 1);
converged  = false(n, 1);

for k = 1:n
    r           = sweep(k).result;
    obj_vals(k) = compute_objective(r.rho, r.mx, problem);
    iters(k)    = r.iters;
    converged(k) = r.converged;
end

%% Grid label for titles and figure names
grid_str      = sprintf('nt=%d, nx=%d', problem.nt, problem.nx);
grid_str_tex  = sprintf('$n_t=%d,\\, n_x=%d$', problem.nt, problem.nx);

%% Plot: kinetic energy vs eps
figure('Name', sprintf('Kinetic Energy vs epsilon  [%s]', grid_str));
semilogx(eps_values(2:end), obj_vals(2:end), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
hold on;
yline(obj_vals(1), 'k--', 'LineWidth', 1.5);
xlabel('$\epsilon$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Kinetic energy $\int\!\int m^2/\rho\,dt\,dx$', 'Interpreter', 'latex', 'FontSize', 14);
title({'Effect of $\epsilon$ on optimal kinetic energy', grid_str_tex}, ...
      'Interpreter', 'latex', 'FontSize', 13);
legend('$\epsilon > 0$', '$\epsilon = 0$ baseline', 'Interpreter', 'latex', 'Location', 'best');
grid on;

%% Plot: ADMM iterations vs eps
figure('Name', sprintf('Iterations vs epsilon  [%s]', grid_str));
semilogx(eps_values(2:end), iters(2:end), 'rs', 'MarkerSize', 8, 'LineWidth', 1.5);
hold on;
yline(iters(1), 'k--', 'LineWidth', 1.5);
xlabel('$\epsilon$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('ADMM iterations', 'Interpreter', 'latex', 'FontSize', 14);
title({'Effect of $\epsilon$ on ADMM convergence', grid_str_tex}, ...
      'Interpreter', 'latex', 'FontSize', 13);
legend('$\epsilon > 0$', '$\epsilon = 0$ baseline', 'Interpreter', 'latex', 'Location', 'best');
grid on;

%% Plot: density evolution in time for each eps
xx   = problem.xx;
nt   = problem.nt;   ntm = nt - 1;
dt   = problem.dt;

% Time slices to show (interior + boundaries)
t_idx  = round(linspace(1, ntm, 5));   % 5 interior slices
t_vals = t_idx * dt;

colors = lines(length(t_idx));

ncols = ceil(n/2);
nrows = 2;
figure('Name', sprintf('Density evolution per epsilon  [%s]', grid_str), 'Position', [100, 100, 300*ncols, 500]);
for k = 1:n
    subplot(nrows, ncols, k);
    rho_k = sweep(k).result.rho;   % (ntm x nx)

    % plot rho0 and rho1 as dashed boundaries
    plot(xx, problem.rho0, 'k--', 'LineWidth', 1); hold on;
    plot(xx, problem.rho1, 'k:',  'LineWidth', 1);

    for j = 1:length(t_idx)
        plot(xx, rho_k(t_idx(j), :), 'Color', colors(j,:), 'LineWidth', 1.5);
    end

    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 12);
    if mod(k-1, ncols) == 0
        ylabel('$\rho(t,x)$', 'Interpreter', 'latex', 'FontSize', 12);
    end
    title(sprintf('$\\epsilon = %g$', eps_values(k)), ...
          'Interpreter', 'latex', 'FontSize', 12);

    leg_entries = {'$\rho_0$', '$\rho_1$'};
    for j = 1:length(t_idx)
        leg_entries{end+1} = sprintf('$t=%.2f$', t_vals(j)); %#ok<AGROW>
    end
    if k == n
        legend(leg_entries, 'Interpreter', 'latex', 'Location', 'best', 'FontSize', 8);
    end
    grid on;
end

%% Plot: ADMM residual history ||u_{k+1} - u_k|| for each eps
figure('Name', sprintf('ADMM residual history  [%s]', grid_str));
colors2 = lines(n);
leg_res = cell(n, 1);

for k = 1:n
    semilogy(sweep(k).result.residual, 'Color', colors2(k,:), 'LineWidth', 1.5);
    hold on;
    leg_res{k} = sprintf('$\\epsilon = %g$', eps_values(k));
end

xlabel('Iteration', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\|(\rho,m)^{k+1} - (\rho,m)^k\|$', 'Interpreter', 'latex', 'FontSize', 14);
title({'ADMM convergence: $\|u^{k+1} - u^k\|$ per iteration', grid_str_tex}, ...
      'Interpreter', 'latex', 'FontSize', 13);
legend(leg_res, 'Interpreter', 'latex', 'Location', 'northeast');
grid on;

%% Save figures
fig_dir = fullfile(base, '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

figHandles = findall(0, 'Type', 'figure');
for f = 1:length(figHandles)
    fname = strrep(figHandles(f).Name, ' ', '_');
    fname = strrep(fname, '[', ''); fname = strrep(fname, ']', '');
    fname = strrep(fname, ',', ''); fname = strrep(fname, '=', '');
    saveas(figHandles(f), fullfile(fig_dir, [fname '.png']));
end
fprintf('\nFigures saved to: %s\n', fig_dir);

%% Print summary table
fprintf('\n%-10s  %-12s  %-8s  %-10s\n', 'eps', 'KE', 'iters', 'converged');
fprintf('%s\n', repmat('-', 1, 46));
for k = 1:n
    fprintf('%-10.2e  %-12.6f  %-8d  %s\n', ...
            eps_values(k), obj_vals(k), iters(k), mat2str(converged(k)));
end
