%% test_sb_bimodal.m
%
% Schrödinger bridge: bimodal -> unimodal.
%
%   rho0 = N(0.25, 0.05^2) + N(0.75, 0.05^2)   (two symmetric peaks)
%   rho1 = N(0.5,  0.08^2)                       (single central Gaussian)
%
% No closed-form analytical solution exists.  We assess quality via:
%   (a) visual density evolution — two lobes should merge into one;
%   (b) mass conservation at all interior times;
%   (c) KE monotone convergence as the grid refines (coarser grids
%       over-estimate the true minimum KE);
%   (d) ADMM residual histories confirm solver convergence;
%   (e) eps sweep at fixed grid shows transition from OT (sharp) to
%       diffusive (smooth) transport.

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

%% --- Parameters ---
vareps = 1e-1;

%% --- Grid levels ---
nt_vals = [16, 32, 64, 128];
nx_vals = [32, 64, 128, 256];
n_grids = numel(nt_vals);
i_ref   = n_grids;   % finest grid is the reference

%% --- Config + problem ---
cfg            = cfg_staggered_gaussian();
cfg.vareps     = vareps;
cfg.projection = @proj_fokker_planck_banded;

prob_def = prob_bimodal();

%% --- Run all grid levels ---
results  = cell(n_grids, 1);
problems = cell(n_grids, 1);
ke_vals  = zeros(n_grids, 1);

fprintf('Running SB: bimodal -> unimodal,  eps=%.2g\n', vareps);
for k = 1:n_grids
    cfg_k    = cfg;
    cfg_k.nt = nt_vals(k);
    cfg_k.nx = nx_vals(k);
    prob_k   = setup_problem(cfg_k, prob_def);

    fprintf('  [%d/%d] nt=%d, nx=%d ...', k, n_grids, nt_vals(k), nx_vals(k));
    r = cfg_k.pipeline(cfg_k, prob_k);
    fprintf(' iters=%d, err=%.2e, wall=%.1fs\n', r.iters, r.error, r.walltime);

    ke_vals(k)  = compute_objective(r.rho, r.mx, prob_k) / prob_k.dx;
    results{k}  = r;
    problems{k} = prob_k;
end

%% -----------------------------------------------------------------------
%% Figure 1: Density evolution (finest grid)
%% -----------------------------------------------------------------------
prob_ref = problems{i_ref};
r_ref    = results{i_ref};

nt_ref  = prob_ref.nt;   ntm_ref = nt_ref - 1;
dt_ref  = prob_ref.dt;
xx_ref  = prob_ref.xx;

t_fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
colors  = parula(numel(t_fracs));

figure('Name', sprintf('Bimodal SB: density evolution  eps=%.2g', vareps), ...
       'Position', [100, 100, 800, 420]);
hold on;

for p = 1:numel(t_fracs)
    tf = t_fracs(p);
    if tf == 0.0
        row    = prob_ref.rho0;   t_disp = 0;
    elseif tf == 1.0
        row    = prob_ref.rho1;   t_disp = 1;
    else
        k      = max(1, min(ntm_ref, round(tf * nt_ref)));
        row    = r_ref.rho(k, :);
        t_disp = k * dt_ref;
    end
    plot(xx_ref, row, '-', 'Color', colors(p,:), 'LineWidth', 1.8, ...
         'DisplayName', sprintf('$t=%.2f$', t_disp));
end

legend('Interpreter', 'latex', 'Location', 'best', 'FontSize', 9);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho(t,x)$', 'Interpreter', 'latex');
title(sprintf(['SB: bimodal $\\to$ unimodal  ' ...
               '($\\varepsilon=%.2g$,  $n_t=%d$,  $n_x=%d$)'], ...
              vareps, nt_vals(i_ref), nx_vals(i_ref)), 'Interpreter', 'latex');
grid on;

%% -----------------------------------------------------------------------
%% Figure 2: ADMM residual histories (all grid levels)
%% -----------------------------------------------------------------------
colors_grid = lines(n_grids);

figure('Name', 'Bimodal SB: ADMM residuals', 'Position', [100, 540, 600, 300]);
hold on;
for k = 1:n_grids
    semilogy(results{k}.residual, '-', 'Color', colors_grid(k,:), 'LineWidth', 1.2, ...
             'DisplayName', sprintf('$n_x=%d$', nx_vals(k)));
end
yline(cfg.tol, 'k--', sprintf('tol=%.0e', cfg.tol), 'LineWidth', 1.2);
xlabel('Iteration');
ylabel('$\|y^{k+1}-y^k\|$', 'Interpreter', 'latex');
title('ADMM residuals: bimodal SB', 'Interpreter', 'latex');
legend('Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 9);
grid on;

%% -----------------------------------------------------------------------
%% Figure 3: KE convergence with grid refinement
%% -----------------------------------------------------------------------
figure('Name', 'Bimodal: KE vs grid', 'Position', [730, 100, 560, 320]);
semilogx(nx_vals, ke_vals, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Kinetic energy', 'Interpreter', 'latex');
title(sprintf('KE convergence with grid  ($\\varepsilon=%.2g$)', vareps), ...
      'Interpreter', 'latex');
grid on;

% KE should approach the true minimum from above as the grid refines.
fprintf('\nKE values across grid levels:\n');
for k = 1:n_grids
    fprintf('  nt=%3d, nx=%3d  ->  KE = %.6f\n', nt_vals(k), nx_vals(k), ke_vals(k));
end

%% -----------------------------------------------------------------------
%% Figure 4: Mass conservation (finest grid)
%% -----------------------------------------------------------------------
ntm_ref  = nt_ref - 1;
t_full   = [0; (1:ntm_ref)' * dt_ref; 1];
mass_int = sum(r_ref.rho, 2);
mass     = [sum(prob_ref.rho0); mass_int; sum(prob_ref.rho1)];

figure('Name', 'Bimodal: mass conservation', 'Position', [730, 440, 560, 300]);
plot(t_full, mass, 'b-', 'LineWidth', 1.5);
yline(1, 'r--', 'mass = 1', 'LineWidth', 1.2);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\sum_i \rho_i$', 'Interpreter', 'latex');
title(sprintf('Mass conservation  ($n_t=%d$, $n_x=%d$)', nt_vals(i_ref), nx_vals(i_ref)), ...
      'Interpreter', 'latex');
ylim([0.9, 1.1]);
grid on;

mass_err = max(abs(mass - 1));
fprintf('\nMax mass deviation from 1 (fine grid): %.2e\n', mass_err);

%% -----------------------------------------------------------------------
%% Figure 5: eps sweep at the fine grid, density at t=0.5
%% -----------------------------------------------------------------------
eps_vals   = [0, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1];
cfg_fine   = cfg;
cfg_fine.nt = nt_vals(i_ref);
cfg_fine.nx = nx_vals(i_ref);
prob_fine  = setup_problem(cfg_fine, prob_def);
colors_eps = parula(numel(eps_vals));

k_half  = round(0.5 * nt_vals(i_ref));

figure('Name', 'Bimodal SB: eps sweep at t=0.5', 'Position', [100, 870, 800, 400]);
hold on;

for j = 1:numel(eps_vals)
    cfg_j        = cfg_fine;
    cfg_j.vareps = eps_vals(j);
    r_j          = cfg_j.pipeline(cfg_j, prob_fine);
    ke_j         = compute_objective(r_j.rho, r_j.mx, prob_fine) / prob_fine.dx;
    plot(prob_fine.xx, r_j.rho(k_half,:), '-', 'Color', colors_eps(j,:), ...
         'LineWidth', 1.5, ...
         'DisplayName', sprintf('$\\varepsilon=%.0e$  KE=%.4f', eps_vals(j), ke_j));
end

xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho(t=0.5,x)$', 'Interpreter', 'latex');
title(['Bimodal $\to$ unimodal at $t=0.5$: $\varepsilon$ sweep  ' ...
       sprintf('($n_t=%d$, $n_x=%d$)', nt_vals(i_ref), nx_vals(i_ref))], ...
      'Interpreter', 'latex');
legend('Interpreter', 'latex', 'Location', 'best', 'FontSize', 9, 'NumColumns', 2);
grid on;

%% --- Summary ---
fprintf('\n%-12s  %-8s  %-12s  %-10s\n', 'nt x nx', 'iters', 'KE', 'converged');
fprintf('%s\n', repmat('-', 1, 48));
for k = 1:n_grids
    fprintf('%3d x %-6d  %-8d  %-12.6f  %d\n', ...
            nt_vals(k), nx_vals(k), results{k}.iters, ke_vals(k), results{k}.converged);
end
fprintf('\nFine-grid KE (reference) = %.6f\n', ke_vals(i_ref));

%% --- Save all figures ---
fig_dir = fullfile(base, '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

test_tag = 'sb_bimodal';
cfg_tag  = sprintf('nt%d_nx%d_eps%g', nt_vals(i_ref), nx_vals(i_ref), vareps);
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
