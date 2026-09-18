% SWEEP_GAMMA_TAU
%
% Sweeps (gamma, tau/gamma ratio) for the LADMM solver on a small 2D grid
% to empirically identify good step-size choices.  CPU-only, no GPU needed.
%
% Grid: nt=8, nx=16, ny=16  (fast enough to run all combinations on laptop)
%
% For each (gamma, c=tau/gamma, eps) combination the solver runs for a fixed
% budget of max_iter iterations and records the final values of all three
% ADMM residuals:
%   res_x      = ||x^{k+1} - x^k||   (convergence criterion)
%   res_y      = ||y^{k+1} - y^k||
%   res_primal = ||Ax - y||           (primal feasibility)
%
% Outputs: heatmaps of log10(final residual) over the (gamma, c) grid,
%          one figure per eps value, plus a console summary table.

clear; clc;

base = fileparts(mfilename('fullpath'));
run(fullfile(base, '..', 'setup_paths.m'));

rng(42);

%% --- Problem: small 2D Gaussian SB, CPU grid ---
nt = 8;   nx = 16;  ny = 16;
dt = 1/nt;  dx = 1/nx;  dy = 1/ny;
ntm = nt-1;  nxm = nx-1;  nym = ny-1;

xx = ((1:nx)' - 0.5) * dx;
yy = ((1:ny)  - 0.5) * dy;
G  = @(x, mu, s) exp(-0.5*((x - mu)/s).^2);
rho0 = G(xx, 0.3, 0.08) .* G(yy, 0.5, 0.10);
rho1 = G(xx, 0.7, 0.08) .* G(yy, 0.5, 0.10);
rho0 = rho0 / sum(rho0(:));
rho1 = rho1 / sum(rho1(:));

problem.nt  = nt;   problem.nx  = nx;   problem.ny  = ny;
problem.dt  = dt;   problem.dx  = dx;   problem.dy  = dy;
problem.rho0 = rho0;  problem.rho1 = rho1;
problem.lambda_t = (2 - 2*cos(pi*dt*(0:ntm)')) / dt^2;
problem.lambda_x = (2 - 2*cos(pi*dx*(0:nxm)  )) / dx^2;
problem.lambda_y = (2 - 2*cos(pi*dy*(0:nym)   )) / dy^2;
problem.ops = disc_staggered_1st(problem);
problem.xx  = xx;
problem.yy  = yy;

%% --- Sweep parameters ---
gamma_vals = logspace(-2, 1, 7);          % [0.01, 0.046, 0.215, 1.0, 4.64, 10]
c_vals     = [1.05, 1.2, 1.5, 2.0, 5.0, 10.0];   % tau = c * gamma
eps_vals   = [0.1, 1.0, 10.0];

n_g   = numel(gamma_vals);
n_c   = numel(c_vals);
n_eps = numel(eps_vals);

max_iter = 2000;
tol      = 1e-10;   % tight so runs go to full budget in most cases

%% --- Storage ---
res_x_fin      = nan(n_g, n_c, n_eps);
res_y_fin      = nan(n_g, n_c, n_eps);
res_primal_fin = nan(n_g, n_c, n_eps);
iters_done     = nan(n_g, n_c, n_eps);
converged_mat  = false(n_g, n_c, n_eps);
walltime_mat   = nan(n_g, n_c, n_eps);

n_total = n_g * n_c * n_eps;
n_done  = 0;
t_sweep_start = tic;

%% --- Sweep ---
for ie = 1:n_eps
    vareps = eps_vals(ie);
    fprintf('\n=== eps = %.2g  (%d/%d) ===\n', vareps, ie, n_eps);
    fprintf('%-8s  %-6s  %-12s  %-12s  %-12s  %-6s  %-8s\n', ...
        'gamma', 'c', 'res_x', 'res_y', 'primal', 'iters', 'wall(s)');
    fprintf('%s\n', repmat('-', 1, 72));

    for ig = 1:n_g
        for ic = 1:n_c
            gamma = gamma_vals(ig);
            tau   = c_vals(ic) * gamma;

            cfg.name        = 'sweep';
            cfg.pipeline    = @discretize_then_optimize;
            cfg.disc        = @disc_staggered_1st;
            cfg.prox_ke     = @prox_ke_cc;
            cfg.projection  = @proj_fokker_planck_spike2;
            cfg.gamma       = gamma;
            cfg.tau         = tau;
            cfg.alpha       = 1.0;
            cfg.vareps      = vareps;
            cfg.max_iter    = max_iter;
            cfg.tol         = tol;
            cfg.print_every = 0;      % silent: we print our own summary
            cfg.use_gpu     = false;  % CPU only

            result = discretize_then_optimize(cfg, problem);

            res_x_fin(ig, ic, ie)      = result.res_x(end);
            res_y_fin(ig, ic, ie)      = result.res_y(end);
            res_primal_fin(ig, ic, ie) = result.res_primal(end);
            iters_done(ig, ic, ie)     = result.iters;
            converged_mat(ig, ic, ie)  = result.converged;
            walltime_mat(ig, ic, ie)   = result.walltime;

            n_done = n_done + 1;
            elapsed = toc(t_sweep_start);
            eta     = elapsed / n_done * (n_total - n_done);

            conv_str = '';
            if result.converged, conv_str = ' *'; end
            fprintf('%-8.3g  %-6.2f  %-12.2e  %-12.2e  %-12.2e  %-6d  %-8.2f  ETA %.0fs%s\n', ...
                gamma, c_vals(ic), result.res_x(end), result.res_y(end), ...
                result.res_primal(end), result.iters, result.walltime, eta, conv_str);
        end
    end
end

fprintf('\nTotal sweep time: %.1f s\n', toc(t_sweep_start));

%% --- Save results ---
save_path = fullfile(base, 'sweep_gamma_tau_results.mat');
save(save_path, 'res_x_fin', 'res_y_fin', 'res_primal_fin', ...
    'iters_done', 'converged_mat', 'walltime_mat', ...
    'gamma_vals', 'c_vals', 'eps_vals', 'nt', 'nx', 'ny', 'max_iter', 'tol');
fprintf('Results saved to %s\n', save_path);

%% --- Figures ---
gamma_labels = arrayfun(@(g) sprintf('%.2g', g), gamma_vals, 'UniformOutput', false);
c_labels     = arrayfun(@(c) sprintf('%.2g', c), c_vals,     'UniformOutput', false);

residual_names  = {'res\_x', 'res\_y', 'primal'};
residual_data   = {res_x_fin, res_y_fin, res_primal_fin};

for ie = 1:n_eps
    vareps = eps_vals(ie);
    fig = figure('Name', sprintf('eps=%.2g', vareps), ...
                 'Position', [50 50 1100 320]);

    for ir = 1:3
        Z = log10(residual_data{ir}(:, :, ie));   % (n_g x n_c)

        subplot(1, 3, ir);
        imagesc(Z);
        colorbar;
        xlabel('c = \tau/\gamma');
        ylabel('\gamma');
        set(gca, 'XTick', 1:n_c, 'XTickLabel', c_labels, ...
                 'YTick', 1:n_g, 'YTickLabel', gamma_labels);
        title(sprintf('log_{10}(%s) at iter %d', residual_names{ir}, max_iter));

        % Mark converged cells
        hold on;
        [row, col] = find(converged_mat(:, :, ie));
        for k = 1:numel(row)
            text(col(k), row(k), '\checkmark', ...
                'HorizontalAlignment', 'center', 'Color', 'w', 'FontSize', 9);
        end
    end

    sgtitle(sprintf('\\epsilon = %.2g   (nt=%d, nx=%d, ny=%d,  max\\_iter=%d)', ...
        vareps, nt, nx, ny, max_iter));

    fig_path = fullfile(base, sprintf('sweep_gamma_tau_eps%.2g.png', vareps));
    exportgraphics(fig, fig_path, 'Resolution', 150);
    fprintf('Figure saved: %s\n', fig_path);
end
