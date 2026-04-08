% TEST_OT_UNIFORM  OT from uniform to Gaussian (spectral ADMM, eps=0).

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config and problem ---
cfg        = cfg_admm_spectral_fe();
cfg.vareps = 0;
prob_def   = prob_uniform();
problem    = setup_problem_spectral(cfg, prob_def);

%% --- Run ---
fprintf('Running %s  (nt=%d, nx=%d, eps=%.4g)...\n', cfg.name, cfg.nt, cfg.nx, cfg.vareps);
result = cfg.pipeline(cfg, problem);
fprintf('  iters=%d,  converged=%d,  error=%.2e,  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);

%% --- Analytical solution (if available) ---
try
    [rho_ana, ~] = analytical_ot_uniform(problem);
    has_ana = true;
catch
    has_ana = false;
end

ftag = sprintf('nt%d_nx%d_gam%g_eps%g', cfg.nt, cfg.nx, cfg.gamma, cfg.vareps);

%% --- Density evolution ---
t_fracs = [0.0, 0.25, 0.5, 0.75, 1.0];
nt = problem.nt;  nx = problem.nx;  xx = problem.xx;  dt = problem.dt;
n_t = numel(t_fracs);  colors = parula(n_t);

figure('Name', 'OT uniform density', 'Position', [100 100 700 400]);
hold on;
for p = 1:n_t
    k      = max(1, min(nt, round(t_fracs(p) * nt + 0.5)));
    stride = max(1, floor(nx / 60));  idx = 1:stride:nx;
    h_num  = plot(xx(idx), result.rho(k,idx), 'o', 'Color', colors(p,:), ...
                  'MarkerSize', 5, 'MarkerFaceColor', colors(p,:));
    if has_ana
        plot(xx, rho_ana(k,:), '-', 'Color', colors(p,:), 'LineWidth', 1.5);
    end
end
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title('OT uniform$\to$Gaussian   $\varepsilon=0$', 'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('density_ot_uniform_%s.png', ftag)));

%% --- Residual ---
figure('Position', [720 100 600 300]);
semilogy(result.residual, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol=%.1e', cfg.tol));
xlabel('Iteration');  ylabel('Residual');
title(sprintf('ADMM residual   iters=%d', result.iters));
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('residual_ot_uniform_%s.png', ftag)));
