% TEST_TIME_REVERSAL  Time-reversal symmetry check (spectral ADMM).
%
%   For SB, swapping (rho0, rho1) should give the time-reversed solution:
%   rho_reversed(t, x) = rho_forward(1-t, x).
%   Checks that the solver satisfies this symmetry.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Config ---
cfg      = cfg_admm_spectral_fe();
prob_def = prob_gaussian();

%% --- Forward run ---
problem_fwd = setup_problem_spectral(cfg, prob_def);
fprintf('Running forward...\n');
r_fwd = cfg.pipeline(cfg, problem_fwd);
fprintf('  iters=%d  error=%.2e\n', r_fwd.iters, r_fwd.error);

%% --- Reversed run: swap rho0 <-> rho1 ---
prob_rev          = prob_def;
prob_rev.rho0_func = prob_def.rho1_func;
prob_rev.rho1_func = prob_def.rho0_func;
problem_rev = setup_problem_spectral(cfg, prob_rev);
fprintf('Running reversed...\n');
r_rev = cfg.pipeline(cfg, problem_rev);
fprintf('  iters=%d  error=%.2e\n', r_rev.iters, r_rev.error);

ftag = sprintf('nt%d_nx%d_gam%g_eps%g', cfg.nt, cfg.nx, cfg.gamma, cfg.vareps);
nt = problem_fwd.nt;  nx = problem_fwd.nx;  xx = problem_fwd.xx;  dt = problem_fwd.dt;

%% --- Compare: r_fwd.rho(t) vs flipud(r_rev.rho)(t) ---
rho_fwd     = r_fwd.rho;
rho_rev_flipped = flipud(r_rev.rho);   % flip time axis
symmetry_err = sqrt(problem_fwd.dx * mean(sum((rho_fwd - rho_rev_flipped).^2, 2)));
fprintf('Time-reversal symmetry error: %.2e  (should be ~0)\n', symmetry_err);

%% --- Plot comparison at mid-time ---
k_mid = round(nt / 2);
stride = max(1, floor(nx / 60));  idx = 1:stride:nx;

figure('Position', [100 100 700 350]);
plot(xx, rho_fwd(k_mid,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'forward'); hold on;
plot(xx(idx), rho_rev_flipped(k_mid,idx), 'ro', 'MarkerSize', 5, ...
    'MarkerFaceColor', 'r', 'DisplayName', 'reversed (flipped)');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\rho$', 'Interpreter', 'latex');
title(sprintf('Time reversal at $t=%.2f$   err=%.2e', (k_mid-0.5)*dt, symmetry_err), ...
    'Interpreter', 'latex');
legend('Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, sprintf('time_reversal_%s.png', ftag)));
