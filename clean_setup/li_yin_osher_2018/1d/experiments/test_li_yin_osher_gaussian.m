% TEST_LI_YIN_OSHER_GAUSSIAN  Run the Fisher-info Newton solver on the
%   same Gaussian-to-Gaussian Schrodinger bridge test case used by
%   test_sb_gaussian.m, check it against the analytical SB solution, and
%   (if available) compare iteration count / wall time against the
%   existing linearized-ADMM solver on the same problem instance.
%
%   NOTE: newton_fisher_sb.m was written without a MATLAB/Octave
%   interpreter available to test it. Read the self-test output printed
%   at the top (gradient check, constraint check) before trusting
%   anything below it. If either FAILS, stop and report the printed
%   numbers back rather than debugging blind.
%
%   Run setup_paths.m first (see below), or run this script directly --
%   it adds its own paths.

clear; close all;

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, '..', 'core'));

% Also pull in the existing 1D codebase (prob_gaussian, setup_problem,
% cfg_ladmm_gaussian, discretize_then_optimize) for a shared problem
% instance and an ADMM comparison.
existing_1d = fullfile(this_dir, '..', '..', '..', 'discretize_first', '1d');
run(fullfile(existing_1d, 'setup_paths.m'));

%% --- Problem setup (same Gaussian-to-Gaussian test as test_sb_gaussian.m) ---
cfg      = cfg_ladmm_gaussian();
prob_def = prob_gaussian();

nx     = 512;
L      = 63;              % free interior time levels for the Newton solver
                           % (NOT the same thing as problem.L below, which
                           %  setup_problem.m uses for the domain LENGTH,
                           %  defaulting to 1 -- unrelated, unlucky name clash)
vareps = 0.1;                % same convention as cfg.vareps (Schrodinger bridge strength)

cfg.nx = nx;
cfg.nt = L + 1;            % match ADMM's nt to Newton's L+1 flux levels
cfg.vareps = vareps;

problem = setup_problem(cfg, prob_def);

% Floor the boundary densities away from zero before normalizing, exactly
% as Li-Yin-Osher do in their own Example 1 ("p0 = exp(...) + 0.01"): the
% Fisher-information term diverges as p -> 0, and at nx=32 the domain
% edges here sit ~6.7 sigma from either Gaussian mean, so the raw
% (unfloored) density there (~1e-11) makes the Newton gradient/Hessian
% numerically absurd (see the ~1e29 self-test values without this floor).
% Applied to BOTH rho0 and rho1, and written back into `problem` so the
% ADMM comparison below solves the exact same boundary data.
floor_val = 0.01;
rho0 = problem.rho0 + floor_val; rho0 = rho0 / sum(rho0);
rho1 = problem.rho1 + floor_val; rho1 = rho1 / sum(rho1);
problem.rho0 = rho0;
problem.rho1 = rho1;

dx   = problem.dx;
dt   = 1 / (L + 1);

fprintf('=== Problem: %s,  nx=%d, L=%d, vareps=%.4g ===\n\n', prob_def.name, nx, L, vareps);

%% --- Run Newton (Li-Yin-Osher 2018) ---
opts = struct('max_iter', 50, 'tol', 1e-5, 'alpha', 0.3, 'verbose', true, 'run_selftest', true);

fprintf('--- Newton (Fisher-information regularized) ---\n');
[p_newton, m_newton, info_newton] = newton_fisher_sb(rho0, rho1, dx, dt, L, vareps, opts);

fprintf('\nNewton: iters=%d, converged=%d, wall=%.3fs, final ||constraint||=%.3e\n\n', ...
    info_newton.iters, info_newton.converged, info_newton.walltime, info_newton.final_constraint_residual);

if ~info_newton.selftest.gradient.passed || ~info_newton.selftest.constraint.passed
    warning('Self-tests failed -- inspect info_newton.selftest before trusting p_newton/m_newton.');
end

%% --- Analytical Schrodinger bridge reference, evaluated at this solver's own grid ---
% CAVEAT: this is the closed form for PURE Gaussian boundary data, but the
% solver above is (deliberately) solving for the FLOORED boundary data
% (rho0+floor_val, rho1+floor_val). These are genuinely different problems
% -- flooring a Gaussian breaks the clean Gaussian-bridge closed form, so
% there is no simple analytical solution for the floored problem itself.
% Expect a small, floor_val-sized residual mismatch concentrated near the
% domain edges (x near 0 and 1) even for a fully-converged, bug-free
% solve; that is NOT evidence of an error. A large mismatch in the BULK
% of the domain (near the means) would be.
%
% Same closed-form Gaussian SB solution as analytical_sb_gaussian.m
% (mu_t linear interpolation, sigma_t^2 from the Ornstein-Uhlenbeck-type
% bridge variance), but evaluated directly at THIS solver's node grid
% (xx = problem.xx, nx nodes) and time levels (t_l = l*dt, l=1..L),
% rather than reusing analytical_sb_gaussian.m's staggered-grid layout.
mu0 = problem.mu0; mu1 = problem.mu1; sigma = problem.sigma;
alpha_var = sqrt(sigma^4 + vareps^2) - sigma^2;

xx = problem.xx;
p_ana = zeros(L, nx);
for l = 1:L
    t = l * dt;
    mu_t   = (1 - t) * mu0 + t * mu1;
    sig2_t = sigma^2 + 2 * alpha_var * t * (1 - t);
    row = exp(-0.5 * ((xx - mu_t) / sqrt(sig2_t)).^2) / (sqrt(2*pi*sig2_t));
    p_ana(l, :) = row / sum(row);   % normalize to match this codebase's mass convention
end

err_newton = sqrt(dx * sum((p_newton - p_ana).^2, 2));   % (L x 1), L2-in-x error per time level
fprintf('Newton vs analytical:  max L2(x) error over time = %.4e,  mean = %.4e\n\n', ...
    max(err_newton), mean(err_newton));

%% --- Comparison against existing ADMM on the same problem instance ---
fprintf('--- Existing linearized ADMM (discretize_then_optimize) for comparison ---\n');
result_admm = cfg.pipeline(cfg, problem);
fprintf('ADMM: iters=%d, converged=%d, wall=%.3fs, residual=%.3e\n\n', ...
    result_admm.iters, result_admm.converged, result_admm.walltime, result_admm.error);

%% --- Summary ---
fprintf('=== Summary (nx=%d, L=%d, vareps=%.4g) ===\n', nx, L, vareps);
fprintf('%-12s %10s %10s %14s\n', 'Method', 'iters', 'wall(s)', 'max L2 err');
fprintf('%-12s %10d %10.3f %14.4e\n', 'Newton', info_newton.iters, info_newton.walltime, max(err_newton));
fprintf('%-12s %10d %10.3f %14s\n', 'ADMM', result_admm.iters, result_admm.walltime, '(see own plot)');

%% --- Plot: density snapshots, Newton vs analytical ---
figure('Name', 'Newton (Fisher-info) vs analytical SB', 'Position', [100 100 700 400]);
n_plot = 5;
l_plot = round(linspace(1, L, n_plot));
colors = parula(n_plot);
hold on;
for k = 1:n_plot
    l = l_plot(k);
    plot(xx, p_ana(l, :), '-', 'Color', colors(k,:), 'LineWidth', 1.5);
    plot(xx, p_newton(l, :), 'o', 'Color', colors(k,:), 'MarkerSize', 4);
end
xlabel('x'); ylabel('p');
title(sprintf('Newton (Fisher-info) density, nx=%d L=%d vareps=%.3g  (solid=analytical, o=Newton)', nx, L, vareps));
grid on;

%% --- Plot: Newton objective convergence ---
figure('Name', 'Newton objective history', 'Position', [820 100 600 350]);
semilogy(abs(diff(info_newton.obj_hist)) ./ abs(info_newton.obj_hist(1:end-1)), 'b-o', 'LineWidth', 1.2);
yline(opts.tol, 'r--', sprintf('tol=%.0e', opts.tol));
xlabel('Newton iteration');
ylabel('relative objective change');
title('Newton convergence (Li-Yin-Osher 2018 stopping criterion)');
grid on;
