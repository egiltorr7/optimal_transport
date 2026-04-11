% TEST_ANALYTICAL_MATCH
%
% Checks whether the 2D ADMM solution matches the analytical SB solution.
% Runs on a small grid so it can be executed locally without GPU.
%
% Diagnostic flow:
%   1. Evaluate the analytical solution on the staggered grid
%   2. Check its discrete FP residual  (large -> discretization error, not a code bug)
%   3. Run ADMM and compare converged rho with analytical
%   4. Print key diagnostics to narrow down the bug

clear; clc;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

% --- Small grid for local testing ---
cfg.name       = 'diag';
cfg.disc       = @disc_staggered_1st;
cfg.prox_ke    = @prox_ke_cc;
cfg.projection = @proj_fokker_planck_banded;
cfg.pipeline   = @discretize_then_optimize;
cfg.nt         = 16;
cfg.nx         = 64;
cfg.ny         = 64;
cfg.gamma      = 100;
cfg.tau        = 101;
cfg.alpha      = 1.0;
cfg.vareps     = 0.0;   % pure OT  (eps=0 -> analytical is linear interp of Gaussians)
cfg.max_iter   = 5000;
cfg.tol        = 1e-8;
cfg.use_gpu    = false;

prob_def = prob_gaussian();
problem  = setup_problem(cfg, prob_def);

nt = problem.nt;   ntm = nt - 1;   dt = problem.dt;
nx = problem.nx;   nxm = nx - 1;   dx = problem.dx;
ny = problem.ny;   nym = ny - 1;   dy = problem.dy;
ops  = problem.ops;
rho0 = problem.rho0;
rho1 = problem.rho1;

zeros_x = zeros(nt, ny);
zeros_y = zeros(nt, nx);

fprintf('Grid: nt=%d nx=%d ny=%d  eps=%.4g\n\n', nt, nx, ny, cfg.vareps);

%% --- 1. Analytical solution on staggered grid ---
[rho_ana, mx_ana, my_ana] = analytical_sb_gaussian(problem, cfg.vareps);

% --- 1a. Mass of analytical rho at each time ---
mass_ana = squeeze(sum(sum(rho_ana, 2), 3)) * dx * dy;   % (ntm x 1)
fprintf('Analytical rho mass: min=%.6f  max=%.6f  (should all be ~1)\n', ...
    min(mass_ana), max(mass_ana));

% --- 1b. FP residual of analytical solution ---
rho_phi   = ops.interp_t_at_phi(rho_ana, rho0, rho1);
nabla_rho = ops.deriv_x_at_phi(ops.deriv_x_at_m(rho_phi), zeros_x, zeros_x) ...
          + ops.deriv_y_at_phi(ops.deriv_y_at_m(rho_phi), zeros_y, zeros_y);
fp_res = ops.deriv_t_at_phi(rho_ana, rho0, rho1) ...
       + ops.deriv_x_at_phi(mx_ana, zeros_x, zeros_x) ...
       + ops.deriv_y_at_phi(my_ana, zeros_y, zeros_y) ...
       - cfg.vareps * nabla_rho;

fp_max = max(abs(fp_res(:)));
fp_l2  = sqrt(dt*dx*dy * sum(fp_res(:).^2));
fprintf('FP residual of ANALYTICAL solution:\n');
fprintf('  max |fp_res| = %.2e\n', fp_max);
fprintf('  L2  |fp_res| = %.2e\n', fp_l2);
fprintf('  (Large values mean the analytical soln does not fit the discrete constraint.)\n\n');

%% --- 2. Run ADMM ---
fprintf('Running ADMM ...\n');
result = cfg.pipeline(cfg, problem);
fprintf('  iters=%d  converged=%d  final_res=%.2e  wall=%.1fs\n\n', ...
    result.iters, result.converged, result.error, result.walltime);

%% --- 3. Compare rho ---
rho_num_cc  = result.rho_cc;
rho_ana_cc  = ops.interp_t_at_phi(rho_ana, rho0, rho1);

l2_err = sqrt(dx*dy * sum(sum((rho_num_cc - rho_ana_cc).^2, 2), 3));   % (nt x 1)
fprintf('L2 error in rho vs analytical (cell-centre):\n');
fprintf('  mean = %.2e\n', mean(l2_err));
fprintf('  max  = %.2e\n', max(l2_err));

% --- 3b. Mass conservation of numerical rho ---
mass_num = squeeze(sum(sum(rho_num_cc, 2), 3)) * dx * dy;
fprintf('\nNumerical rho_cc mass: min=%.6f  max=%.6f  (should all be ~1)\n', ...
    min(mass_num), max(mass_num));

% --- 3c. FP residual of numerical staggered solution ---
fp_res_num = ops.deriv_t_at_phi(result.rho_stag, rho0, rho1) ...
           + ops.deriv_x_at_phi(result.mx_stag, zeros_x, zeros_x) ...
           + ops.deriv_y_at_phi(result.my_stag, zeros_y, zeros_y) ...
           - cfg.vareps * (ops.deriv_x_at_phi(ops.deriv_x_at_m( ...
               ops.interp_t_at_phi(result.rho_stag, rho0, rho1)), zeros_x, zeros_x) ...
             + ops.deriv_y_at_phi(ops.deriv_y_at_m( ...
               ops.interp_t_at_phi(result.rho_stag, rho0, rho1)), zeros_y, zeros_y));

fprintf('\nFP residual of NUMERICAL staggered solution:\n');
fprintf('  max |fp_res| = %.2e\n', max(abs(fp_res_num(:))));
fprintf('  L2  |fp_res| = %.2e\n', sqrt(dt*dx*dy*sum(fp_res_num(:).^2)));

%% --- 4. Snapshot comparison at t=0.5 ---
k_mid = round(0.5 * nt);
[~, iy] = min(abs(problem.yy - 0.5));
fprintf('\nAt t ~ 0.5, x-slice at y=0.5:\n');
fprintf('  max |rho_num - rho_ana| = %.2e\n', ...
    max(abs(rho_num_cc(k_mid, :, iy) - rho_ana_cc(k_mid, :, iy))));
fprintf('  max rho_ana             = %.2e\n', max(rho_ana_cc(k_mid, :, iy)));
fprintf('  max rho_num             = %.2e\n', max(rho_num_cc(k_mid, :, iy)));
