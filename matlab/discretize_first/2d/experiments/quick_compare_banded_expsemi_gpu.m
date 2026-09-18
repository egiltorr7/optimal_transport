% QUICK_COMPARE_BANDED_EXPSEMI_GPU  GPU version of the CPU sanity check.
%
%   Runs banded vs vanilla expsemi (proj_fokker_planck_expsemi, NOT the
%   _gpu-optimized variant) for a modest number of ADMM iterations at
%   eps=1e-8, comparing res_x trajectories and final rho directly against
%   each other (not against the analytical reference).
%
%   On CPU, this comparison showed banded/expsemi agreeing to ~1e-12
%   relative at EVERY checkpoint from iter 1 through 1000. If this GPU run
%   shows a divergence -- even a small one, even before iteration 1000 --
%   that isolates the discrepancy to GPU execution specifically.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

MU0 = [0.3, 0.3]; MU1 = [0.7, 0.7]; SIGMA = 0.05;
vareps = 1e-8;

Normal2d = @(xx, yy, mux, muy, sig) ...
    exp(-((xx - mux).^2 + (yy - muy).^2) / (2*sig^2)) / (2*pi*sig^2);

prob_def = prob_gaussian();
prob_def.rho0_func = @(xx, yy) Normal2d(xx, yy, MU0(1), MU0(2), SIGMA);
prob_def.rho1_func = @(xx, yy) Normal2d(xx, yy, MU1(1), MU1(2), SIGMA);

NT = 64; NX = 128; NY = 128;
MAX_ITER = 1000;

results = struct();
projs = {'banded', @proj_fokker_planck_banded; 'expsemi', @proj_fokker_planck_expsemi};

for k = 1:size(projs,1)
    name = projs{k,1};
    cfg = cfg_ladmm_gaussian();
    cfg.projection = projs{k,2};
    cfg.nt = NT; cfg.nx = NX; cfg.ny = NY;
    cfg.vareps = vareps;
    cfg.gamma = 0.1;
    cfg.tau   = 0.11;
    cfg.use_gpu = true;
    cfg.gpu_device = 1;
    cfg.max_iter = MAX_ITER;
    cfg.tol = 1e-8;
    cfg.print_every = 200;

    problem = setup_problem(cfg, prob_def);
    fprintf('\n=== %s (GPU) ===\n', name);
    t0 = tic;
    res = discretize_then_optimize(cfg, problem);
    wall = toc(t0);
    fprintf('%s: iters=%d wall=%.1fs final res_x=%.3e res_primal=%.3e\n', ...
        name, res.iters, wall, res.res_x(end), res.res_primal(end));
    results.(name).res = res;
    results.(name).problem = problem;
end

%% --- Direct banded-vs-expsemi comparison (not vs analytical) ---
rb = results.banded.res;
re = results.expsemi.res;

n = min(numel(rb.res_x), numel(re.res_x));
fprintf('\n=== res_x trajectory comparison (banded vs expsemi), GPU ===\n');
checkpoints = [1, 10, 25, 50, 100, 200, 400, 600, 800, n];
checkpoints = unique(min(checkpoints, n));
fprintf('%6s  %14s  %14s  %10s\n', 'iter', 'res_x(banded)', 'res_x(expsemi)', 'ratio');
for c = checkpoints
    fprintf('%6d  %14.6e  %14.6e  %10.4f\n', c, rb.res_x(c), re.res_x(c), re.res_x(c)/rb.res_x(c));
end

fprintf('\n=== Direct output comparison at iter %d ===\n', n);
dV = problem.dt * problem.dx * problem.dy;
diff_rho = sqrt(dV * sum((results.banded.res.rho_stag(:) - results.expsemi.res.rho_stag(:)).^2));
nrm_rho  = sqrt(dV * sum(results.banded.res.rho_stag(:).^2));
fprintf('||rho_banded - rho_expsemi||:        %.3e\n', diff_rho);
fprintf('||rho_banded||:                       %.3e\n', nrm_rho);
fprintf('relative diff (rho, banded vs expsemi): %.3e\n', diff_rho/nrm_rho);
