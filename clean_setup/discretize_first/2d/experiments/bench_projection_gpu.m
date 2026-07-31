% BENCH_PROJECTION_GPU  Microbenchmark harness for the 2D GPU FP projection.
%
%   Isolates the projection step from the full ADMM loop so different
%   implementations can be timed against each other on realistic problem
%   sizes, with a correctness cross-check (relative L2 diff vs the first
%   variant) so a "faster" variant that's also wrong gets caught immediately.
%
%   Runs a 2x2 factorial: DEVICE (cpu/gpu) x ALGORITHM (sparse-per-mode
%   backslash / batched dense Thomas), so you can attribute a speedup to
%   "moved to GPU" vs "batched the solve" instead of conflating the two:
%
%                    CPU                   GPU
%     backslash  expsemi_backslash    expsemi_backslash_gpu
%     batched    expsemi              expsemi_gpu
%
%   Fairness notes (read before trusting a number):
%     - problem/x_in are pre-cast ONCE per device, OUTSIDE the timed loop
%       (see the two "Build ..." sections below). If a projection function
%       had to cast its inputs on every call, every timed rep would pay a
%       host<->device transfer that a real ADMM run (which casts once before
%       thousands of iterations) never pays -- that would make GPU variants
%       look artificially slow.
%     - gputimeit is used for ALL variants, not timeit, even the CPU ones.
%       Plain timeit/tic-toc on a GPU function can return before the GPU has
%       actually finished (kernel launches are asynchronous), silently timing
%       only the dispatch, not the compute. gputimeit forces the
%       synchronization. It works fine on CPU-only functions too (a no-op
%       wait), so using it uniformly keeps the numbers comparable.
%     - rel_diff is computed after gathering both outputs to CPU, so it's
%       well-defined regardless of which device produced them.
%
%   To test a new variant:
%     1. Write proj_fn(x_in, problem, cfg) -> x_out with the same signature
%        as proj_fokker_planck_expsemi_gpu (see that file's header for the
%        expected fields on problem.expsemi_proj / x_in shapes).
%     2. Add {'name', @proj_fn, 'cpu'|'gpu'} to VARIANTS below.
%     3. If it needs precomputed fields beyond what precomp_expsemi_proj
%        already provides, add them to both the CPU and GPU precompute
%        sections.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% --- Config ---
NT = 64; NX = 256; NY = 256;
vareps = 1e-8;

VARIANTS = {
    'expsemi',                @proj_fokker_planck_expsemi,                'cpu'
    'expsemi_backslash',      @proj_fokker_planck_expsemi_backslash,      'cpu'
    % 'expsemi_on_gpu',        @proj_fokker_planck_expsemi,                'gpu'
    % ^ SAME code as 'expsemi', just fed GPU-resident data: isolates the pure
    %   hardware effect from the algorithm-optimization effect in expsemi_gpu.
    %   Requires dct()/idct() (Signal Processing Toolbox) to support gpuArray
    %   in your MATLAB version -- if this errors, that itself is informative
    %   (it's exactly why proj_fokker_planck_expsemi_gpu wrote its own
    %   FFT-based dct_rows_gpu instead of using the built-in dct/idct).
    % 'expsemi_gpu',           @proj_fokker_planck_expsemi_gpu,            'gpu'
    % 'expsemi_backslash_gpu', @proj_fokker_planck_expsemi_backslash_gpu,  'gpu'
    % WARNING: expsemi_backslash_gpu loops nx*ny sparse-gpuArray solves of
    % tiny (nt x nt) systems -- kernel-launch overhead dominates. Try nx=ny=16
    % or 32 first; don't jump straight to 256x256 with it enabled.
};

%% --- Build problem (grid, ops, boundary densities) ---
MU0 = [0.3, 0.3]; MU1 = [0.7, 0.7]; SIGMA = 0.05;
Normal2d = @(xx, yy, mux, muy, sig) ...
    exp(-((xx - mux).^2 + (yy - muy).^2) / (2*sig^2)) / (2*pi*sig^2);

prob_def = prob_gaussian();
prob_def.rho0_func = @(xx, yy) Normal2d(xx, yy, MU0(1), MU0(2), SIGMA);
prob_def.rho1_func = @(xx, yy) Normal2d(xx, yy, MU1(1), MU1(2), SIGMA);

cfg = cfg_ladmm_gaussian();
cfg.nt = NT; cfg.nx = NX; cfg.ny = NY;
cfg.vareps = vareps;

problem_cpu = setup_problem(cfg, prob_def);
problem_cpu.expsemi_proj = precomp_expsemi_proj(problem_cpu, vareps);

nt = problem_cpu.nt;
nx = problem_cpu.nx; nxm = nx - 1;
ny = problem_cpu.ny; nym = ny - 1;
ntm = nt - 1;

%% --- Build x_in once (staggered grid, matches ADMM x0 shape), CPU ---
t_stag  = reshape(linspace(0, 1, ntm)', ntm, 1, 1);
rho0_3d = reshape(problem_cpu.rho0, 1, nx, ny);
rho1_3d = reshape(problem_cpu.rho1, 1, nx, ny);

x_in_cpu.rho = (1 - t_stag) .* rho0_3d + t_stag .* rho1_3d;   % (ntm x nx x ny)
x_in_cpu.mx  = zeros(nt, nxm, ny);
x_in_cpu.my  = zeros(nt, nx,  nym);

%% --- GPU-resident copies: cast ONCE, outside the timed loop ---
need_gpu = any(strcmp(VARIANTS(:,3), 'gpu'));
if need_gpu
    gpuDevice(1);

    problem_gpu = problem_cpu;
    problem_gpu.rho0     = gpuArray(problem_cpu.rho0);
    problem_gpu.rho1     = gpuArray(problem_cpu.rho1);
    problem_gpu.lambda_t = gpuArray(problem_cpu.lambda_t);

    ep = problem_cpu.expsemi_proj;
    gpu_fields = fieldnames(ep);
    for i = 1:numel(gpu_fields)
        ep.(gpu_fields{i}) = gpuArray(ep.(gpu_fields{i}));
    end
    problem_gpu.expsemi_proj = ep;

    x_in_gpu.rho = gpuArray(x_in_cpu.rho);
    x_in_gpu.mx  = gpuArray(x_in_cpu.mx);
    x_in_gpu.my  = gpuArray(x_in_cpu.my);
end

%% --- Time + cross-check each variant ---
dV  = problem_cpu.dt * problem_cpu.dx * problem_cpu.dy;
ref = [];
fprintf('Grid: nt=%d nx=%d ny=%d  (%.2fM staggered cells)\n\n', nt, nx, ny, nt*nx*ny/1e6);
fprintf('%-24s  %-4s  %10s  %14s\n', 'variant', 'dev', 'time (ms)', 'rel_diff_vs_1st');

for k = 1:size(VARIANTS,1)
    name   = VARIANTS{k,1};
    fn     = VARIANTS{k,2};
    device = VARIANTS{k,3};

    if strcmp(device, 'gpu')
        x_in    = x_in_gpu;
        problem = problem_gpu;
    else
        x_in    = x_in_cpu;
        problem = problem_cpu;
    end

    t = gputimeit(@() fn(x_in, problem, cfg));   % uniform timing + GPU sync, all variants
    out = fn(x_in, problem, cfg);                % one more call, for the correctness check
    out_rho = out.rho;
    if isa(out_rho, 'gpuArray')
        out_rho = gather(out_rho);
    end

    if isempty(ref)
        ref = out_rho;
        rel_diff = 0;
    else
        rel_diff = sqrt(dV * sum((out_rho(:) - ref(:)).^2)) / sqrt(dV * sum(ref(:).^2));
    end

    fprintf('%-24s  %-4s  %10.4f  %14.3e\n', name, device, t*1e3, rel_diff);
end
