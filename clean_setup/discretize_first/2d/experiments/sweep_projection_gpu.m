% SWEEP_PROJECTION_GPU  Scaling study: expsemi_gpu (batched Thomas) vs
%   expsemi_backslash_gpu (per-mode sparse), on two independent axes.
%
%   M = nx*ny (the batch/mode dimension) and nt (the sequential recursion
%   depth) test different mechanisms and are swept ONE AT A TIME, not
%   together, so a change in the timing ratio can be attributed to one
%   cause instead of conflating both:
%
%     Sweep A: fix NT, vary (NX,NY) -- tests whether M is "nearly free" for
%              the batched solver vs ~linear-in-M for the per-mode solver.
%     Sweep B: fix (NX,NY), vary NT -- tests the sequential-depth floor;
%              expect BOTH variants to scale ~linearly in nt (batching
%              fixes the M axis, not the nt axis).
%
%   Keep M small for expsemi_backslash_gpu -- it's nx*ny sequential
%   sparse-gpuArray solves of tiny systems, and per-call kernel-launch
%   overhead dominates. NXY_LIST/NXY_FIXED below start modest; scale up
%   once you've confirmed a run completes in reasonable time.
%
%   To check robustness (recommended before trusting either scaling law):
%   call run_sweep again with a different NT_FIXED (Sweep A) or a different
%   NXY_FIXED (Sweep B) and confirm the same qualitative shape holds.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

gpuDevice(1);

VARIANTS = {
    'expsemi_gpu',           @proj_fokker_planck_expsemi_gpu
    'expsemi_backslash_gpu', @proj_fokker_planck_expsemi_backslash_gpu
};

vareps = 1e-8;
MU0 = [0.3, 0.3]; MU1 = [0.7, 0.7]; SIGMA = 0.05;
Normal2d = @(xx, yy, mux, muy, sig) ...
    exp(-((xx - mux).^2 + (yy - muy).^2) / (2*sig^2)) / (2*pi*sig^2);
prob_def = prob_gaussian();
prob_def.rho0_func = @(xx, yy) Normal2d(xx, yy, MU0(1), MU0(2), SIGMA);
prob_def.rho1_func = @(xx, yy) Normal2d(xx, yy, MU1(1), MU1(2), SIGMA);

cfg_base = cfg_ladmm_gaussian();

%% --- Sweep A: fix NT, vary (NX,NY) ---
NT_FIXED = 32;
NXY_LIST = [8, 16, 32, 64];   % raise once you've timed a full pass at these

fprintf('=== Sweep A: fixed nt=%d, varying nx=ny ===\n', NT_FIXED);
fprintf('%8s  %10s  %16s  %20s  %8s\n', 'nx=ny', 'M=nx*ny', 'expsemi_gpu(ms)', 'backslash_gpu(ms)', 'ratio');
resultsA = run_sweep(VARIANTS, cfg_base, prob_def, vareps, NT_FIXED, NXY_LIST, true);

%% --- Sweep B: fix (NX,NY), vary NT ---
NXY_FIXED = 32;
NT_LIST = [8, 16, 32, 64, 128];

fprintf('\n=== Sweep B: fixed nx=ny=%d, varying nt ===\n', NXY_FIXED);
fprintf('%8s  %10s  %16s  %20s  %8s\n', 'nt', 'M=nx*ny', 'expsemi_gpu(ms)', 'backslash_gpu(ms)', 'ratio');
resultsB = run_sweep(VARIANTS, cfg_base, prob_def, vareps, NT_LIST, NXY_FIXED, false);

%% ---------------------------------------------------------------------
function results = run_sweep(variants, cfg_base, prob_def, vareps, nt_arg, nxy_arg, vary_nxy)
% RUN_SWEEP  Times VARIANTS across one swept axis, holding the other fixed.
%   vary_nxy = true:  nt_arg is a scalar (fixed nt), nxy_arg is a list.
%   vary_nxy = false: nxy_arg is a scalar (fixed nx=ny), nt_arg is a list.

    if vary_nxy
        n_pts = numel(nxy_arg);
    else
        n_pts = numel(nt_arg);
    end

    n_var = size(variants, 1);
    results.nt = zeros(n_pts, 1);
    results.M  = zeros(n_pts, 1);
    results.t  = zeros(n_pts, n_var);

    for p = 1:n_pts
        if vary_nxy
            nt = nt_arg; nxy = nxy_arg(p);
        else
            nt = nt_arg(p); nxy = nxy_arg;
        end

        cfg = cfg_base;
        cfg.nt = nt; cfg.nx = nxy; cfg.ny = nxy;
        cfg.vareps = vareps;

        problem = setup_problem(cfg, prob_def);
        problem.expsemi_proj = precomp_expsemi_proj_block(problem, vareps);

        % GPU-cast once per grid size, outside the timed calls (fair timing:
        % see bench_projection_gpu.m for why this matters).
        problem.rho0     = gpuArray(problem.rho0);
        problem.rho1     = gpuArray(problem.rho1);
        problem.lambda_t = gpuArray(problem.lambda_t);
        ep = problem.expsemi_proj;
        epf = fieldnames(ep);
        for i = 1:numel(epf)
            ep.(epf{i}) = gpuArray(ep.(epf{i}));
        end
        problem.expsemi_proj = ep;

        nxm = nxy - 1; nym = nxy - 1; ntm = nt - 1;
        t_stag  = reshape(linspace(0, 1, ntm)', ntm, 1, 1);
        rho0_3d = reshape(problem.rho0, 1, nxy, nxy);
        rho1_3d = reshape(problem.rho1, 1, nxy, nxy);

        x_in.rho = gpuArray((1 - t_stag) .* rho0_3d + t_stag .* rho1_3d);
        x_in.mx  = gpuArray.zeros(nt, nxm, nxy);
        x_in.my  = gpuArray.zeros(nt, nxy,  nym);

        row_times = zeros(1, n_var);
        for k = 1:n_var
            fn = variants{k, 2};
            row_times(k) = gputimeit(@() fn(x_in, problem, cfg));
        end

        results.nt(p)   = nt;
        results.M(p)    = nxy * nxy;
        results.t(p, :) = row_times;

        ratio = row_times(2) / row_times(1);   % backslash_gpu / expsemi_gpu
        if vary_nxy
            axis_val = nxy;
        else
            axis_val = nt;
        end
        fprintf('%8d  %10d  %16.4f  %20.4f  %8.2f\n', ...
            axis_val, nxy*nxy, row_times(1)*1e3, row_times(2)*1e3, ratio);
    end
end
