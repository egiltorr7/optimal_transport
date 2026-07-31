% SWEEP_PROJECTION_GPU  Scaling study: expsemi_gpu (batched Thomas) vs
%   expsemi_backslash_block_gpu (one big block-diagonal sparse solve), on
%   two independent axes, with per-variant GPU memory footprint reported.
%
%   M = nx*ny (the batch/mode dimension) and nt (the sequential recursion
%   depth) test different mechanisms and are swept ONE AT A TIME, not
%   together, so a change in the timing ratio can be attributed to one
%   cause instead of conflating both:
%
%     Sweep A: fix NT, vary (NX,NY) -- tests whether M is "nearly free" for
%              the batched solver vs ~linear-in-M for the per-mode solver.
%     Sweep B: fix (NX,NY), vary NT -- tests the sequential-depth floor.
%
%   The two variants need DIFFERENT, non-overlapping precomputed fields
%   (Thomas needs lower_T/main_T/upper_T/main_T_mod; the block method needs
%   T_block instead), so each VARIANTS row tags which precompute it needs,
%   and this script builds ONE problem struct per distinct tag actually in
%   use -- not a single shared struct carrying both (that was the earlier,
%   wasteful design: precomp_expsemi_proj_block used to call
%   precomp_expsemi_proj first and inherit all its unused fields, roughly
%   doubling GPU memory for no benefit). See precomp_expsemi_proj_block.m.
%
%   Memory reporting: gpuDevice().AvailableMemory is snapshotted immediately
%   before and after each tag's precompute+cast step (with wait(gpuDevice())
%   in between to make sure all GPU work has actually finished), so the
%   delta reflects that tag's real persistent GPU footprint. The GPU is
%   reset (reset(gpuDevice(GPU_IDX))) at the start of EVERY sweep point,
%   before any allocation for that point -- both so memory deltas start
%   from a clean baseline every time, and so a long sweep spanning many
%   different array sizes doesn't accumulate fragmented, stale memory from
%   earlier iterations. MATLAB does not proactively return freed gpuArray
%   memory to the driver (it pools it for reuse), so without this reset,
%   AvailableMemory readings across iterations would not be reliable.
%
%   To check robustness (recommended before trusting either scaling law):
%   call run_sweep again with a different NT_FIXED (Sweep A) or a different
%   NXY_FIXED (Sweep B) and confirm the same qualitative shape holds.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));


GPU_IDX = 1;   % set to whichever device index is actually idle on a shared machine
gpuDevice(GPU_IDX);

% {name, function handle, precompute tag: 'thomas' or 'block'}
VARIANTS = {
    'expsemi_gpu',                 @proj_fokker_planck_expsemi_gpu,                 'thomas'
    'expsemi_backslash_block_gpu', @proj_fokker_planck_expsemi_backslash_block_gpu, 'block'
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
NT_FIXED = 64;
NXY_LIST = [8, 16, 32, 64, 128, 256];   % raise once you've timed a full pass at these

fprintf('=== Sweep A: fixed nt=%d, varying nx=ny ===\n', NT_FIXED);
print_header(VARIANTS, 'nx=ny');
resultsA = run_sweep(VARIANTS, cfg_base, prob_def, vareps, NT_FIXED, NXY_LIST, true, GPU_IDX);

%% --- Sweep B: fix (NX,NY), vary NT ---
NXY_FIXED = 16;
NT_LIST = [8, 16, 32, 64, 128, 256, 512];

fprintf('\n=== Sweep B: fixed nx=ny=%d, varying nt ===\n', NXY_FIXED);
print_header(VARIANTS, 'nt');
resultsB = run_sweep(VARIANTS, cfg_base, prob_def, vareps, NT_LIST, NXY_FIXED, false, GPU_IDX);

%% ---------------------------------------------------------------------
function print_header(variants, axis_name)
    fprintf('%8s  %10s', axis_name, 'M=nx*ny');
    for k = 1:size(variants,1)
        label = variants{k,1};
        fprintf('  %20s  %10s', [label '(ms)'], 'mem(MB)');
        if k > 1
            fprintf('  %12s', 'rel_diff');
        end
    end
    fprintf('  %8s\n', 'ratio');
end

%% ---------------------------------------------------------------------
function results = run_sweep(variants, cfg_base, prob_def, vareps, nt_arg, nxy_arg, vary_nxy, gpu_idx)
% RUN_SWEEP  Times VARIANTS across one swept axis, holding the other fixed.
%   vary_nxy = true:  nt_arg is a scalar (fixed nt), nxy_arg is a list.
%   vary_nxy = false: nxy_arg is a scalar (fixed nx=ny), nt_arg is a list.

    if vary_nxy
        n_pts = numel(nxy_arg);
    else
        n_pts = numel(nt_arg);
    end

    n_var = size(variants, 1);
    tags  = variants(:, 3);
    uniq_tags = unique(tags, 'stable');

    results.nt      = zeros(n_pts, 1);
    results.M       = zeros(n_pts, 1);
    results.t       = zeros(n_pts, n_var);
    results.mem     = zeros(n_pts, n_var);
    results.reldiff = zeros(n_pts, n_var);   % vs variant 1; column 1 is always 0

    for p = 1:n_pts
        if vary_nxy
            nt = nt_arg; nxy = nxy_arg(p);
        else
            nt = nt_arg(p); nxy = nxy_arg;
        end

        cfg = cfg_base;
        cfg.nt = nt; cfg.nx = nxy; cfg.ny = nxy;
        cfg.vareps = vareps;

        % Flush GPU memory before building anything for this problem size --
        % clean baseline for the memory deltas below, and avoids fragmented
        % leftovers from the previous (differently-sized) iteration.
        reset(gpuDevice(gpu_idx));

        problem_base = setup_problem(cfg, prob_def);

        nxm = nxy - 1; nym = nxy - 1; ntm = nt - 1;
        t_stag  = reshape(linspace(0, 1, ntm)', ntm, 1, 1);
        rho0_3d = reshape(problem_base.rho0, 1, nxy, nxy);
        rho1_3d = reshape(problem_base.rho1, 1, nxy, nxy);
        x_in_cpu.rho = (1 - t_stag) .* rho0_3d + t_stag .* rho1_3d;
        x_in_cpu.mx  = zeros(nt, nxm, nxy);
        x_in_cpu.my  = zeros(nt, nxy,  nym);

        % --- Build ONE gpu-cast problem struct per distinct precompute tag
        %     actually used by the active variants, tracking each tag's own
        %     GPU memory footprint via a before/after AvailableMemory delta.
        problems_by_tag = struct();
        mem_by_tag      = struct();
        for t = 1:numel(uniq_tags)
            tag = uniq_tags{t};

            wait(gpuDevice());
            mem_before = gpuDevice().AvailableMemory;

            problem_tag = problem_base;
            problem_tag.rho0     = gpuArray(problem_base.rho0);
            problem_tag.rho1     = gpuArray(problem_base.rho1);
            problem_tag.lambda_t = gpuArray(problem_base.lambda_t);

            switch tag
                case 'thomas'
                    ep = precomp_expsemi_proj(problem_base, vareps);
                case 'block'
                    ep = precomp_expsemi_proj_block(problem_base, vareps);
                otherwise
                    error('run_sweep: unknown precompute tag "%s"', tag);
            end
            epf = fieldnames(ep);
            for i = 1:numel(epf)
                ep.(epf{i}) = gpuArray(ep.(epf{i}));
            end
            problem_tag.expsemi_proj = ep;

            wait(gpuDevice());
            mem_after = gpuDevice().AvailableMemory;

            problems_by_tag.(tag) = problem_tag;
            mem_by_tag.(tag)      = (mem_before - mem_after) / 1e6;   % MB
        end

        x_in.rho = gpuArray(x_in_cpu.rho);
        x_in.mx  = gpuArray(x_in_cpu.mx);
        x_in.my  = gpuArray(x_in_cpu.my);

        dV = problem_base.dt * problem_base.dx * problem_base.dy;

        row_times   = zeros(1, n_var);
        row_mem     = zeros(1, n_var);
        row_reldiff = zeros(1, n_var);
        ref_rho     = [];
        for k = 1:n_var
            fn  = variants{k, 2};
            tag = variants{k, 3};
            problem = problems_by_tag.(tag);

            row_times(k) = gputimeit(@() fn(x_in, problem, cfg));
            row_mem(k)   = mem_by_tag.(tag);

            % One plain (untimed) call per variant, purely to compare outputs
            % -- correctness cross-check independent of the timing above.
            out = fn(x_in, problem, cfg);
            out_rho = out.rho;
            if isa(out_rho, 'gpuArray')
                out_rho = gather(out_rho);
            end
            if isempty(ref_rho)
                ref_rho = out_rho;
                row_reldiff(k) = 0;
            else
                row_reldiff(k) = sqrt(dV * sum((out_rho(:) - ref_rho(:)).^2)) / ...
                                  sqrt(dV * sum(ref_rho(:).^2));
            end
        end

        results.nt(p)       = nt;
        results.M(p)        = nxy * nxy;
        results.t(p, :)     = row_times;
        results.mem(p, :)   = row_mem;
        results.reldiff(p,:) = row_reldiff;

        ratio = row_times(2) / row_times(1);   % variant 2 / variant 1
        if vary_nxy
            axis_val = nxy;
        else
            axis_val = nt;
        end

        fprintf('%8d  %10d', axis_val, nxy*nxy);
        for k = 1:n_var
            fprintf('  %20.4f  %10.2f', row_times(k)*1e3, row_mem(k));
            if k > 1
                fprintf('  %12.3e', row_reldiff(k));
            end
        end
        fprintf('  %8.2f\n', ratio);
    end
end
