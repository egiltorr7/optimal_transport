function result = discretize_then_optimize(cfg, problem)
% DISCRETIZE_THEN_OPTIMIZE  Linearized ADMM pipeline for 2D OT / SB.
%
%   result = discretize_then_optimize(cfg, problem)
%
%   Solves:
%       min   I_FP(x) + KE(y)   s.t.   A*x - y = 0
%
%   using ladmm_solve (Fang et al. 2015) with:
%
%     x   = (rho, mx, my) on the staggered grid  -- FP constraint variable
%     y   = (rho, mx, my) on the cell-centre grid -- KE variable
%     A   = affine interpolation staggered -> cell-centres (BCs baked in)
%     B   = -I  (on cell-centres)
%     b   = 0   (BCs absorbed into affine A)
%
%   Variable grids:
%     x.rho  (ntm x nx  x ny)   staggered density
%     x.mx   (nt  x nxm x ny)   staggered x-momentum
%     x.my   (nt  x nx  x nym)  staggered y-momentum
%     y.rho  (nt  x nx  x ny)   cell-centre density
%     y.mx   (nt  x nx  x ny)   cell-centre x-momentum
%     y.my   (nt  x nx  x ny)   cell-centre y-momentum
%
%   A is affine: A_fn(x).rho = interp_t_at_phi(x.rho, rho0, rho1)
%                A_fn(x).mx  = interp_x_at_phi(x.mx,  0,    0   )
%                A_fn(x).my  = interp_y_at_phi(x.my,  0,    0   )
%
%   At_fn is the adjoint of the LINEAR part of A (no BC correction):
%                At_fn(v).rho = interp_t_at_rho(v.rho)
%                At_fn(v).mx  = interp_x_at_m(v.mx)
%                At_fn(v).my  = interp_y_at_m(v.my)
%
%   x-step (linearised):
%     r = A_fn(x^t) - y^t
%     g = At_fn( gamma*r - delta^t )
%     x^{t+1} = proj_FP( x^t - (1/tau)*g )
%
%   y-step (exact):
%     y^{t+1} = prox_KE( z_hat - delta^t/gamma,  1/gamma )
%
%   Convergence requires tau > gamma * ||A_linear||^2 <= gamma.

    dt   = problem.dt;
    dx   = problem.dx;
    dy   = problem.dy;
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;   nxm = nx - 1;
    ny   = problem.ny;   nym = ny - 1;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    ops  = problem.ops;

    % Precompute banded FP projection factors
    if isequal(cfg.projection, @proj_fokker_planck_banded)
        problem.banded_proj = precomp_banded_proj(problem, cfg.vareps);
    elseif isequal(cfg.projection, @proj_fokker_planck_spike2)
        problem.banded_proj = precomp_banded_proj_spike2(problem, cfg.vareps);
    end

    % --- GPU setup (cast all persistent arrays before closures are formed) ---
    use_gpu = isfield(cfg, 'use_gpu') && cfg.use_gpu;
    if use_gpu
        gpu_id = 1;
        if isfield(cfg, 'gpu_device'), gpu_id = cfg.gpu_device; end
        gpuDevice(gpu_id);
        rho0 = gpuArray(rho0);
        rho1 = gpuArray(rho1);
        problem.rho0     = rho0;
        problem.rho1     = rho1;
        problem.lambda_t = gpuArray(problem.lambda_t);
        if isequal(cfg.projection, @proj_fokker_planck_banded)
            problem.banded_proj.lower_all = gpuArray(problem.banded_proj.lower_all);
            problem.banded_proj.main_all  = gpuArray(problem.banded_proj.main_all);
            problem.banded_proj.upper_all = gpuArray(problem.banded_proj.upper_all);
        elseif isequal(cfg.projection, @proj_fokker_planck_spike2)
            problem.banded_proj.lower_all        = gpuArray(problem.banded_proj.lower_all);
            problem.banded_proj.main_all         = gpuArray(problem.banded_proj.main_all);
            problem.banded_proj.upper_all        = gpuArray(problem.banded_proj.upper_all);
            problem.banded_proj.spike_pivots     = gpuArray(problem.banded_proj.spike_pivots);
            problem.banded_proj.spike_v          = gpuArray(problem.banded_proj.spike_v);
            problem.banded_proj.thomas_mults     = gpuArray(problem.banded_proj.thomas_mults);
            problem.banded_proj.spike_pivots_inv = gpuArray(problem.banded_proj.spike_pivots_inv);
        end
    end

    % --- Initial guesses ---
    % x on staggered grid
    t_stag  = reshape(linspace(0, 1, ntm)', ntm, 1, 1);
    rho0_3d = reshape(rho0, 1, nx, ny);
    rho1_3d = reshape(rho1, 1, nx, ny);
    x0.rho = (1 - t_stag) .* rho0_3d + t_stag .* rho1_3d;   % (ntm x nx x ny)
    x0.mx  = zeros(nt, nxm, ny);
    x0.my  = zeros(nt, nx,  nym);

    % y on cell-centre grid
    t_cc    = reshape(((1:nt)' - 0.5) * dt, nt, 1, 1);
    y0.rho = (1 - t_cc) .* rho0_3d + t_cc .* rho1_3d;        % (nt x nx x ny)
    y0.mx  = zeros(nt, nx, ny);
    y0.my  = zeros(nt, nx, ny);

    % Cast initial guesses to GPU after they are built
    if use_gpu
        x0 = structfun(@gpuArray, x0, 'UniformOutput', false);
        y0 = structfun(@gpuArray, y0, 'UniformOutput', false);
    end

    % b = 0 on cell-centre grid (BCs absorbed into affine A_fn)
    b = s_zeros(y0);

    % --- Operators ---
    gamma   = cfg.gamma;
    sigma   = 1 / gamma;
    zeros_x = zeros(nt, ny);    % x-wall BCs  (nt x ny)
    zeros_y = zeros(nt, nx);    % y-wall BCs  (nt x nx)
    if use_gpu
        zeros_x = gpuArray(zeros_x);
        zeros_y = gpuArray(zeros_y);
    end

    % A: affine interpolation staggered -> cell-centres (BCs baked in)
    A_fn  = @(x) struct('rho', ops.interp_t_at_phi(x.rho, rho0, rho1), ...
                        'mx',  ops.interp_x_at_phi(x.mx, zeros_x, zeros_x), ...
                        'my',  ops.interp_y_at_phi(x.my, zeros_y, zeros_y));

    % A^T: adjoint of the LINEAR part of A (no BC terms)
    At_fn = @(v) struct('rho', ops.interp_t_at_rho(v.rho), ...
                        'mx',  ops.interp_x_at_m(v.mx), ...
                        'my',  ops.interp_y_at_m(v.my));

    B_fn  = @(y) s_scale(-1, y);

    % prox_f1 = projection onto FP constraint (x on staggered grid)
    prox_f1 = @(v, step) cfg.projection(v, problem, cfg);

    % solve_y = prox of KE (y on cell-centre grid)
    solve_y = @(delta, z_hat) cfg.prox_ke( ...
        s_sub(z_hat, s_scale(sigma, delta)), sigma, problem);

    % Weighted L2 norm on cell-centre grid for convergence
    norm_fn = @(v) sqrt(dt * dx * dy * ...
        (sum(v.rho(:).^2) + sum(v.mx(:).^2) + sum(v.my(:).^2)));

    % --- ADMM options ---
    admm_opts.gamma    = gamma;
    admm_opts.tau      = cfg.tau;
    admm_opts.alpha    = get_alpha(cfg);
    admm_opts.max_iter = cfg.max_iter;
    admm_opts.tol      = cfg.tol;
    admm_opts.norm_fn  = norm_fn;
    admm_opts.use_gpu  = use_gpu;

    % --- Solve ---
    % Snapshot GPU memory just before solve (all arrays allocated)
    if use_gpu
        gpu_info_pre  = gpuDevice();
        mem_pre_bytes = gpu_info_pre.TotalMemory - gpu_info_pre.AvailableMemory;
    end

    [x, y, ~, info] = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, admm_opts);

    % GPU memory after solve (includes result arrays still on device)
    if use_gpu
        gpu_info_post  = gpuDevice();
        mem_post_bytes = gpu_info_post.TotalMemory - gpu_info_post.AvailableMemory;
    end

    % Gather from GPU if needed, then return
    if use_gpu
        x.rho = gather(x.rho);  x.mx = gather(x.mx);  x.my = gather(x.my);
        y.rho = gather(y.rho);  y.mx = gather(y.mx);  y.my = gather(y.my);
    end

    % Return both grids: x on staggered (FP-feasible), y on cell-centres (KE-optimal)
    result.rho_stag  = x.rho;
    result.mx_stag   = x.mx;
    result.my_stag   = x.my;
    result.rho_cc    = y.rho;
    result.mx_cc     = y.mx;
    result.my_cc     = y.my;
    result.residual   = info.residual;
    result.iters      = info.iters;
    result.converged  = info.converged;
    result.error      = info.residual(end);
    result.walltime   = info.walltime;
    result.iter_times = info.iter_times;
    result.cfg       = cfg;

    % Computational cost metrics
    result.time_per_iter = info.walltime / info.iters;          % seconds/iter (grid-normalised)
    result.throughput    = info.iters / info.walltime;           % iters/sec
    result.N_cells       = nt * nx * ny;                         % total cell count
    result.time_per_iter_per_cell = info.walltime / (info.iters * nt * nx * ny);  % sec/(iter·cell)
    if use_gpu
        result.gpu_mem_pre_mb  = mem_pre_bytes  / 1e6;
        result.gpu_mem_post_mb = mem_post_bytes / 1e6;
        result.gpu_total_mb    = gpu_info_post.TotalMemory / 1e6;
    else
        result.gpu_mem_pre_mb  = 0;
        result.gpu_mem_post_mb = 0;
        result.gpu_total_mb    = 0;
    end
end

function alpha = get_alpha(cfg)
    if isfield(cfg, 'alpha')
        alpha = cfg.alpha;
    else
        alpha = 1.0;
    end
end
