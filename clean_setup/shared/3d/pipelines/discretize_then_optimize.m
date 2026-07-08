function result = discretize_then_optimize(cfg, problem)
% DISCRETIZE_THEN_OPTIMIZE  Linearized ADMM pipeline for 3D OT / SB.
%
%   result = discretize_then_optimize(cfg, problem)
%
%   Solves:
%       min   I_FP(x) + KE(y)   s.t.   A*x - y = 0
%
%   Variable grids:
%     x.rho  (ntm x nx  x ny  x nz )  staggered density
%     x.mx   (nt  x nxm x ny  x nz )  staggered x-momentum
%     x.my   (nt  x nx  x nym x nz )  staggered y-momentum
%     x.mz   (nt  x nx  x ny  x nzm)  staggered z-momentum
%     y.rho  (nt  x nx  x ny  x nz )  cell-centre density
%     y.mx   (nt  x nx  x ny  x nz )  cell-centre x-momentum
%     y.my   (nt  x nx  x ny  x nz )  cell-centre y-momentum
%     y.mz   (nt  x nx  x ny  x nz )  cell-centre z-momentum

    dt   = problem.dt;
    dx   = problem.dx;
    dy   = problem.dy;
    dz   = problem.dz;
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;   nxm = nx - 1;
    ny   = problem.ny;   nym = ny - 1;
    nz   = problem.nz;   nzm = nz - 1;
    rho0 = problem.rho0;   % (nx x ny x nz)
    rho1 = problem.rho1;
    ops  = problem.ops;

    % Precompute ETD projection factors
    if isequal(cfg.projection, @proj_fokker_planck_expsemi_gpu)
        problem.expsemi_proj = precomp_expsemi_proj(problem, cfg.vareps);
    end

    % --- GPU setup ---
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
        if isequal(cfg.projection, @proj_fokker_planck_expsemi_gpu)
            ep = problem.expsemi_proj;
            ep.lower_all = gpuArray(ep.lower_all);
            ep.main_all  = gpuArray(ep.main_all);
            ep.upper_all = gpuArray(ep.upper_all);
            ep.c_vals    = gpuArray(ep.c_vals);
            ep.phi_vals  = gpuArray(ep.phi_vals);
            ep.tw_x      = gpuArray(ep.tw_x);
            ep.w_x       = gpuArray(ep.w_x);
            ep.itw_x     = gpuArray(ep.itw_x);
            ep.tw_y      = gpuArray(ep.tw_y);
            ep.w_y       = gpuArray(ep.w_y);
            ep.itw_y     = gpuArray(ep.itw_y);
            ep.tw_z      = gpuArray(ep.tw_z);
            ep.w_z       = gpuArray(ep.w_z);
            ep.itw_z     = gpuArray(ep.itw_z);
            problem.expsemi_proj = ep;
        end
    end

    % --- Initial guesses ---
    t_stag  = reshape(linspace(0, 1, ntm)', ntm, 1, 1, 1);
    rho0_4d = reshape(rho0, 1, nx, ny, nz);
    rho1_4d = reshape(rho1, 1, nx, ny, nz);
    x0.rho = (1 - t_stag) .* rho0_4d + t_stag .* rho1_4d;
    x0.mx  = zeros(nt, nxm, ny, nz);
    x0.my  = zeros(nt, nx,  nym, nz);
    x0.mz  = zeros(nt, nx,  ny,  nzm);

    t_cc    = reshape(((1:nt)' - 0.5) * dt, nt, 1, 1, 1);
    y0.rho = (1 - t_cc) .* rho0_4d + t_cc .* rho1_4d;
    y0.mx  = zeros(nt, nx, ny, nz);
    y0.my  = zeros(nt, nx, ny, nz);
    y0.mz  = zeros(nt, nx, ny, nz);

    if use_gpu
        x0 = structfun(@gpuArray, x0, 'UniformOutput', false);
        y0 = structfun(@gpuArray, y0, 'UniformOutput', false);
    end

    b = s_zeros(y0);

    % --- Operators ---
    gamma   = cfg.gamma;
    sigma   = 1 / gamma;
    zeros_x = zeros(nt, ny, nz);   % x-wall BCs
    zeros_y = zeros(nt, nx, nz);   % y-wall BCs
    zeros_z = zeros(nt, nx, ny);   % z-wall BCs
    if use_gpu
        zeros_x = gpuArray(zeros_x);
        zeros_y = gpuArray(zeros_y);
        zeros_z = gpuArray(zeros_z);
    end

    A_fn = @(x) struct( ...
        'rho', ops.interp_t_at_phi(x.rho, rho0, rho1), ...
        'mx',  ops.interp_x_at_phi(x.mx,  zeros_x, zeros_x), ...
        'my',  ops.interp_y_at_phi(x.my,  zeros_y, zeros_y), ...
        'mz',  ops.interp_z_at_phi(x.mz,  zeros_z, zeros_z));

    At_fn = @(v) struct( ...
        'rho', ops.interp_t_at_rho(v.rho), ...
        'mx',  ops.interp_x_at_m(v.mx), ...
        'my',  ops.interp_y_at_m(v.my), ...
        'mz',  ops.interp_z_at_m(v.mz));

    B_fn  = @(y) s_scale(-1, y);

    prox_f1 = @(v, step) cfg.projection(v, problem, cfg);

    solve_y = @(delta, z_hat) cfg.prox_ke( ...
        s_sub(z_hat, s_scale(sigma, delta)), sigma, problem);

    norm_fn = @(v) sqrt(dt * dx * dy * dz * ...
        (sum(v.rho(:).^2) + sum(v.mx(:).^2) + sum(v.my(:).^2) + sum(v.mz(:).^2)));

    % --- ADMM options ---
    admm_opts.gamma    = gamma;
    admm_opts.tau      = cfg.tau;
    admm_opts.alpha    = get_alpha(cfg);
    admm_opts.max_iter = cfg.max_iter;
    admm_opts.tol      = cfg.tol;
    admm_opts.norm_fn  = norm_fn;
    admm_opts.use_gpu  = use_gpu;
    if isfield(cfg, 'print_every')
        admm_opts.print_every = cfg.print_every;
    end

    % --- Solve ---
    if use_gpu
        gpu_info_pre  = gpuDevice();
        mem_pre_bytes = gpu_info_pre.TotalMemory - gpu_info_pre.AvailableMemory;
    end

    [x, y, ~, info] = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, admm_opts);

    if use_gpu
        gpu_info_post  = gpuDevice();
        mem_post_bytes = gpu_info_post.TotalMemory - gpu_info_post.AvailableMemory;
    end

    if use_gpu
        x.rho = gather(x.rho);  x.mx = gather(x.mx);
        x.my  = gather(x.my);   x.mz = gather(x.mz);
        y.rho = gather(y.rho);  y.mx = gather(y.mx);
        y.my  = gather(y.my);   y.mz = gather(y.mz);
    end

    result.rho_stag  = x.rho;
    result.mx_stag   = x.mx;
    result.my_stag   = x.my;
    result.mz_stag   = x.mz;
    result.rho_cc    = y.rho;
    result.mx_cc     = y.mx;
    result.my_cc     = y.my;
    result.mz_cc     = y.mz;
    result.res_x      = info.res_x;
    result.res_y      = info.res_y;
    result.res_primal = info.res_primal;
    result.residual   = info.res_x;
    result.iters      = info.iters;
    result.converged  = info.converged;
    result.error      = info.res_x(end);
    result.walltime   = info.walltime;
    result.iter_times = info.iter_times;
    result.cfg        = cfg;

    result.time_per_iter = info.walltime / info.iters;
    result.throughput    = info.iters / info.walltime;
    result.N_cells       = nt * nx * ny * nz;
    result.time_per_iter_per_cell = info.walltime / (info.iters * nt * nx * ny * nz);

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
