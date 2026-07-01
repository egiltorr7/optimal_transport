function result = discretize_then_optimize_nonuniform(cfg, problem)
% DISCRETIZE_THEN_OPTIMIZE_NONUNIFORM  Like discretize_then_optimize but with
%   a non-uniform time grid refined near t=1 via a power law.
%
%   The time grid is generated from make_nonuniform_tgrid(nt, vareps) and
%   stored in problem.dt_vec and problem.t_grid.  Everything else (ADMM
%   structure, KE prox, interpolation operators) is unchanged.

    dt   = problem.dt;   % 1/nt — used only for norm scaling
    dx   = problem.dx;
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;   nxm = nx - 1;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    ops  = problem.ops;

    % --- Build time grid via pluggable generator ---
    tgrid_fn = get_tgrid_fn(cfg);
    [t_grid, dt_vec]   = tgrid_fn(nt, cfg.vareps);
    problem.t_grid     = t_grid;   % (nt+1 x 1)
    problem.dt_vec     = dt_vec;   % (nt   x 1)

    % --- Precompute FP projection ---
    if isequal(cfg.projection, @proj_fokker_planck_expsemi_nonuniform)
        problem.expsemi_nonuniform_proj = ...
            precomp_expsemi_proj_nonuniform(problem, cfg.vareps);
    else
        error('discretize_then_optimize_nonuniform: unsupported projection %s', ...
              func2str(cfg.projection));
    end

    % --- Initial guesses using non-uniform time points ---
    t_stag = t_grid(2:nt);          % interior nodes t_1,...,t_{nt-1}  (ntm x 1)
    t_cc   = (t_grid(1:nt) + t_grid(2:nt+1)) / 2;  % midpoints (nt x 1)

    x0.rho = (1 - t_stag) .* rho0 + t_stag .* rho1;   % (ntm x nx)
    x0.mx  = zeros(nt, nxm);

    y0.rho = (1 - t_cc) .* rho0 + t_cc .* rho1;        % (nt x nx)
    y0.mx  = zeros(nt, nx);

    b = s_zeros(y0);

    % --- Operators ---
    gamma    = cfg.gamma;
    sigma    = 1 / gamma;
    zeros_nt = zeros(nt, 1);

    A_fn  = @(x) struct('rho', ops.interp_t_at_phi(x.rho, rho0, rho1), ...
                        'mx',  ops.interp_x_at_phi(x.mx, zeros_nt, zeros_nt));
    At_fn = @(v) struct('rho', ops.interp_t_at_rho(v.rho), ...
                        'mx',  ops.interp_x_at_m(v.mx));
    B_fn  = @(y) s_scale(-1, y);

    prox_f1 = @(v, step) cfg.projection(v, problem, cfg);

    solve_y = @(delta, z_hat) cfg.prox_ke( ...
        s_sub(z_hat, s_scale(sigma, delta)), sigma, problem);

    norm_fn = @(v) sqrt(dt * dx * (sum(v.rho(:).^2) + sum(v.mx(:).^2)));

    % --- ADMM options ---
    admm_opts.gamma    = gamma;
    admm_opts.tau      = cfg.tau;
    admm_opts.alpha    = get_alpha(cfg);
    admm_opts.max_iter = cfg.max_iter;
    admm_opts.tol      = cfg.tol;
    admm_opts.norm_fn  = norm_fn;

    % --- Solve ---
    [x, y, ~, info] = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, admm_opts);

    result.rho_stag  = x.rho;
    result.mx_stag   = x.mx;
    result.rho_cc    = y.rho;
    result.mx_cc     = y.mx;
    result.t_grid    = t_grid;
    result.dt_vec    = dt_vec;
    result.residual  = info.residual;
    result.iters     = info.iters;
    result.converged = info.converged;
    result.error     = info.residual(end);
    result.walltime  = info.walltime;
    result.cfg       = cfg;
end

function alpha = get_alpha(cfg)
    if isfield(cfg, 'alpha')
        alpha = cfg.alpha;
    else
        alpha = 1.0;
    end
end

function fn = get_tgrid_fn(cfg)
    if isfield(cfg, 'tgrid_fn') && ~isempty(cfg.tgrid_fn)
        fn = cfg.tgrid_fn;
    else
        fn = @make_nonuniform_tgrid;
    end
end
