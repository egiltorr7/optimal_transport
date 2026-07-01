function result = discretize_then_optimize_etd2(cfg, problem)
% DISCRETIZE_THEN_OPTIMIZE_ETD2  ADMM pipeline for ETD2 FP projection.
%
%   Like discretize_then_optimize but x.mx lives at nt+1 INTEGER time nodes
%   {t_0,...,t_nt} instead of nt half-integer nodes.  All other aspects of
%   the ADMM structure (LADMM, prox_KE, dual variable) are unchanged.
%
%   Grid:
%     x.rho  (ntm   x nx)   staggered density (interior integer times)
%     x.mx   (nt+1  x nxm)  momentum at integer times t_0,...,t_nt
%     y.rho  (nt    x nx)   cell-centre density
%     y.mx   (nt    x nx)   cell-centre momentum
%
%   A operator for mx:
%     Forward:  average adjacent integer-time m values -> (nt x nxm) half-int,
%               then interp x face->center -> (nt x nx).
%     Adjoint:  interp x center->face -> (nt x nxm),
%               then spread half-int weights back to integer times -> (nt+1 x nxm).
%
%   Use with cfg.projection = @proj_fokker_planck_expsemi_etd2.

    dt   = problem.dt;
    dx   = problem.dx;
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;   nxm = nx - 1;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    ops  = problem.ops;

    if isfield(cfg, 'use_gpu') && cfg.use_gpu
        error('discretize_then_optimize_etd2: GPU path not yet implemented.');
    end

    % Precompute ETD2 projection factors
    problem.expsemi_proj_etd2 = precomp_expsemi_proj_etd2(problem, cfg.vareps);

    %% --- Initial guesses ---
    if isfield(cfg, 'x0') && isfield(cfg, 'y0')
        x0 = cfg.x0;
        y0 = cfg.y0;
    else
        t_stag = linspace(0, 1, ntm)';
        x0.rho = (1 - t_stag) .* rho0 + t_stag .* rho1;   % (ntm x nx)
        x0.mx  = zeros(nt+1, nxm);                          % (nt+1 x nxm) at integer times

        t_cc   = ((1:nt)' - 0.5) * dt;
        y0.rho = (1 - t_cc) .* rho0 + t_cc .* rho1;        % (nt x nx)
        y0.mx  = zeros(nt, nx);                              % (nt x nx)
    end

    b        = s_zeros(y0);
    gamma    = cfg.gamma;
    sigma    = 1 / gamma;
    zeros_nt = zeros(nt, 1);

    %% --- Operators ---
    % A_rho: staggered-in-t interior -> cell-centre (same as ETD1)
    % A_mx:  integer-time m -> half-int (time average) -> cell-centre (x interp)
    A_fn = @(x) struct( ...
        'rho', ops.interp_t_at_phi(x.rho, rho0, rho1), ...
        'mx',  ops.interp_x_at_phi(0.5*(x.mx(1:nt,:) + x.mx(2:nt+1,:)), zeros_nt, zeros_nt));

    % At_rho: cell-centre -> staggered-in-t interior (same as ETD1)
    % At_mx:  cell-centre -> face (x interp) -> integer-time (t spread)
    At_fn = @(v) struct( ...
        'rho', ops.interp_t_at_rho(v.rho), ...
        'mx',  t_spread(ops.interp_x_at_m(v.mx), nt));

    B_fn    = @(y) s_scale(-1, y);
    prox_f1 = @(v, step) cfg.projection(v, problem, cfg);
    solve_y = @(delta, z_hat) cfg.prox_ke( ...
        s_sub(z_hat, s_scale(sigma, delta)), sigma, problem);
    norm_fn = @(v) sqrt(dt * dx * (sum(v.rho(:).^2) + sum(v.mx(:).^2)));

    admm_opts.gamma    = gamma;
    admm_opts.tau      = cfg.tau;
    admm_opts.alpha    = get_alpha(cfg);
    admm_opts.max_iter = cfg.max_iter;
    admm_opts.tol      = cfg.tol;
    admm_opts.norm_fn  = norm_fn;

    [x, y, ~, info] = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, admm_opts);

    result.rho_stag  = x.rho;
    result.mx_stag   = x.mx;
    result.rho_cc    = y.rho;
    result.mx_cc     = y.mx;
    result.residual  = info.residual;
    result.iters     = info.iters;
    result.converged = info.converged;
    result.error     = info.residual(end);
    result.walltime  = info.walltime;
    result.cfg       = cfg;
end

function out = t_spread(v_face, nt)
% Adjoint of time-average: spread (nt x nxm) -> (nt+1 x nxm).
% Each v_face(j,:) contributes 0.5 to out(j,:) and 0.5 to out(j+1,:).
    out = zeros(nt+1, size(v_face, 2));
    out(1, :)    = 0.5 * v_face(1, :);
    out(2:nt, :) = 0.5 * (v_face(1:nt-1, :) + v_face(2:nt, :));
    out(nt+1, :) = 0.5 * v_face(nt, :);
end

function alpha = get_alpha(cfg)
    if isfield(cfg, 'alpha'), alpha = cfg.alpha; else, alpha = 1.0; end
end
