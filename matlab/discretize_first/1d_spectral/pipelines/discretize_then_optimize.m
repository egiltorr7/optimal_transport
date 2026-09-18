function result = discretize_then_optimize(cfg, problem)
% DISCRETIZE_THEN_OPTIMIZE  Standard ADMM pipeline for spectral 1D OT / SB.
%
%   result = discretize_then_optimize(cfg, problem)
%
%   Solves the problem on the collocated edge grid (nt+1) x nx:
%
%       min   KE(y)         s.t.   FP(x) = 0,   x = y
%
%   using standard ADMM (admm_solve) with A = I, B = -I, b = 0:
%
%     x-step:  x = proj_FP( y + delta/gamma )    [exact projection]
%     y-step:  y = prox_KE( x - delta/gamma, 1/gamma )
%     dual  :  delta = delta - gamma * (x - y)
%
%   Time grid: nt+1 edge points t_n = n/nt, n = 0,...,nt.
%   The BCs rho(t=0)=rho0 and rho(t=T)=rho1 are grid points (n=0 and n=nt)
%   and are enforced exactly by the FP projection.
%
%   Variables:
%     x.rho  (nt+1 x nx)   FP-feasible density
%     x.mx   (nt+1 x nx)   FP-feasible momentum
%     y.rho  (nt+1 x nx)   KE-optimal density
%     y.mx   (nt+1 x nx)   KE-optimal momentum

    nt   = problem.nt;
    nx   = problem.nx;
    dt   = problem.dt;
    dx   = problem.dx;
    rho0 = problem.rho0;   % (1 x nx)
    rho1 = problem.rho1;

    gamma = cfg.gamma;
    sigma = 1 / gamma;

    % --- Initial guess: linear interpolation on edge grid ---
    t_edges = linspace(0, 1, nt+1)';                          % (nt+1 x 1)
    x0.rho  = (1 - t_edges) .* rho0 + t_edges .* rho1;       % (nt+1 x nx)
    x0.mx   = zeros(nt+1, nx);
    y0      = x0;
    b       = s_zeros(x0);

    % --- Operators ---
    A_fn = @(x) x;                   % A = I
    B_fn = @(y) s_scale(-1, y);      % B = -I

    % x-step: project (y + delta/gamma) onto FP constraint
    solve_x = @(delta, y) cfg.projection( ...
        s_add(y, s_scale(sigma, delta)), problem, cfg);

    % y-step: prox of KE at (x - delta/gamma)
    solve_y = @(delta, z_hat) cfg.prox_ke( ...
        s_sub(z_hat, s_scale(sigma, delta)), sigma, problem);

    % Weighted L2 norm for convergence check
    norm_fn = @(v) sqrt(dt * dx * (sum(v.rho(:).^2) + sum(v.mx(:).^2)));

    % --- ADMM options ---
    admm_opts.gamma    = gamma;
    admm_opts.alpha    = get_alpha(cfg);
    admm_opts.max_iter = cfg.max_iter;
    admm_opts.tol      = cfg.tol;
    admm_opts.norm_fn  = norm_fn;

    % --- Solve ---
    [x, y, ~, info] = admm_solve(solve_x, solve_y, A_fn, B_fn, b, x0, y0, admm_opts);

    % Return both: x is FP-feasible, y is KE-optimal
    result.rho       = y.rho;    % (nt+1 x nx), time at t=0,dt,...,1
    result.mx        = y.mx;
    result.rho_proj  = x.rho;
    result.mx_proj   = x.mx;
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
