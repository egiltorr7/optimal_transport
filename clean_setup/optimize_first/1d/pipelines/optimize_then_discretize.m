function result = optimize_then_discretize(cfg, problem)
% OPTIMIZE_THEN_DISCRETIZE  ADMM pipeline for 1D optimal transport / SB.
%
%   result = optimize_then_discretize(cfg, problem)
%
%   Solves the consensus problem
%
%       min   KE(x) + I_FP(y)   s.t.   x - y = 0
%
%   which fits the GADMM template (Fang et al. 2015, eq. 3) with
%   A = I, B = -I, b = 0.
%
%   The x-subproblem reduces to the proximal operator of KE:
%
%     x^{t+1} = prox_{KE/ρ}( y^t + γ^t/ρ )
%
%   The y-subproblem reduces to the projection onto the FP constraint:
%
%     y^{t+1} = proj_FP( x^{t+1}_relax - γ^t/ρ )
%              where x^{t+1}_relax = z_hat (passed in by admm_solve)
%
%   Precomputes banded projection factors if cfg.projection is
%   proj_fokker_planck_banded.

    dt   = problem.dt;
    dx   = problem.dx;
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;   nxm = nx - 1;
    rho0 = problem.rho0;
    rho1 = problem.rho1;

    % Precompute banded projection factors if needed (once per run)
    if isequal(cfg.projection, @proj_fokker_planck_banded)
        problem.banded_proj = precomp_banded_proj(problem, cfg.vareps);
    end

    % Initial guess: linear interpolation in time
    t  = linspace(0, 1, nt + 1)';
    tt = t(2:end-1);   % (ntm x 1)

    x0.rho = (1 - tt) .* rho0 + tt .* rho1;   % (ntm x nx)
    x0.mx  = zeros(nt, nxm);                   % (nt  x nxm)
    y0     = x0;

    % b = 0 (consensus constraint: A=I, B=-I, b=0)
    b = s_zeros(x0);

    % For A=I, B=-I, b=0:
    %   x-subproblem: min KE(x) - x^T γ + (ρ/2)||x - y||²
    %               = prox_{KE/ρ}( y + γ/ρ )
    %   y-subproblem receives z_hat = α*x^{t+1} + (1-α)*y^t
    %               : min I_FP(y) + y^T γ + (ρ/2)||z_hat - y||²
    %               = proj_FP( z_hat - γ/ρ )
    %   A(x) = x,  B(y) = -y
    gamma = cfg.gamma;
    sigma = 1 / gamma;

    solve_x = @(delta, y) cfg.prox_ke( ...
        s_add(y, s_scale(sigma, delta)), sigma, problem);

    solve_y = @(delta, z_hat) cfg.projection( ...
        s_sub(z_hat, s_scale(sigma, delta)), problem, cfg);

    A_fn = @(x) x;
    B_fn = @(y) s_scale(-1, y);

    % Weighted L2 norm for convergence check
    norm_fn = @(v) sqrt(dt * dx * (sum(v.rho(:).^2) + sum(v.mx(:).^2)));

    % ADMM options
    admm_opts.gamma    = gamma;
    admm_opts.alpha    = get_alpha(cfg);
    admm_opts.max_iter = cfg.max_iter;
    admm_opts.tol      = cfg.tol;
    admm_opts.norm_fn  = norm_fn;

    % Solve
    [x, ~, ~, info] = admm_solve(solve_x, solve_y, A_fn, B_fn, b, x0, y0, admm_opts); %#ok<ASGLU>

    result.rho      = x.rho;
    result.mx       = x.mx;
    result.residual = info.residual;
    result.iters    = info.iters;
    result.converged = info.converged;
    result.error    = info.residual(end);
    result.walltime = info.walltime;
    result.cfg      = cfg;
end

function alpha = get_alpha(cfg)
    if isfield(cfg, 'alpha')
        alpha = cfg.alpha;
    else
        alpha = 1.0;   % standard ADMM
    end
end
