function result = optimize_then_discretize(cfg, problem)
% OPTIMIZE_THEN_DISCRETIZE  ADMM pipeline for 1D optimal transport / SB.
%
%   result = optimize_then_discretize(cfg, problem)
%
%   Solves the consensus problem
%
%       min   KE(x) + I_FP(z)   s.t.   x = z
%
%   using admm_solve (over-relaxed ADMM, Fang et al. GADMM notation).
%   The two proximal operators are:
%
%     prox_f = prox of kinetic energy KE     (cfg.prox_ke)
%     prox_g = prox of Fokker-Planck indicator = orthogonal projection
%              onto the FP constraint set     (cfg.projection)
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

    % Proximal operators as closures (capture problem and cfg)
    prox_f = @(v, sigma) cfg.prox_ke(v, sigma, problem);
    prox_g = @(v, sigma) cfg.projection(v, problem, cfg);

    % Weighted L2 norm for convergence check
    norm_fn = @(v) sqrt(dt * dx * (sum(v.rho(:).^2) + sum(v.mx(:).^2)));

    % ADMM options
    admm_opts.rho      = cfg.gamma;
    admm_opts.alpha    = get_alpha(cfg);
    admm_opts.max_iter = cfg.max_iter;
    admm_opts.tol      = cfg.tol;
    admm_opts.norm_fn  = norm_fn;

    % Solve
    [x, ~, info] = admm_solve(prox_f, prox_g, x0, admm_opts);

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
