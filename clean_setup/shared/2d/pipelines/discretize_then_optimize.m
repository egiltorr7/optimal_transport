function result = discretize_then_optimize(cfg, problem)
% DISCRETIZE_THEN_OPTIMIZE  Linearized ADMM pipeline for 1D OT / SB.
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
%     x.rho  (ntm x nx x ny)   staggered density
%     x.mx   (nt  x nxm x ny)  staggered momentum
%     x.my   (nt  x nx x nym)  staggered momentum
%     y.rho  (nt  x nx x ny)   cell-centre density
%     y.mx   (nt  x nx x ny)   cell-centre momentum
%
%   A is affine: A_fn(x).rho = interp_t_at_phi(x.rho, rho0, rho1)
%                A_fn(x).mx  = interp_x_at_phi(x.mx,  0,    0   )
%                A_fn(x).my  = interp_y_at_phi(x.my,  0,    0   )
%   The BC values rho0, rho1 enter as ghost values in the forward
%   time interpolation, enforcing the boundary conditions on x at
%   every projected-gradient x-step.
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
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;   nxm = nx - 1;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    ops  = problem.ops;

    % Precompute banded FP projection factors
    if isequal(cfg.projection, @proj_fokker_planck_banded)
        problem.banded_proj = precomp_banded_proj(problem, cfg.vareps);
    end

    % --- Initial guesses ---
    % x on staggered grid
    t_stag = linspace(0, 1, ntm)';
    x0.rho = (1 - t_stag) .* rho0 + t_stag .* rho1;   % (ntm x nx)
    x0.mx  = zeros(nt, nxm);                            % (nt  x nxm)

    % y on cell-centre grid
    t_cc   = ((1:nt)' - 0.5) * dt;
    y0.rho = (1 - t_cc) .* rho0 + t_cc .* rho1;        % (nt x nx)
    y0.mx  = zeros(nt, nx);                             % (nt x nx)

    % b = 0 on cell-centre grid (BCs absorbed into affine A_fn)
    b = s_zeros(y0);

    % --- Operators ---
    gamma     = cfg.gamma;
    sigma     = 1 / gamma;
    zeros_nt  = zeros(nt, 1);

    % A: affine interpolation staggered -> cell-centres (BCs baked in)
    A_fn  = @(x) struct('rho', ops.interp_t_at_phi(x.rho, rho0, rho1), ...
                        'mx',  ops.interp_x_at_phi(x.mx, zeros_nt, zeros_nt));

    % A^T: adjoint of the LINEAR part of A (no BC terms)
    At_fn = @(v) struct('rho', ops.interp_t_at_rho(v.rho), ...
                        'mx',  ops.interp_x_at_m(v.mx));

    B_fn  = @(y) s_scale(-1, y);

    % prox_f1 = projection onto FP constraint (x on staggered grid)
    prox_f1 = @(v, step) cfg.projection(v, problem, cfg);

    % solve_y = prox of KE (y on cell-centre grid)
    solve_y = @(delta, z_hat) cfg.prox_ke( ...
        s_sub(z_hat, s_scale(sigma, delta)), sigma, problem);

    % Weighted L2 norm on cell-centre grid for convergence
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

    % Return both grids: x on staggered (FP-feasible), y on cell-centres (KE-optimal)
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

function alpha = get_alpha(cfg)
    if isfield(cfg, 'alpha')
        alpha = cfg.alpha;
    else
        alpha = 1.0;
    end
end
