function result = optimize_then_discretize(cfg, problem)
% OPTIMIZE_THEN_DISCRETIZE  ADMM pipeline for 1D optimal transport.
%
%   result = optimize_then_discretize(cfg, problem)
%
%   Alternates between:
%     1. Proximal step   (cfg.prox_ke)    -- minimizes kinetic energy
%     2. Projection step (cfg.projection) -- enforces Fokker-Planck constraint
%     3. Dual variable update
%
%   The ADMM combination steps (adding/subtracting dual variables) are
%   computed here in the pipeline. The prox and projection functions receive
%   and return plain (rho, mx) pairs with no knowledge of the ADMM structure.
%
%   State struct fields:
%     state.rho        (ntm x nx)  primal density
%     state.mx         (nt  x nxm) primal momentum
%     state.rho_tilde  (ntm x nx)  projected density
%     state.bx         (nt  x nxm) projected momentum
%     state.delta_rho  (ntm x nx)  dual variable for density
%     state.delta_mx   (nt  x nxm) dual variable for momentum

    nt    = problem.nt;   ntm = nt - 1;
    nx    = problem.nx;   nxm = nx - 1;
    dt    = problem.dt;
    dx    = problem.dx;
    rho0  = problem.rho0;
    rho1  = problem.rho1;
    gamma = cfg.gamma;
    sigma = 1 / gamma;

    % Initial guess: linear interpolation in time
    t  = linspace(0, 1, nt + 1)';
    tt = t(2:end-1);  % (ntm x 1)

    state.rho       = (1 - tt) .* rho0 + tt .* rho1;  % (ntm x nx)
    state.mx        = zeros(nt, nxm);
    state.rho_tilde = state.rho;
    state.bx        = state.mx;
    state.delta_rho = zeros(ntm, nx);
    state.delta_mx  = zeros(nt, nxm);

    % Precompute banded projection factors if needed
    if isequal(cfg.projection, @proj_fokker_planck_banded)
        problem.banded_proj = precomp_banded_proj(problem, cfg.vareps);
    end

    residual = zeros(cfg.max_iter, 1);

    tic;
    for iter = 1:cfg.max_iter

        rho_tilde_prev = state.rho_tilde;
        bx_prev        = state.bx;

        % Step 1: proximal update
        %   prox input = projected variable + dual/gamma
        prox_in.rho = state.rho_tilde + state.delta_rho ./ gamma;
        prox_in.mx  = state.bx        + state.delta_mx  ./ gamma;
        prox_out    = cfg.prox_ke(prox_in, sigma, problem);
        state.rho   = prox_out.rho;
        state.mx    = prox_out.mx;

        % Step 2: projection onto Fokker-Planck constraint
        %   projection input = primal variable - dual/gamma
        proj_in.rho     = state.rho - state.delta_rho ./ gamma;
        proj_in.mx      = state.mx  - state.delta_mx  ./ gamma;
        proj_out        = cfg.projection(proj_in, problem, cfg);
        state.rho_tilde = proj_out.rho;
        state.bx        = proj_out.mx;

        % Step 3: dual update
        state.delta_rho = state.delta_rho - gamma .* (state.rho - state.rho_tilde);
        state.delta_mx  = state.delta_mx  - gamma .* (state.mx  - state.bx);

        % Residual: change in projected variables
        drho = (state.rho_tilde - rho_tilde_prev).^2;
        dmx  = (state.bx - bx_prev).^2;
        residual(iter) = sqrt(dt * dx * (sum(drho(:)) + sum(dmx(:))));

        if residual(iter) < cfg.tol
            residual = residual(1:iter);
            break;
        end
    end

    result.rho       = state.rho;
    result.mx        = state.mx;
    result.residual  = residual;
    result.iters     = length(residual);
    result.converged = residual(end) < cfg.tol;
    result.error     = residual(end);
    result.walltime  = toc;
    result.cfg       = cfg;
end
