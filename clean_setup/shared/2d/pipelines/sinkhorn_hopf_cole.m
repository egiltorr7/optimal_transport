function result = sinkhorn_hopf_cole(problem, cfg)
% SINKHORN_HOPF_COLE  2D dynamic Schrödinger bridge via Hopf-Cole / Sinkhorn iteration.
%
%   result = sinkhorn_hopf_cole(problem, cfg)
%
%   Finds the Schrödinger bridge ρ(t,x,y) between ρ₀ and ρ₁ with diffusion
%   coefficient ε by alternating projections (Sinkhorn).
%
%   Hopf-Cole decomposition:  ρ(t,x,y) = φ(t,x,y) · ψ(t,x,y)
%   where:
%     ∂_t φ =  ε Δφ   (forward heat equation)
%    -∂_t ψ =  ε Δψ   (backward heat equation)
%
%   Algorithm (one-shot propagation):
%     Each Sinkhorn iteration applies the full-time kernel H_T in one shot:
%       1. φ(0)  ← ρ₀ / ψ(0)
%       2. φ(T)  = H_T[ φ(0) ]
%       3. ψ(T)  ← ρ₁ / φ(T)
%       4. ψ(0)  = H_T[ ψ(T) ]
%     After convergence the trajectory is recovered by applying H_t and
%     H_{T-t} to the converged φ(0) and ψ(T) for each output time.
%
%   Inputs:
%     problem   struct with fields: nt, nx, ny, dt, dx, dy, rho0, rho0_pdf,
%               rho1, rho1_pdf, lambda_x, lambda_y
%     cfg       struct with fields: vareps, max_iter, tol, precomp_heat
%     cfg.precomp_heat   function handle: heat = precomp_heat(problem, cfg)
%                        Built-in choices:
%                          @precomp_heat_neumann_2d    — DCT, reflecting walls
%                          @precomp_heat_free_space_2d — Gaussian conv, free-space BM on R^2
%     cfg.use_pdf_marginals  (optional, default false)
%                        true  — use raw PDF values as marginals
%                        false — use discrete probability masses (sum = 1)
%
%   Output:
%     result.rho        (nt+1 x nx x ny)    density at t = 0, dt, ..., T
%     result.mx         (nt+1 x nxm x ny)   x-momentum at staggered x-positions
%     result.my         (nt+1 x nx  x nym)  y-momentum at staggered y-positions
%     result.phi        (nt+1 x nx x ny)    Hopf-Cole forward factor φ
%     result.psi        (nt+1 x nx x ny)    Hopf-Cole backward factor ψ
%     result.t_grid     (nt+1 x 1)          time grid [0, dt, ..., T]
%     result.x_grid     (nx x 1)            cell-centre x-grid
%     result.y_grid     (1 x ny)            cell-centre y-grid
%     result.x_stag     (nxm x 1)           staggered x-grid
%     result.y_stag     (1 x nym)           staggered y-grid
%     result.errors     (iters x 1)         left-marginal L2 error per iteration
%     result.iters      scalar              number of Sinkhorn iterations run
%     result.error      scalar              final left-marginal L2 error
%     result.converged  logical             true if error < tol
%     result.walltime   scalar              wall-clock time in seconds
%     result.heat_name  char                name of the heat kernel used

    nt     = problem.nt;
    nx     = problem.nx;
    ny     = problem.ny;
    nxm    = nx - 1;
    nym    = ny - 1;
    dt     = problem.dt;
    dx     = problem.dx;
    dy     = problem.dy;
    vareps = cfg.vareps;

    % Marginals
    if isfield(cfg, 'use_pdf_marginals') && cfg.use_pdf_marginals
        rho0 = problem.rho0_pdf;   % raw PDF, sum ~ 1/(dx*dy)
        rho1 = problem.rho1_pdf;
    else
        % Discrete probability mass: sum(rho0(:)) = 1
        rho0 = problem.rho0_pdf / sum(problem.rho0_pdf(:));
        rho1 = problem.rho1_pdf / sum(problem.rho1_pdf(:));
    end

    max_iter = cfg.max_iter;
    tol      = cfg.tol;

    t_start = tic;

    % Precompute heat kernels
    heat = cfg.precomp_heat(problem, cfg);

    % Grids
    t_grid = (0:nt)' * dt;
    x_grid = problem.xx;   % (nx x 1)
    y_grid = problem.yy;   % (1  x ny)
    x_stag = (1:nxm)' * dx;
    y_stag = (1:nym)  * dy;

    % -----------------------------------------------------------------
    % Sinkhorn loop: one-shot propagation with H_T at each iteration
    % -----------------------------------------------------------------
    psi_0    = ones(nx, ny);
    errors   = zeros(max_iter, 1);
    converged = false;
    iter      = 0;

    for iter = 1:max_iter

        % 1. Left marginal: φ(0) = ρ₀ / ψ(0)
        phi_0 = rho0 ./ max(psi_0, 1e-300);

        % 2. Forward: φ(T) = H_T[ φ(0) ]
        phi_T = heat.apply_full(phi_0);

        % 3. Right marginal: ψ(T) = ρ₁ / φ(T)
        psi_T = rho1 ./ max(phi_T, 1e-300);

        % 4. Backward: ψ(0) = H_T[ ψ(T) ]
        psi_0 = heat.apply_full(psi_T);

        % 5. Convergence: L2 error on left marginal
        err = sqrt(dx*dy) * norm(phi_0(:) .* psi_0(:) - rho0(:));
        errors(iter) = err;

        if err < tol
            converged = true;
            break;
        end
    end

    walltime = toc(t_start);

    % -----------------------------------------------------------------
    % Trajectory recovery: apply H_t and H_{T-t} to converged factors
    % -----------------------------------------------------------------
    phi_traj = zeros(nt+1, nx, ny);
    psi_traj = zeros(nt+1, nx, ny);

    for k = 0:nt
        t_k = k * dt;
        phi_traj(k+1, :, :) = reshape(heat.apply_time(phi_0, t_k),     1, nx, ny);
        psi_traj(k+1, :, :) = reshape(heat.apply_time(psi_T, 1 - t_k), 1, nx, ny);
    end

    % Density
    rho_traj = phi_traj .* psi_traj;

    % Momentum on staggered grids
    %   mx(t, i, j) = 2ε · φ̃_x(i,j,t) · (ψ(i+1,j,t) - ψ(i,j,t)) / dx
    %   my(t, i, j) = 2ε · φ̃_y(i,j,t) · (ψ(i,j+1,t) - ψ(i,j,t)) / dy
    mx_traj = zeros(nt+1, nxm, ny);
    my_traj = zeros(nt+1, nx,  nym);

    for k = 1:(nt+1)
        phi_k = squeeze(phi_traj(k, :, :));   % (nx x ny)
        psi_k = squeeze(psi_traj(k, :, :));   % (nx x ny)

        phi_stag_x = 0.5 * (phi_k(1:nxm, :) + phi_k(2:nx, :));      % (nxm x ny)
        dpsi_x     = (psi_k(2:nx, :) - psi_k(1:nxm, :)) / dx;
        mx_traj(k, :, :) = reshape(2 * vareps * phi_stag_x .* dpsi_x, 1, nxm, ny);

        phi_stag_y = 0.5 * (phi_k(:, 1:nym) + phi_k(:, 2:ny));      % (nx x nym)
        dpsi_y     = (psi_k(:, 2:ny) - psi_k(:, 1:nym)) / dy;
        my_traj(k, :, :) = reshape(2 * vareps * phi_stag_y .* dpsi_y, 1, nx, nym);
    end

    result.rho       = rho_traj;
    result.mx        = mx_traj;
    result.my        = my_traj;
    result.phi       = phi_traj;
    result.psi       = psi_traj;
    result.t_grid    = t_grid;
    result.x_grid    = x_grid;
    result.y_grid    = y_grid;
    result.x_stag    = x_stag;
    result.y_stag    = y_stag;
    result.errors    = errors(1:iter);
    result.iters     = iter;
    result.error     = errors(iter);
    result.converged  = converged;
    result.walltime   = walltime;
    result.heat_name  = heat.name;
end
