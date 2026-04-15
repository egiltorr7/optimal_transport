function result = sinkhorn_hopf_cole(problem, cfg)
% SINKHORN_HOPF_COLE  Dynamic Schrödinger bridge via Hopf-Cole / Sinkhorn iteration.
%
%   result = sinkhorn_hopf_cole(problem, cfg)
%
%   Finds the Schrödinger bridge ρ(t) and momentum m(t) between ρ₀ and ρ₁
%   with diffusion coefficient ε by alternating projections (Sinkhorn).
%
%   Hopf-Cole decomposition:  ρ(t,x) = φ(t,x) · ψ(t,x)
%   where:
%     ∂_t φ =  ε Δφ   (forward heat equation)
%    -∂_t ψ =  ε Δψ   (backward heat equation)
%
%   Marginal conditions:
%     φ(0) · ψ(0) = ρ₀     (left marginal)
%     φ(T) · ψ(T) = ρ₁     (right marginal)
%
%   Algorithm (one-shot propagation):
%     Each Sinkhorn iteration applies the full-time kernel H_T in one shot:
%       1. φ(0)  ← ρ₀ / ψ(0)
%       2. φ(T)  = H_T[ φ(0) ]        (one application of K_T)
%       3. ψ(T)  ← ρ₁ / φ(T)
%       4. ψ(0)  = H_T[ ψ(T) ]        (one application of K_T)
%     After convergence the trajectory is recovered by applying H_t and
%     H_{T-t} to the converged φ(0) and ψ(T) for each output time.
%
%   Why one-shot?
%     Step-by-step propagation (nt small steps with zero-padding) compounds
%     truncation errors: after k steps φ has spread outside [0,1] and gets
%     clipped at each subsequent step.  φ(0) and ψ(T) are narrow (they equal
%     the marginals divided by a slowly-varying factor), so a single zero-
%     padded convolution of K_T with a narrow function is accurate even for
%     large ε.
%
%   Inputs:
%     problem        struct with fields: nt, nx, dt, dx, rho0, rho0_pdf, lambda_x
%     cfg            struct with fields: vareps, max_iter, tol, precomp_heat
%     cfg.precomp_heat   function handle: heat = precomp_heat(problem, cfg)
%                        heat must provide:
%                          heat.apply_full(phi)     — apply H_T (full time)
%                          heat.apply_time(phi, t)  — apply H_t for arbitrary t
%                        Built-in choices:
%                          @precomp_heat_neumann     — DCT, reflecting walls
%                          @precomp_heat_free_space  — Gaussian conv, free-space BM
%     cfg.use_pdf_marginals  (optional, default false)
%                        true  — use raw PDF values as marginals (for free-space / R)
%                        false — use discrete probability masses normalised over [0,1]
%
%   Output:
%     result.rho        (nt+1 x nx)   density at t = 0, dt, 2dt, ..., T
%     result.mx         (nt+1 x nxm)  momentum at staggered x-positions j*dx
%     result.phi        (nt+1 x nx)   Hopf-Cole forward factor φ
%     result.psi        (nt+1 x nx)   Hopf-Cole backward factor ψ
%     result.t_grid     (nt+1 x 1)    time grid [0, dt, ..., T]
%     result.x_grid     (1 x nx)      cell-centre x-grid
%     result.x_stag     (1 x nxm)     staggered x-grid for momentum
%     result.errors     (iters x 1)   left-marginal L2 error per iteration
%     result.iters      scalar        number of Sinkhorn iterations run
%     result.error      scalar        final left-marginal L2 error
%     result.converged  logical       true if error < tol
%     result.walltime   scalar        wall-clock time in seconds
%     result.heat_name  char          name of the heat kernel used

    nt     = problem.nt;
    nx     = problem.nx;
    nxm    = nx - 1;
    dt     = problem.dt;
    dx     = problem.dx;
    vareps = cfg.vareps;

    % Marginals
    if isfield(cfg, 'use_pdf_marginals') && cfg.use_pdf_marginals
        rho0 = problem.rho0_pdf(:);   % raw PDF, sum ~ 1/dx
        rho1 = problem.rho1_pdf(:);
    else
        rho0 = problem.rho0(:);       % normalised, sum = 1
        rho1 = problem.rho1(:);
    end

    max_iter = cfg.max_iter;
    tol      = cfg.tol;

    t_start = tic;

    % Precompute heat kernels
    heat = cfg.precomp_heat(problem, cfg);

    % Grids
    t_grid = (0:nt)' * dt;
    x_grid = problem.xx;
    x_stag = (1:nxm)' * dx;

    % -----------------------------------------------------------------
    % Sinkhorn loop: one-shot propagation with H_T at each iteration
    % -----------------------------------------------------------------
    psi_0    = ones(nx, 1);
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
        err = sqrt(dx) * norm(phi_0 .* psi_0 - rho0);
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
    phi_traj = zeros(nt+1, nx);
    psi_traj = zeros(nt+1, nx);

    for k = 0:nt
        t_k = k * dt;
        phi_traj(k+1, :) = heat.apply_time(phi_0, t_k)';
        psi_traj(k+1, :) = heat.apply_time(psi_T, 1 - t_k)';
    end

    % Density
    rho_traj = phi_traj .* psi_traj;

    % Momentum on staggered x-grid
    %   m(t, x_j) = 2ε · φ̃_j(t) · (ψ_{j+1}(t) - ψ_j(t)) / dx
    mx_traj = zeros(nt+1, nxm);
    for k = 1:(nt+1)
        phi_k     = phi_traj(k, :)';
        psi_k     = psi_traj(k, :)';
        phi_stag  = 0.5 * (phi_k(1:nxm) + phi_k(2:nx));
        dpsi_stag = (psi_k(2:nx) - psi_k(1:nxm)) / dx;
        mx_traj(k, :) = (2 * vareps * phi_stag .* dpsi_stag)';
    end

    result.rho       = rho_traj;
    result.mx        = mx_traj;
    result.phi       = phi_traj;
    result.psi       = psi_traj;
    result.t_grid    = t_grid;
    result.x_grid    = x_grid;
    result.x_stag    = x_stag;
    result.errors    = errors(1:iter);
    result.iters     = iter;
    result.error     = errors(iter);
    result.converged  = converged;
    result.walltime   = walltime;
    result.heat_name  = heat.name;
end
