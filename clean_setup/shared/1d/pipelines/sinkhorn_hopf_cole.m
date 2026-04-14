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
%     ∂_t φ = ε Δφ          (forward heat equation)
%    -∂_t ψ = ε Δψ          (backward heat equation)
%
%   Marginal conditions:
%     φ(0) · ψ(0) = ρ₀     (left marginal)
%     φ(T) · ψ(T) = ρ₁     (right marginal)
%
%   Sinkhorn iteration (each step enforces one marginal):
%     1. φ(0)  ← ρ₀ / ψ(0)         (left marginal update)
%     2. φ(t)  = H_t[ φ(0) ]        (forward propagation via H_{dt} steps)
%     3. ψ(T)  ← ρ₁ / φ(T)         (right marginal update)
%     4. ψ(t)  = H_{T-t}[ ψ(T) ]   (backward propagation)
%
%   Momentum recovery (on staggered x-grid, j = 1..nx-1):
%     m(t, x_j) = 2ε · φ̃(t, x_j) · (ψ(t, x_{j+1/2}) - ψ(t, x_{j-1/2})) / dx
%   where φ̃ = average of φ at the two adjacent cell centres.
%
%   Implementation:
%     Heat semigroup H_t applied via DCT-II (Neumann BCs):
%       H_t[φ]_hat(k) = exp(-ε λ_k t) φ_hat(k)
%     One-step decay for H_{dt} is precomputed once.
%
%   Inputs:
%     problem   struct with fields: nt, nx, dt, dx, rho0, rho1, lambda_x
%     cfg       struct with fields: vareps, max_iter, tol
%
%   Output:
%     result.rho        (nt+1 x nx)   density at t = 0, dt, 2dt, ..., T
%     result.mx         (nt+1 x nxm)  momentum at staggered x-positions j*dx
%     result.phi        (nt+1 x nx)   Hopf-Cole forward factor φ
%     result.psi        (nt+1 x nx)   Hopf-Cole backward factor ψ
%     result.t_grid     (nt+1 x 1)    time grid [0, dt, ..., T]
%     result.x_grid     (1 x nx)      cell-centre x-grid
%     result.x_stag     (1 x nxm)     staggered x-grid for momentum
%     result.errors     (iters x 1)   right-marginal L2 error per iteration
%     result.iters      scalar        number of Sinkhorn iterations run
%     result.error      scalar        final right-marginal L2 error
%     result.converged  logical       true if error < tol
%     result.walltime   scalar        wall-clock time in seconds

    nt       = problem.nt;
    nx       = problem.nx;
    nxm      = nx - 1;
    dt       = problem.dt;
    dx       = problem.dx;
    rho0     = problem.rho0(:);       % (nx x 1)
    rho1     = problem.rho1(:);       % (nx x 1)
    lambda_x = problem.lambda_x(:);   % (nx x 1)
    vareps   = cfg.vareps;
    max_iter = cfg.max_iter;
    tol      = cfg.tol;

    t_start = tic;

    % Precompute one-step decay: H_{dt}[φ]_hat(k) = decay_dt(k) * φ_hat(k)
    decay_dt = exp(-vareps * lambda_x * dt);   % (nx x 1)

    % Grids
    t_grid = (0:nt)' * dt;                         % (nt+1 x 1)
    x_grid = problem.xx;                            % (1 x nx)  cell centres
    x_stag = (1:nxm)' * dx;                        % (nxm x 1) staggered

    % Trajectory storage: (nt+1) x nx,  row k+1 = time k*dt
    phi_traj = zeros(nt+1, nx);
    psi_traj = zeros(nt+1, nx);

    % Initialise ψ(0) = ones  (will be updated each iteration)
    psi_0 = ones(nx, 1);

    errors    = zeros(max_iter, 1);
    converged = false;
    iter      = 0;

    for iter = 1:max_iter

        % ------------------------------------------------------------------
        % 1. Left marginal: φ(0) = ρ₀ / ψ(0)
        % ------------------------------------------------------------------
        phi_0 = rho0 ./ max(psi_0, 1e-300);
        phi_traj(1, :) = phi_0';

        % ------------------------------------------------------------------
        % 2. Forward propagation: φ(k+1) = H_{dt}[ φ(k) ]
        % ------------------------------------------------------------------
        phi_cur = phi_0;
        for k = 1:nt
            phi_cur = forward_step(phi_cur, decay_dt);
            phi_traj(k+1, :) = phi_cur';
        end

        % ------------------------------------------------------------------
        % 3. Right marginal: ψ(T) = ρ₁ / φ(T)
        % ------------------------------------------------------------------
        phi_T = phi_traj(nt+1, :)';
        psi_T = rho1 ./ max(phi_T, 1e-300);
        psi_traj(nt+1, :) = psi_T';

        % ------------------------------------------------------------------
        % 4. Backward propagation: ψ(k-1) = H_{dt}[ ψ(k) ]
        %    (backward in time = same heat equation, applied in reverse)
        % ------------------------------------------------------------------
        psi_cur = psi_T;
        for k = nt:-1:1
            psi_cur = forward_step(psi_cur, decay_dt);
            psi_traj(k, :) = psi_cur';
        end
        psi_0 = psi_traj(1, :)';

        % ------------------------------------------------------------------
        % Convergence: L2 error in right marginal
        %   At convergence: φ(T)·ψ(T) = ρ₁  (ψ(T) was just set so this is exact;
        %   check the LEFT marginal instead: φ(0)·ψ(0) vs ρ₀)
        % ------------------------------------------------------------------
        rho_0_check = phi_0 .* psi_0;
        err = sqrt(dx) * norm(rho_0_check - rho0);
        errors(iter) = err;

        if err < tol
            converged = true;
            break;
        end
    end

    walltime = toc(t_start);

    % --- Recover density ---
    rho_traj = phi_traj .* psi_traj;   % (nt+1 x nx)

    % --- Recover momentum on staggered x-grid ---
    %   m(t, x_j) = 2ε · φ̃_j(t) · (ψ_{j+1}(t) - ψ_j(t)) / dx,  j = 1..nxm
    %   φ̃_j = average of φ at adjacent cell centres j and j+1
    mx_traj = zeros(nt+1, nxm);
    for k = 1:(nt+1)
        phi_k  = phi_traj(k, :)';    % (nx x 1)
        psi_k  = psi_traj(k, :)';    % (nx x 1)
        phi_stag = 0.5 * (phi_k(1:nxm) + phi_k(2:nx));   % (nxm x 1)
        dpsi_stag = (psi_k(2:nx) - psi_k(1:nxm)) / dx;   % (nxm x 1)
        mx_traj(k, :) = (2 * vareps * phi_stag .* dpsi_stag)';
    end

    result.rho       = rho_traj;               % (nt+1 x nx)
    result.mx        = mx_traj;                % (nt+1 x nxm)
    result.phi       = phi_traj;               % (nt+1 x nx)
    result.psi       = psi_traj;               % (nt+1 x nx)
    result.t_grid    = t_grid;                 % (nt+1 x 1)
    result.x_grid    = x_grid;                 % (1 x nx)
    result.x_stag    = x_stag;                 % (nxm x 1)
    result.errors    = errors(1:iter);
    result.iters     = iter;
    result.error     = errors(iter);
    result.converged = converged;
    result.walltime  = walltime;
end

% -------------------------------------------------------------------------

function phi_out = forward_step(phi_in, decay_dt)
% FORWARD_STEP  Apply H_{dt} to phi_in via DCT.
    phi_out = idct(decay_dt .* dct(phi_in));
    phi_out = max(phi_out, 0);
end
