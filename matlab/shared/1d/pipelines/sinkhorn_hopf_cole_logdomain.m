function result = sinkhorn_hopf_cole_logdomain(problem, cfg)
% SINKHORN_HOPF_COLE_LOGDOMAIN  Log-domain Sinkhorn for Schrödinger bridge.
%
%   result = sinkhorn_hopf_cole_logdomain(problem, cfg)
%
%   Maintains Schrödinger potentials a = log(phi_0), b = log(psi_T) to
%   avoid overflow/underflow for small epsilon.  Supports epsilon-annealing
%   for cases where the heat kernel spread sqrt(2*eps*T) is much smaller
%   than the Wasserstein distance between marginals.
%
%   Log-domain Sinkhorn updates (2 heat-kernel calls per iteration):
%     a  <-  log(rho0) - log(H_T[exp(b)])
%     b  <-  log(rho1) - log(H_T[exp(a)])
%
%   Epsilon-annealing (enabled by cfg.eps_init > cfg.vareps):
%     Runs a geometric schedule  eps_init -> ... -> vareps,  converging at
%     each level before moving to the next.  Each level warm-starts from
%     the previous converged potentials.  This avoids the machine-zero
%     issue where H_T[rho0] has zero support near supp(rho1) for small eps.
%
%   Inputs:
%     problem   struct: nt, nx, dt, dx, rho0, lambda_x, ...
%     cfg       struct:
%       cfg.vareps          target diffusion epsilon
%       cfg.max_iter        max Sinkhorn iterations (per level if annealing)
%       cfg.tol             convergence tolerance (-1 to disable)
%       cfg.precomp_heat    log-domain heat handle, e.g. @precomp_heat_neumann_log
%       cfg.eps_init        (optional) starting eps for annealing; must be > vareps
%       cfg.anneal_factor   (optional) geometric ratio between levels (default 2)
%
%   Output fields (identical to sinkhorn_hopf_cole plus log-potentials):
%     result.rho, .mx, .phi, .psi, .t_grid, .x_grid, .x_stag
%     result.errors    left-marginal L2 error, concatenated across all levels
%     result.iters     total iterations across all levels
%     result.error     final left-marginal L2 error
%     result.converged logical
%     result.walltime  scalar
%     result.heat_name char
%     result.log_phi0  (nx x 1)  a = log(phi_0) at convergence
%     result.log_psiT  (nx x 1)  b = log(psi_T) at convergence
%     result.eps_schedule  annealing levels used (scalar if no annealing)

    nt     = problem.nt;
    nx     = problem.nx;
    nxm    = nx - 1;
    dt     = problem.dt;
    dx     = problem.dx;
    vareps = cfg.vareps;

    if isfield(cfg, 'use_pdf_marginals') && cfg.use_pdf_marginals
        rho0 = problem.rho0_pdf(:);
        rho1 = problem.rho1_pdf(:);
    else
        rho0 = problem.rho0(:);
        rho1 = problem.rho1(:);
    end

    max_iter = cfg.max_iter;
    tol      = cfg.tol;

    % Precompute heat (may be rebuilt per annealing level)
    precomp_fn = @precomp_heat_neumann_log;
    if isfield(cfg, 'precomp_heat') && ~isempty(cfg.precomp_heat)
        precomp_fn = cfg.precomp_heat;
    end

    % -----------------------------------------------------------------
    % Build epsilon-annealing schedule
    % -----------------------------------------------------------------
    anneal_factor = 2.0;
    if isfield(cfg, 'anneal_factor'), anneal_factor = cfg.anneal_factor; end

    if isfield(cfg, 'eps_init') && cfg.eps_init > vareps
        eps_start = cfg.eps_init;
        n_levels  = ceil(log(eps_start / vareps) / log(anneal_factor));
        eps_sched = eps_start ./ (anneal_factor .^ (0:n_levels));
        eps_sched(end) = vareps;   % pin exact target
    else
        eps_sched = vareps;
    end

    t_start = tic;

    % Grids
    t_grid = (0:nt)' * dt;
    x_grid = problem.xx;
    x_stag = (1:nxm)' * dx;

    log_rho0 = log(rho0);
    log_rho1 = log(rho1);

    % -----------------------------------------------------------------
    % Annealing loop: converge at each eps level, warm-start next level
    % -----------------------------------------------------------------
    a        = log_rho0;        % initial guess: phi_0 = rho0
    b        = zeros(nx, 1);    % initial guess: psi_T = ones
    all_errors = [];
    total_iters = 0;
    converged   = false;

    for s = 1:numel(eps_sched)
        eps_s  = eps_sched(s);
        cfg_s  = cfg;   cfg_s.vareps = eps_s;
        heat_s = precomp_fn(problem, cfg_s);

        % Recompute log_psi0 for the new eps level (warm-start consistency)
        log_psi0 = heat_s.apply_full(b);

        level_converged = false;
        for iter = 1:max_iter
            % 1. Update a
            a = log_rho0 - log_psi0;

            % 2. Update b
            log_phi_T = heat_s.apply_full(a);
            b = log_rho1 - log_phi_T;

            % 3. Update log_psi0 for convergence check and next iteration
            log_psi0 = heat_s.apply_full(b);

            % 4. Convergence: left-marginal L2 error
            left_marg = exp(a + log_psi0);
            err = sqrt(dx) * norm(left_marg - rho0);
            all_errors(end+1) = err; %#ok<AGROW>
            total_iters = total_iters + 1;

            if tol > 0 && err < tol
                level_converged = true;
                break;
            end
        end

        % Final level: record convergence status
        if s == numel(eps_sched)
            converged = level_converged;
            heat      = heat_s;   % keep for trajectory recovery
        end
    end

    walltime = toc(t_start);

    % -----------------------------------------------------------------
    % Trajectory recovery using final-level heat kernel
    % -----------------------------------------------------------------
    phi_traj = zeros(nt+1, nx);
    psi_traj = zeros(nt+1, nx);

    for k = 0:nt
        t_k = k * dt;
        phi_traj(k+1, :) = exp(heat.apply_time(a, t_k))';
        psi_traj(k+1, :) = exp(heat.apply_time(b, 1 - t_k))';
    end

    rho_traj = phi_traj .* psi_traj;

    mx_traj = zeros(nt+1, nxm);
    for k = 1:(nt+1)
        phi_k     = phi_traj(k, :)';
        psi_k     = psi_traj(k, :)';
        phi_stag  = 0.5 * (phi_k(1:nxm) + phi_k(2:nx));
        dpsi_stag = (psi_k(2:nx) - psi_k(1:nxm)) / dx;
        mx_traj(k, :) = (2 * vareps * phi_stag .* dpsi_stag)';
    end

    result.rho          = rho_traj;
    result.mx           = mx_traj;
    result.phi          = phi_traj;
    result.psi          = psi_traj;
    result.t_grid       = t_grid;
    result.x_grid       = x_grid;
    result.x_stag       = x_stag;
    result.errors       = all_errors(:);
    result.iters        = total_iters;
    result.error        = all_errors(end);
    result.converged    = converged;
    result.walltime     = walltime;
    result.heat_name    = heat.name;
    result.log_phi0     = a;
    result.log_psiT     = b;
    result.eps_schedule = eps_sched;
end
