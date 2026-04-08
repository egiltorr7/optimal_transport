function [rho, mx, outs] = sb1d_admm(rho0, rho1, opts)
%SB1D_ADMM  Solve the 1-D Schrödinger bridge problem via ADMM.
%
%   [RHO, MX, OUTS] = SB1D_ADMM(RHO0, RHO1, OPTS) finds the density RHO
%   (size (NT-1) x NX) and momentum MX (size NT x (NX-1)) that solve
%
%       min   int_0^1 int_0^1  |m|^2 / rho  dx dt
%       s.t.  d_t rho + d_x m - eps * Delta_x rho = 0
%             rho(0, .) = rho0,   rho(1, .) = rho1
%             m * n = 0  on {x = 0, x = 1}
%
%   using a staggered-grid ADMM scheme.
%
%   OPTS fields:
%     nt              - number of time intervals
%     nx              - number of space intervals  (= length(rho0))
%     maxIter         - maximum ADMM iterations
%     gamma           - ADMM penalty parameter
%     vareps          - Schrödinger regularisation (diffusion) coefficient
%     rho_star, mx_star  - (optional) reference solution for true-error tracking
%     postprocess     - (optional, default false) if true, compute the kinetic
%                       energy cost and Fokker–Planck constraint violation at
%                       every iteration (stored in outs.cost and outs.constraint_viol)

    %% Grid
    nt  = opts.nt;
    nx  = length(rho0);
    ntm = nt - 1;
    nxm = nx - 1;
    dx  = 1 / nx;
    dt  = 1 / nt;

    %% Parameters
    gamma   = opts.gamma;
    maxIter = opts.maxIter;
    vareps  = opts.vareps;
    sigma   = 1 / gamma;   % proximal step size (fixed throughout)

    %% Optional reference solution for true-error tracking
    compute_true_error = isfield(opts, 'rho_star') && isfield(opts, 'mx_star');
    if compute_true_error
        rho_star   = opts.rho_star;
        mx_star    = opts.mx_star;
        true_error = zeros(maxIter, 1);
    end

    %% Optional post-processing (cost + constraint violation)
    track_postprocess = isfield(opts, 'postprocess') && opts.postprocess;
    if track_postprocess
        cost_history    = zeros(maxIter, 1);
        constraint_viol = zeros(maxIter, 1);
    end

    %% Precompute staggered-grid operators
    %
    %  Grid layout (1-D space, time):
    %    phi-grid  : nt  rows (time cell-centres),  nx  cols (space cell-centres)
    %    rho-grid  : ntm rows (time faces/edges),   nx  cols
    %    mx-grid   : nt  rows,                      nxm cols (space faces/edges)
    %
    %  Naming convention: <op>_<result-grid>
    %    interp_t_at_phi  : rho-grid  -> phi-grid  (time interpolation)
    %    interp_t_at_rho  : phi-grid  -> rho-grid
    %    interp_x_at_phi  : mx-grid   -> phi-grid  (space interpolation)
    %    interp_x_at_m    : phi-grid  -> mx-grid
    %    deriv_t_at_phi   : rho-grid  -> phi-grid  (time finite difference)
    %    deriv_t_at_rho   : phi-grid  -> rho-grid
    %    deriv_x_at_phi   : mx-grid   -> phi-grid  (space finite difference / divergence)
    %    deriv_x_at_m     : phi-grid  -> mx-grid   (space finite difference / gradient)

    % Time interpolation
    It_phi = 0.5 * toeplitz([1 1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]);  % nt  x ntm
    It_rho = 0.5 * toeplitz([1 zeros(1,ntm-1)], [1 1 zeros(1,ntm-1)]); % ntm x nt

    % Space interpolation
    Ix_phi = 0.5 * toeplitz([1 1 zeros(1,nxm-1)], [1 zeros(1,nxm-1)]); % nx  x nxm
    Ix_m   = 0.5 * toeplitz([1 zeros(1,nxm-1)], [1 1 zeros(1,nxm-1)]); % nxm x nx

    % Time derivatives
    Dt_phi = toeplitz([1 -1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]) / dt;  % nt  x ntm
    Dt_rho = toeplitz([-1 zeros(1,ntm-1)], [-1 1 zeros(1,ntm-1)]) / dt; % ntm x nt

    % Space derivatives
    Dx_phi = toeplitz([1 -1 zeros(1,nxm-1)], [1 zeros(1,nxm-1)]) / dx;  % nx  x nxm
    Dx_m   = toeplitz([-1 zeros(1,nxm-1)], [-1 1 zeros(1,nxm-1)]) / dx; % nxm x nx

    %% DCT eigenvalues for the operator  -Delta_xt + eps^2 * Delta_x^2
    lambda_x          = (2 - 2*cos(pi*dx*(0:nxm))) / dx^2;   % 1 x nx
    lambda_t          = (2 - 2*cos(pi*dt*(0:ntm)')) / dt^2;  % nt x 1
    lambda_biharmonic = lambda_x + lambda_t + vareps^2 * lambda_x.^2;

    %% Zero boundary arrays (reused throughout)
    zeros_x = zeros(nt, 1);   % zero-flux BC at x = 0 and x = 1

    %% Initialisation
    t  = linspace(0, 1, nt+1)';
    tt = t(2:end-1);

    % Primal variables (rho on rho-grid, mx on mx-grid)
    rho = (1 - tt) .* rho0 + tt .* rho1;   % linear interpolation in time
    mx  = zeros(nt, nxm);

    % Consensus variables (projected copies that satisfy the constraint)
    rho_tilde = rho;
    mx_tilde  = mx;

    % Dual variables (scaled Lagrange multipliers)
    delta_rho = zeros(size(rho));
    delta_mx  = zeros(size(mx));

    %% ADMM iterations
    residual_diff = zeros(maxIter, 1);

    for iter = 1:maxIter

        % ---- Proximal step for the kinetic-energy term ----
        % Add dual variable and interpolate consensus copies to cell-centres
        tmp_rho = interp_t_at_phi(rho_tilde + delta_rho/gamma, rho0, rho1);
        tmp_mx  = interp_x_at_phi(mx_tilde  + delta_mx /gamma, zeros_x, zeros_x);

        % Solve the proximal subproblem element-wise via Cardano's formula
        rho_new = solve_cubic(1, ...
                              2*sigma - tmp_rho, ...
                              sigma^2 - 2*sigma*tmp_rho, ...
                              -sigma*(sigma*tmp_rho + 0.5*tmp_mx.^2));
        mx_new  = rho_new .* tmp_mx ./ (rho_new + sigma);

        % Enforce non-negativity
        neg = rho_new <= 1e-12;
        rho_new(neg) = 0;
        mx_new(neg)  = 0;

        % Interpolate back to staggered locations
        rho_new = interp_t_at_rho(rho_new);
        mx_new  = interp_x_at_m(mx_new);

        % ---- Projection onto the Fokker–Planck constraint set ----
        [rho_tilde_new, mx_tilde_new] = proj_div_biharmonic( ...
            rho_new - delta_rho/gamma, mx_new - delta_mx/gamma);

        % ---- Convergence monitor ----
        residual_diff(iter) = sqrt(dt*dx * ( ...
            sum((rho_tilde_new(:) - rho_tilde(:)).^2) + ...
            sum((mx_tilde_new(:)  - mx_tilde(:)).^2)));

        if compute_true_error
            true_error(iter) = sqrt(dt*dx) * norm([rho(:) - rho_star(:); ...
                                                    mx(:)  - mx_star(:)]);
        end

        % ---- Dual update (Arrow–Hurwicz / extra-gradient) ----
        delta_rho = delta_rho - gamma * (rho_new - rho_tilde_new);
        delta_mx  = delta_mx  - gamma * (mx_new  - mx_tilde_new);

        % ---- Variable update ----
        rho       = rho_new;
        mx        = mx_new;
        rho_tilde = rho_tilde_new;
        mx_tilde  = mx_tilde_new;

        % ---- Post-processing (on updated primal variables) ----
        if track_postprocess
            cost_history(iter)    = calc_cost(rho, mx);
            constraint_viol(iter) = calc_constraint_viol(rho, mx);
        end

    end

    %% Output
    outs.residual_diff = residual_diff;
    if compute_true_error
        outs.true_error = true_error;
    end
    if track_postprocess
        outs.cost            = cost_history;
        outs.constraint_viol = constraint_viol;
    end


    %% ================================================================
    %% Staggered-grid interpolation operators
    %% ================================================================

    % rho-grid -> phi-grid, with Dirichlet time BCs (rho0 at t=0, rho1 at t=1)
    function out = interp_t_at_phi(in, bc_start, bc_end)
        out = It_phi * in;
        out(1,:)   = out(1,:)   + 0.5 * bc_start;
        out(end,:) = out(end,:) + 0.5 * bc_end;
    end

    % phi-grid -> rho-grid (no boundary terms needed)
    function out = interp_t_at_rho(in)
        out = It_rho * in;
    end

    % mx-grid -> phi-grid, with zero-flux BCs at x = 0 and x = 1
    function out = interp_x_at_phi(in, bc_in, bc_out)
        out = in * Ix_phi';
        out(:,1)   = out(:,1)   + 0.5 * bc_in;
        out(:,end) = out(:,end) + 0.5 * bc_out;
    end

    % phi-grid -> mx-grid (no boundary terms needed)
    function out = interp_x_at_m(in)
        out = in * Ix_m';
    end

    %% ================================================================
    %% Staggered-grid derivative operators
    %% ================================================================

    % rho-grid -> phi-grid, with time BCs
    function out = deriv_t_at_phi(in, bc_start, bc_end)
        out = Dt_phi * in;
        out(1,:)   = out(1,:)   - bc_start / dt;
        out(end,:) = out(end,:) + bc_end   / dt;
    end

    % phi-grid -> rho-grid
    function out = deriv_t_at_rho(in)
        out = Dt_rho * in;
    end

    % mx-grid -> phi-grid (divergence in x), with zero-flux BCs
    function out = deriv_x_at_phi(in, bc_in, bc_out)
        out = in * Dx_phi';
        out(:,1)   = out(:,1)   - bc_in  / dx;
        out(:,end) = out(:,end) + bc_out / dx;
    end

    % phi-grid -> mx-grid (gradient in x)
    function out = deriv_x_at_m(in)
        out = in * Dx_m';
    end

    %% ================================================================
    %% Projection onto the Fokker–Planck constraint
    %% ================================================================

    % Invert  -(Delta_xt + eps^2 * Delta_x^2) phi = f
    % with homogeneous Neumann BCs, using the 2-D DCT.
    function phi = invert_biharmonic(f)
        phi_hat      = mirt_dctn(f);
        phi_hat      = phi_hat ./ lambda_biharmonic;
        phi_hat(1,1) = 0;          % zero mean (gauge fix)
        phi          = mirt_idctn(phi_hat);
    end

    %% ================================================================
    %% Post-processing metrics
    %% ================================================================

    % Kinetic energy:  int_0^1 int_0^1  |m|^2 / rho  dx dt
    %
    % Convention (Benamou–Brenier):
    %   f(rho, m) = |m|^2 / rho   if rho > 0
    %             = 0              if rho = 0  and  m = 0
    %             = +Inf           if rho = 0  and  m ~= 0
    function cost = calc_cost(rho_in, mx_in)
        rho_phi = interp_t_at_phi(rho_in, rho0, rho1);
        mx_phi  = interp_x_at_phi(mx_in, zeros_x, zeros_x);

        integrand      = zeros(size(rho_phi));
        pos            = rho_phi > 1e-8;
        integrand(pos) = mx_phi(pos).^2 ./ rho_phi(pos);

        % % If rho = 0 but m ~= 0, report +Inf (physically correct)
        % if any(~pos(:) & abs(mx_phi(:)) > 1e-12)
        %     cost = Inf;
        %     return;
        % end

        cost = sum(integrand(:)) * dt * dx;
    end

    % Fokker–Planck constraint violation (L^2 norm of the residual):
    %   || d_t rho + d_x m - eps * Delta_x rho ||_{L^2(Omega)}
    function viol = calc_constraint_viol(rho_in, mx_in)
        zeros_xx = zeros_x(2:end);   % ntm-row zero column for rho-grid BCs

        dmu_dt  = deriv_t_at_phi(rho_in, rho0, rho1);
        dpsi_dx = deriv_x_at_phi(mx_in, zeros_x, zeros_x);

        % Discrete Laplacian of rho_in in x, lifted to phi-grid
        nablax_rho  = deriv_x_at_phi(deriv_x_at_m(rho_in), zeros_xx, zeros_xx);
        nablax_rho0 = deriv_x_at_phi(deriv_x_at_m(rho0), 0, 0);
        nablax_rho1 = deriv_x_at_phi(deriv_x_at_m(rho1), 0, 0);
        nablax_rho  = interp_t_at_phi(nablax_rho, nablax_rho0, nablax_rho1);

        residual = dmu_dt + dpsi_dx - vareps * nablax_rho;
        viol = sqrt(dt * dx * sum(residual(:).^2));
    end

    % Project (mu, psi) onto the set satisfying the Fokker–Planck equation
    %   d_t mu + d_x psi - eps * Delta_x mu = 0
    % with BCs  mu(0,.) = rho0,  mu(1,.) = rho1,  psi * n = 0.
    % The correction is (mu, psi) <- (mu, psi) - (d_t phi, d_x phi),
    % where phi solves the regularised Poisson problem above.
    function [rho_out, m_out] = proj_div_biharmonic(mu, psi)

        zeros_xx = zeros_x(2:end);   % zero-flux BC for the ntm-row rho-grid

        % Compute the divergence residual: d_t mu + d_x psi - eps * Delta_x mu
        dmu_dt  = deriv_t_at_phi(mu, rho0, rho1);
        dpsi_dx = deriv_x_at_phi(psi, zeros_x, zeros_x);

        % Discrete Laplacian of mu in x (gradient then divergence)
        nablax_mu   = deriv_x_at_phi(deriv_x_at_m(mu),   zeros_xx, zeros_xx);
        nablax_rho0 = deriv_x_at_phi(deriv_x_at_m(rho0), 0, 0);
        nablax_rho1 = deriv_x_at_phi(deriv_x_at_m(rho1), 0, 0);
        nablax_mu   = interp_t_at_phi(nablax_mu, nablax_rho0, nablax_rho1);

        % Solve for the correction potential phi
        phi = invert_biharmonic(dmu_dt + dpsi_dx - vareps * nablax_mu);

        % Apply correction: subtract grad phi from (mu, psi)
        dphi_dx    = deriv_x_at_m(phi);
        nablax_phi = interp_t_at_rho(deriv_x_at_phi(dphi_dx, zeros_x, zeros_x));

        rho_out = mu  + deriv_t_at_rho(phi) + vareps * nablax_phi;
        m_out   = psi + dphi_dx;

        % Debug
        dmu_dt = deriv_t_at_phi(rho_out, rho0, rho1);
        dpsi_dx = deriv_x_at_phi(m_out, zeros_x, zeros_x);
        nablax_mu   = deriv_x_at_phi(deriv_x_at_m(rho_out),   zeros_xx, zeros_xx);
        nablax_rho0 = deriv_x_at_phi(deriv_x_at_m(rho0), 0, 0);
        nablax_rho1 = deriv_x_at_phi(deriv_x_at_m(rho1), 0, 0);
        nablax_mu   = interp_t_at_phi(nablax_mu, nablax_rho0, nablax_rho1);

        max(max(abs(dmu_dt + dpsi_dx - vareps*nablax_mu)))/(nt*nx)

        

    end

end
