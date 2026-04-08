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
%   using a staggered-grid ADMM scheme.  The projection onto the
%   Fokker-Planck constraint is solved exactly via DCT in x followed by
%   independent tridiagonal (banded) solves in t for each spatial mode.
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
    sigma   = 1 / gamma;

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

    %% Banded projection precomputation  (done once at O(N) cost)
    %
    %  DCT in x decouples AA* into nx independent nt x nt tridiagonal systems:
    %
    %    T_k = M0 + lambda_x(k)*M1 + lambda_x(k)^2*M2,   k = 1,...,nx
    %
    %  where:
    %    lambda_x(k) = (2 - 2cos(pi*(k-1)*dx)) / dx^2   (Neumann eigenvalue)
    %    M0 = Dt_phi * Dt_rho                             (tridiag in t)
    %    M1 = diag(1 - eps/dt, 1, ..., 1, 1 + eps/dt)    (diagonal in t)
    %    M2 = eps^2 * It_phi * It_rho                     (tridiag in t)
    %
    %  T_k is SPD for k>=2 with kappa(T_k) = O(Nt^2/lambda_x(k)),
    %  independent of eps.  Each T_k is LU-factorised once; each projection
    %  call costs O(Nx Nt log Nx) for the DCT plus O(Nx Nt) for the solves.
    %
    %  Note: It*It' is DST-II diagonalisable (not DCT), so there is no
    %  separable 2-D transform that diagonalises AA* exactly.  The banded
    %  approach sidesteps this by transforming only in x.
    lambda_x = (2 - 2*cos(pi*dx*(0:nxm))) / dx^2;   % 1 x nx
    lambda_t = (2 - 2*cos(pi*dt*(0:ntm)')) / dt^2;  % nt x 1

    M0_bnd = -Dt_phi * Dt_rho;             % nt x nt, tridiagonal
    M1d    = ones(nt, 1);
    M1d(1) = 1 + vareps/dt;
    M1d(nt)= 1 - vareps/dt;
    M2_bnd = vareps^2 * It_phi * It_rho;  % nt x nt, tridiagonal

    % LU-factorize T_k for k=2..nx.  k=1 (lambda_x=0, T_1=M0) is singular
    % and handled separately via DCT-t (mass conservation makes RHS zero).
    Tk_L = cell(1, nx);
    Tk_U = cell(1, nx);
    Tk_P = cell(1, nx);
    for kk = 2:nx
        lxk = lambda_x(kk);
        Tk  = sparse(M0_bnd + diag(lxk * M1d) + lxk^2 * M2_bnd);
        [Tk_L{kk}, Tk_U{kk}, Tk_P{kk}] = lu(Tk);
    end

    %% Zero boundary arrays (reused throughout)
    zeros_x = zeros(nt, 1);

    %% Initialisation
    t  = linspace(0, 1, nt+1)';
    tt = t(2:end-1);

    rho = (1 - tt) .* rho0 + tt .* rho1;
    mx  = zeros(nt, nxm);

    rho_tilde = rho;
    mx_tilde  = mx;

    delta_rho = zeros(size(rho));
    delta_mx  = zeros(size(mx));

    %% ADMM iterations
    residual_diff = zeros(maxIter, 1);

    for iter = 1:maxIter

        % ---- Proximal step for the kinetic-energy term ----
        tmp_rho = interp_t_at_phi(rho_tilde + delta_rho/gamma, rho0, rho1);
        tmp_mx  = interp_x_at_phi(mx_tilde  + delta_mx /gamma, zeros_x, zeros_x);

        rho_new = solve_cubic(1, ...
                              2*sigma - tmp_rho, ...
                              sigma^2 - 2*sigma*tmp_rho, ...
                              -sigma*(sigma*tmp_rho + 0.5*tmp_mx.^2));
        mx_new  = rho_new .* tmp_mx ./ (rho_new + sigma);

        neg = rho_new <= 1e-14;
        rho_new(neg) = 0;
        mx_new(neg)  = 0;

        rho_new = interp_t_at_rho(rho_new);
        mx_new  = interp_x_at_m(mx_new);

        % ---- Projection onto the Fokker–Planck constraint set ----
        [rho_tilde_new, mx_tilde_new] = proj_div_banded( ...
            rho_new - delta_rho/gamma, mx_new - delta_mx/gamma);

        % ---- Convergence monitor ----
        residual_diff(iter) = sqrt(dt*dx * ( ...
            sum((rho_tilde_new(:) - rho_tilde(:)).^2) + ...
            sum((mx_tilde_new(:)  - mx_tilde(:)).^2)));

        if compute_true_error
            true_error(iter) = sqrt(dt*dx) * norm([rho(:) - rho_star(:); ...
                                                    mx(:)  - mx_star(:)]);
        end

        % ---- Dual update ----
        delta_rho = delta_rho - gamma * (rho_new - rho_tilde_new);
        delta_mx  = delta_mx  - gamma * (mx_new  - mx_tilde_new);

        rho       = rho_new;
        mx        = mx_new;
        rho_tilde = rho_tilde_new;
        mx_tilde  = mx_tilde_new;

        if mod(iter, 50) == 0
            fprintf('iter %d: max_rho=%.3e  norm_drho=%.3e  viol=%.3e\n', ...
                iter, max(rho_tilde(:)), norm(delta_rho(:)), ...
                calc_constraint_viol(rho_tilde, mx_tilde));
        end

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

    function out = interp_t_at_phi(in, bc_start, bc_end)
        out = It_phi * in;
        out(1,:)   = out(1,:)   + 0.5 * bc_start;
        out(end,:) = out(end,:) + 0.5 * bc_end;
    end
    function out = interp_t_at_rho(in),  out = It_rho * in;      end
    function out = interp_x_at_phi(in, bc_in, bc_out)
        out = in * Ix_phi';
        out(:,1)   = out(:,1)   + 0.5 * bc_in;
        out(:,end) = out(:,end) + 0.5 * bc_out;
    end
    function out = interp_x_at_m(in),    out = in * Ix_m';       end

    %% ================================================================
    %% Staggered-grid derivative operators
    %% ================================================================

    function out = deriv_t_at_phi(in, bc_start, bc_end)
        out = Dt_phi * in;
        out(1,:)   = out(1,:)   - bc_start / dt;
        out(end,:) = out(end,:) + bc_end   / dt;
    end
    function out = deriv_t_at_rho(in),   out = Dt_rho * in;      end
    function out = deriv_x_at_phi(in, bc_in, bc_out)
        out = in * Dx_phi';
        out(:,1)   = out(:,1)   - bc_in  / dx;
        out(:,end) = out(:,end) + bc_out / dx;
    end
    function out = deriv_x_at_m(in),     out = in * Dx_m';       end

    %% ================================================================
    %% Projection: DCT in x + tridiagonal solve in t
    %% ================================================================

    function [rho_out, m_out] = proj_div_banded(mu, psi)
        f = fp_res(mu, psi, rho0, rho1);
        if sqrt(dt*dx * sum(f(:).^2)) < 1e-12
            rho_out = mu;  m_out = psi;  return;
        end

        % DCT in x -> (nt x nx) in physical-time / x-DCT-mode domain
        f_hat_x = dct(f')';

        phi_hat_x = zeros(nt, nx);

        % k=1 (DC in x, lambda_x=0): T_1 = M0 (singular).
        % Invert via DCT-t; joint DC mode (l=1,k=1) is the gauge and zeroed.
        f1_hat_t          = dct(f_hat_x(:, 1));
        phi1_hat_t        = zeros(nt, 1);
        phi1_hat_t(2:end) = f1_hat_t(2:end) ./ lambda_t(2:end);
        phi_hat_x(:, 1)   = idct(phi1_hat_t);

        % k=2..nx: SPD tridiagonal solve with precomputed LU
        for kk = 2:nx
            phi_hat_x(:, kk) = Tk_U{kk} \ (Tk_L{kk} \ (Tk_P{kk} * f_hat_x(:, kk)));
        end

        % IDCT in x -> physical space
        phi = idct(phi_hat_x')';

        % Apply A* to get the projected (rho, m)
        dphi_dx    = deriv_x_at_m(phi);
        nablax_phi = interp_t_at_rho(deriv_x_at_phi(dphi_dx, zeros_x, zeros_x));
        rho_out    = mu  + deriv_t_at_rho(phi) + vareps * nablax_phi;
        m_out      = psi + dphi_dx;
    end

    %% ================================================================
    %% FP residual and post-processing metrics
    %% ================================================================

    function f = fp_res(rho_in, m_in, bc0, bc1)
        nablax_r = interp_t_at_phi(rho_in, bc0, bc1);
        nablax_r = deriv_x_at_phi(deriv_x_at_m(nablax_r), zeros_x, zeros_x);
        f = deriv_t_at_phi(rho_in, bc0, bc1) + ...
            deriv_x_at_phi(m_in, zeros_x, zeros_x) - vareps * nablax_r;
    end

    function cost = calc_cost(rho_in, mx_in)
        rho_phi = interp_t_at_phi(rho_in, rho0, rho1);
        mx_phi  = interp_x_at_phi(mx_in, zeros_x, zeros_x);
        integrand      = zeros(size(rho_phi));
        pos            = rho_phi > 1e-8;
        integrand(pos) = mx_phi(pos).^2 ./ rho_phi(pos);
        cost = sum(integrand(:)) * dt * dx;
    end

    function viol = calc_constraint_viol(rho_in, mx_in)
        dmu_dt     = deriv_t_at_phi(rho_in, rho0, rho1);
        dpsi_dx    = deriv_x_at_phi(mx_in, zeros_x, zeros_x);
        nablax_rho = interp_t_at_phi(rho_in, rho0, rho1);
        nablax_rho = deriv_x_at_phi(deriv_x_at_m(nablax_rho), zeros_x, zeros_x);
        residual   = dmu_dt + dpsi_dx - vareps * nablax_rho;
        viol = sqrt(dt * dx * sum(residual(:).^2));
    end

end
