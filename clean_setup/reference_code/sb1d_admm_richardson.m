function [rho, mx, outs] = sb1d_admm_richardson(rho0, rho1, opts)
%SB1D_ADMM_RICHARDSON  Schrödinger bridge via ADMM with Richardson projection.
%
%   Backup of the working Richardson-iteration version of sb1d_admm.
%   The projection step uses repeated preconditioned Richardson steps
%   (DCT preconditioner) rather than PCG.  For small-to-moderate eps
%   this converges in a handful of inner iterations; for large eps more
%   iterations are needed (see sb1d_admm.m for the PCG version).

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

    %% DCT eigenvalues
    lambda_x        = (2 - 2*cos(pi*dx*(0:nxm))) / dx^2;   % 1 x nx
    lambda_t        = (2 - 2*cos(pi*dt*(0:ntm)')) / dt^2;  % nt x 1
    lambda_x_mat    = repmat(lambda_x, nt, 1);
    lambda_t_mat    = repmat(lambda_t, 1, nx);

    sigma_t     = cos(pi * dt * (0:ntm)' / 2).^2;          % nt x 1
    sigma_t_mat = repmat(sigma_t, 1, nx);

    lambda_lap      = lambda_x_mat + lambda_t_mat;
    lambda_IpEpsLap = ones(size(lambda_x_mat)) + ...
                      vareps^2 * lambda_x_mat.^2 .* sigma_t_mat ./ lambda_lap;
    lambda_IpEpsLap(1,1) = 1.0;

    %% Zero boundary arrays
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

        % Proximal step
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

        % Projection (Richardson)
        [rho_tilde_new, mx_tilde_new] = proj_div_richardson( ...
            rho_new - delta_rho/gamma, mx_new - delta_mx/gamma);

        % Convergence monitor
        residual_diff(iter) = sqrt(dt*dx * ( ...
            sum((rho_tilde_new(:) - rho_tilde(:)).^2) + ...
            sum((mx_tilde_new(:)  - mx_tilde(:)).^2)));

        if compute_true_error
            true_error(iter) = sqrt(dt*dx) * norm([rho(:) - rho_star(:); ...
                                                    mx(:)  - mx_star(:)]);
        end

        % Dual update
        delta_rho = delta_rho - gamma * (rho_new - rho_tilde_new);
        delta_mx  = delta_mx  - gamma * (mx_new  - mx_tilde_new);

        rho       = rho_new;
        mx        = mx_new;
        rho_tilde = rho_tilde_new;
        mx_tilde  = mx_tilde_new;

        if mod(iter,50)==0
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
    if compute_true_error,    outs.true_error      = true_error;      end
    if track_postprocess
        outs.cost            = cost_history;
        outs.constraint_viol = constraint_viol;
    end


    %% ================================================================
    %% Staggered-grid operators
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
    function out = interp_x_at_m(in),   out = in * Ix_m';        end

    function out = deriv_t_at_phi(in, bc_start, bc_end)
        out = Dt_phi * in;
        out(1,:)   = out(1,:)   - bc_start / dt;
        out(end,:) = out(end,:) + bc_end   / dt;
    end
    function out = deriv_t_at_rho(in),  out = Dt_rho * in;       end
    function out = deriv_x_at_phi(in, bc_in, bc_out)
        out = in * Dx_phi';
        out(:,1)   = out(:,1)   - bc_in  / dx;
        out(:,end) = out(:,end) + bc_out / dx;
    end
    function out = deriv_x_at_m(in),    out = in * Dx_m';        end

    %% ================================================================
    %% DCT preconditioner
    %% ================================================================

    function phi = invert_biharmonic(f)
        phi_hat      = mirt_dctn(f);
        phi_hat      = phi_hat ./ lambda_lap;
        phi_hat(1,1) = 0;
        phi_hat      = phi_hat ./ lambda_IpEpsLap;
        phi          = mirt_idctn(phi_hat);
    end

    %% ================================================================
    %% Projection (Richardson iteration)
    %% ================================================================

    function [rho_out, m_out] = proj_div_richardson(mu, psi)
        rho_out = mu;
        m_out   = psi;
        for k_inner = 1:50
            dmu_dt     = deriv_t_at_phi(rho_out, rho0, rho1);
            dpsi_dx    = deriv_x_at_phi(m_out, zeros_x, zeros_x);
            nablax_out = interp_t_at_phi(rho_out, rho0, rho1);
            nablax_out = deriv_x_at_phi(deriv_x_at_m(nablax_out), zeros_x, zeros_x);
            f = dmu_dt + dpsi_dx + vareps * nablax_out;
            if sqrt(dt * dx * sum(f(:).^2)) < 1e-12, break; end
            phi        = invert_biharmonic(f);
            dphi_dx    = deriv_x_at_m(phi);
            nablax_phi = interp_t_at_rho(deriv_x_at_phi(dphi_dx, zeros_x, zeros_x));
            rho_out = rho_out + deriv_t_at_rho(phi) - vareps * nablax_phi;
            m_out   = m_out   + dphi_dx;
        end
    end

    %% ================================================================
    %% Post-processing metrics
    %% ================================================================

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
        residual   = dmu_dt + dpsi_dx + vareps * nablax_rho;
        viol = sqrt(dt * dx * sum(residual(:).^2));
    end

end
