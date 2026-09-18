function [rho, mx, outs] = sb1d_admm_pcg(rho0, rho1, opts)
%SB1D_ADMM_PCG  Schrödinger bridge via ADMM with PCG projection.
%
%   Solves the same problem as sb1d_admm.m but uses preconditioned
%   conjugate gradient (PCG) to solve the projection linear system
%
%       AA* phi = f
%
%   at each ADMM iteration instead of the exact Woodbury / banded solve.
%
%   The preconditioner is the DCT-based approximate inverse (same as in
%   sb1d_admm_richardson.m), which diagonalises AA* approximately by
%   ignoring boundary correction terms.
%
%   OPTS fields (in addition to the standard ones):
%     pcg_maxIter  - max inner PCG iterations        (default 30)
%     pcg_tol      - inner PCG relative residual tol (default 1e-8)
%     pcg_verbose  - print inner CG residuals        (default false)

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

    pcg_maxIter = 30;
    pcg_tol     = 1e-8;
    pcg_verbose = false;
    if isfield(opts, 'pcg_maxIter'), pcg_maxIter = opts.pcg_maxIter; end
    if isfield(opts, 'pcg_tol'),     pcg_tol     = opts.pcg_tol;     end
    if isfield(opts, 'pcg_verbose'), pcg_verbose = opts.pcg_verbose; end

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

    %% DCT eigenvalues for the preconditioner
    %
    %  The preconditioner approximates AA* by ignoring boundary correction
    %  terms, making it block-diagonal in the 2-D DCT basis:
    %
    %    P_approx  <->  lambda_lap * lambda_IpEpsLap
    %
    %  where:
    %    lambda_lap      = lambda_x + lambda_t          (Laplacian eigenvalues)
    %    lambda_IpEpsLap = 1 + eps^2 * lambda_x^2 *
    %                          sigma_t / lambda_lap      (correction factor)
    %
    %  This is the same preconditioner used in sb1d_admm_richardson.m.
    lambda_x        = (2 - 2*cos(pi*dx*(0:nxm))) / dx^2;   % 1 x nx
    lambda_t        = (2 - 2*cos(pi*dt*(0:ntm)')) / dt^2;  % nt x 1
    lambda_x_mat    = repmat(lambda_x, nt, 1);   % nt x nx
    lambda_t_mat    = repmat(lambda_t, 1, nx);   % nt x nx

    sigma_t     = cos(pi * dt * (0:ntm)' / 2).^2;   % nt x 1
    sigma_t_mat = repmat(sigma_t, 1, nx);

    lambda_lap      = lambda_x_mat + lambda_t_mat;
    lambda_IpEpsLap = ones(size(lambda_x_mat)) + ...
                      vareps^2 * lambda_x_mat.^2 .* sigma_t_mat ./ lambda_lap;
    lambda_IpEpsLap(1,1) = 1.0;

    %% Zero boundary arrays
    zeros_x  = zeros(nt,  1);   % zero-flux BC on phi-grid rows
    zeros_nx = zeros(1,  nx);   % zero time BCs for homogeneous A apply

    %% Initialisation
    t  = linspace(0, 1, nt+1)';
    tt = t(2:end-1);

    rho = (1 - tt) .* rho0 + tt .* rho1;
    mx  = zeros(nt, nxm);

    rho_tilde = rho;
    mx_tilde  = mx;

    delta_rho = zeros(size(rho));
    delta_mx  = zeros(size(mx));

    %% Diagnostics
    residual_diff   = zeros(maxIter, 1);
    pcg_iters_hist  = zeros(maxIter, 1);   % inner CG iteration counts
    pcg_res_hist    = zeros(maxIter, 1);   % final inner relative residual

    %% ADMM iterations
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

        % ---- Projection onto the Fokker–Planck constraint via PCG ----
        [rho_tilde_new, mx_tilde_new, pcg_k, pcg_r] = proj_div_pcg( ...
            rho_new - delta_rho/gamma, mx_new - delta_mx/gamma);
        pcg_iters_hist(iter) = pcg_k;
        pcg_res_hist(iter)   = pcg_r;

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

        if mod(iter,50)==0
            fprintf('iter %d: max_rho=%.3e  norm_drho=%.3e  viol=%.3e  pcg_iters=%d  pcg_res=%.2e\n', ...
                iter, max(rho_tilde(:)), norm(delta_rho(:)), ...
                calc_constraint_viol(rho_tilde, mx_tilde), ...
                pcg_iters_hist(iter), pcg_res_hist(iter));
        end

        if track_postprocess
            cost_history(iter)    = calc_cost(rho, mx);
            constraint_viol(iter) = calc_constraint_viol(rho, mx);
        end

    end

    %% Output
    outs.residual_diff  = residual_diff;
    outs.pcg_iters      = pcg_iters_hist;
    outs.pcg_res        = pcg_res_hist;
    if compute_true_error,  outs.true_error       = true_error;       end
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
    function out = interp_x_at_m(in),    out = in * Ix_m';       end

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
    %% FP residual  A(rho_in, m_in)  with prescribed time BCs
    %% ================================================================

    function f = fp_res(rho_in, m_in, bc0, bc1)
        nablax_r = interp_t_at_phi(rho_in, bc0, bc1);
        nablax_r = deriv_x_at_phi(deriv_x_at_m(nablax_r), zeros_x, zeros_x);
        f = deriv_t_at_phi(rho_in, bc0, bc1) + ...
            deriv_x_at_phi(m_in, zeros_x, zeros_x) + vareps * nablax_r;
    end

    %% ================================================================
    %% AA* matvec: phi (nt x nx) -> AA* phi (nt x nx)
    %
    %  This is the EXACT operator, including boundary correction terms.
    %  It is the composition  A o A*  with homogeneous time BCs (bc=0)
    %  because phi lives in the potential space (no prescribed BCs).
    %% ================================================================

    function out = apply_AA_star(phi)
        % A* phi: compute (rho_correction, m_correction)
        dphi_dx        = deriv_x_at_m(phi);
        nablax_phi     = interp_t_at_rho( ...
                             deriv_x_at_phi(dphi_dx, zeros_x, zeros_x));
        rho_corr       = deriv_t_at_rho(phi) - vareps * nablax_phi;
        m_corr         = dphi_dx;

        % A (rho_corr, m_corr): apply FP operator with zero time BCs
        out = fp_res(rho_corr, m_corr, zeros_nx', zeros_nx');
    end

    %% ================================================================
    %% DCT preconditioner  M^{-1} r
    %% ================================================================

    function phi = precond_solve(r)
        r_hat        = mirt_dctn(r);
        r_hat        = r_hat ./ lambda_lap;
        r_hat(1,1)   = 0;
        r_hat        = r_hat ./ lambda_IpEpsLap;
        phi          = mirt_idctn(r_hat);
    end

    %% ================================================================
    %% Projection onto the Fokker–Planck constraint via PCG
    %
    %  Solves  AA* phi = f  (f = FP residual of the input),
    %  then applies the correction  (mu, psi) - A* phi.
    %
    %  Returns: corrected (rho_out, m_out), number of CG iterations,
    %           and final relative residual.
    %% ================================================================

    function [rho_out, m_out, k_out, rel_res] = proj_div_pcg(mu, psi)
        f      = fp_res(mu, psi, rho0, rho1);
        f_norm = sqrt(dt * dx * sum(f(:).^2));

        if f_norm < 1e-12
            rho_out = mu;  m_out = psi;  k_out = 0;  rel_res = 0;
            return;
        end

        % Preconditioned CG for  AA* phi = f,  starting from phi = 0
        phi = zeros(nt, nx);
        r   = f;                     % r = f - AA*0 = f
        z   = precond_solve(r);      % z = M^{-1} r
        p   = z;
        rz  = dt * dx * sum(r(:) .* z(:));   % <r, z>  in L^2 sense

        k_out   = 0;
        rel_res = 1;

        for k = 1:pcg_maxIter
            Ap    = apply_AA_star(p);
            pAp   = dt * dx * sum(p(:) .* Ap(:));

            if pAp <= 0
                % Loss of positive definiteness — operator not SPD in CG sense.
                % Fall back: accept current phi and break.
                warning('sb1d_admm_pcg: CG loss of positive definiteness at iter %d, inner %d', ...
                        iter, k);
                break;
            end

            alpha = rz / pAp;
            phi   = phi + alpha * p;
            r     = r   - alpha * Ap;

            r_norm  = sqrt(dt * dx * sum(r(:).^2));
            rel_res = r_norm / f_norm;

            if pcg_verbose
                fprintf('  CG %d: rel_res = %.4e\n', k, rel_res);
            end

            if rel_res < pcg_tol
                k_out = k;
                break;
            end

            z      = precond_solve(r);
            rz_new = dt * dx * sum(r(:) .* z(:));
            beta   = rz_new / rz;
            p      = z + beta * p;
            rz     = rz_new;
            k_out  = k;
        end

        % Apply the computed potential to project (mu, psi)
        dphi_dx    = deriv_x_at_m(phi);
        nablax_phi = interp_t_at_rho( ...
                         deriv_x_at_phi(dphi_dx, zeros_x, zeros_x));
        rho_out    = mu  + deriv_t_at_rho(phi) - vareps * nablax_phi;
        m_out      = psi + dphi_dx;
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
