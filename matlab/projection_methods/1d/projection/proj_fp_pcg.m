function x_out = proj_fp_pcg(x_in, problem, cfg)
% PROJ_FP_PCG  Project onto the Fokker-Planck constraint via PCG.
%
%   x_out = proj_fp_pcg(x_in, problem, cfg)
%
%   Solves the normal equations  AA* phi = f  using preconditioned conjugate
%   gradient (PCG), where:
%     - A*(phi) = ( d_t^T phi + eps*d_xx^T phi,  d_x^T phi )
%     - A A*(phi) = d_t d_t^T phi + d_x d_x^T phi - eps*d_xx ... (see below)
%     - f = d_t mu + d_x psi - eps * d_xx mu  (FP residual)
%
%   Preconditioner: separable DCT solve, same as proj_fokker_planck.
%   Eigenvalues:  P_{k,l} = lambda_x(k) + lambda_t(l) + eps^2 * lambda_x(k)^2
%   This is exact for eps=0; for eps>0 it ignores boundary corrections
%   in M1d that scale as eps/dt, so convergence degrades as eps*nt grows.
%
%   Inputs:
%     x_in.rho  (ntm x nx)   point to project, density   (staggered)
%     x_in.mx   (nt  x nxm)  point to project, momentum  (staggered)
%     problem                 problem struct (needs lambda_x, lambda_t, ops, ...)
%     cfg                     config struct; uses:
%                               cfg.vareps        regularization
%                               cfg.proj_tol      PCG stopping tolerance (default 1e-6)
%                               cfg.proj_max_iter PCG max iterations     (default 50)
%
%   Output:
%     x_out.rho  (ntm x nx)
%     x_out.mx   (nt  x nxm)

    ops    = problem.ops;
    rho0   = problem.rho0;
    rho1   = problem.rho1;
    nt     = problem.nt;
    vareps = cfg.vareps;

    proj_tol      = 1e-6;
    proj_max_iter = 50;
    if isfield(cfg, 'proj_tol'),      proj_tol      = cfg.proj_tol;      end
    if isfield(cfg, 'proj_max_iter'), proj_max_iter = cfg.proj_max_iter; end

    zeros_x = zeros(nt, 1);

    mu  = x_in.rho;   % (ntm x nx)
    psi = x_in.mx;    % (nt  x nxm)

    %% --- RHS: FP residual  f = d_t mu + d_x psi - eps * d_xx mu  (nt x nx) ---
    laplacian_mu = ops.deriv_x_at_phi( ...
                       ops.deriv_x_at_m( ...
                           ops.interp_t_at_phi(mu, rho0, rho1)), ...
                       zeros_x, zeros_x);

    f = ops.deriv_t_at_phi(mu, rho0, rho1) ...
      + ops.deriv_x_at_phi(psi, zeros_x, zeros_x) ...
      - vareps * laplacian_mu;

    if norm(f(:)) * sqrt(problem.dt * problem.dx) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- PCG to solve AA* phi = f ---
    % Corrections from phi use zero temporal BCs (homogeneous problem).
    phi = pcg_solve(f, problem, cfg, ops, vareps, zeros_x, ...
                    proj_tol, proj_max_iter);

    %% --- Apply A* correction ---
    dphi_dx    = ops.deriv_x_at_m(phi);
    nablax_phi = ops.interp_t_at_rho( ...
                     ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x));

    x_out.rho = mu  + ops.deriv_t_at_rho(phi) + vareps * nablax_phi;
    x_out.mx  = psi + dphi_dx;
end

% ---------------------------------------------------------------------------

function phi = pcg_solve(f, problem, cfg, ops, vareps, zeros_x, tol, max_iter)
% PCG_SOLVE  Solve  AA* phi = f  by preconditioned CG.
%
%   Preconditioner P^{-1}: separable DCT solve (proj_fokker_planck eigenvalues).
%   AA* is self-adjoint positive (semi-)definite on the (nt x nx) phi-grid.

    % Flatten to vectors for CG arithmetic
    n    = numel(f);
    fvec = f(:);

    % Initial guess: zero (or preconditioner applied to f)
    phi_vec = apply_Pinv(fvec, problem, vareps);

    r   = fvec - apply_AAt(phi_vec, problem, ops, vareps, zeros_x);
    z   = apply_Pinv(r, problem, vareps);
    p   = z;
    rz  = r(:)' * z(:);

    for iter = 1:max_iter
        Ap  = apply_AAt(p, problem, ops, vareps, zeros_x);
        pAp = p(:)' * Ap(:);

        if pAp <= 0
            break;   % loss of positive definiteness (shouldn't happen)
        end

        alpha   = rz / pAp;
        phi_vec = phi_vec + alpha * p;
        r       = r - alpha * Ap;

        res_norm = norm(r(:)) * sqrt(problem.dt * problem.dx);
        if res_norm < tol * norm(fvec(:)) * sqrt(problem.dt * problem.dx)
            break;
        end

        z      = apply_Pinv(r, problem, vareps);
        rz_new = r(:)' * z(:);
        beta   = rz_new / rz;
        p      = z + beta * p;
        rz     = rz_new;
    end

    phi = reshape(phi_vec, size(f));
end

% ---------------------------------------------------------------------------

function Ap = apply_AAt(p_vec, problem, ops, vareps, zeros_x)
% APPLY_AAT  Compute AA* p where p is on the (nt x nx) phi-grid.
%
%   A*(phi) = ( d_t^T phi + eps * d_xx^T phi,   d_x^T phi )
%   A(mu, psi) = d_t mu + d_x psi - eps * d_xx mu
%
%   Both operations use zero temporal/spatial BCs (homogeneous corrections).

    rho0_zero = zeros(1, problem.nx);
    rho1_zero = zeros(1, problem.nx);

    phi = reshape(p_vec, problem.nt, problem.nx);

    % A*(phi): back-project to staggered variables
    % mu_corr  = d_t^T phi + eps * d_xx^T phi  (ntm x nx)
    % psi_corr = d_x^T phi                     (nt  x nxm)
    dphi_dx    = ops.deriv_x_at_m(phi);                                    % (nt x nxm)
    nablax_phi = ops.interp_t_at_rho( ...
                     ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x));       % (ntm x nx)

    mu_corr  = ops.deriv_t_at_rho(phi) + vareps * nablax_phi;              % (ntm x nx)
    psi_corr = dphi_dx;                                                     % (nt  x nxm)

    % A(mu_corr, psi_corr): compute FP residual with zero BCs
    lap_mu_corr = ops.deriv_x_at_phi( ...
                      ops.deriv_x_at_m( ...
                          ops.interp_t_at_phi(mu_corr, rho0_zero, rho1_zero)), ...
                      zeros_x, zeros_x);

    Ap_grid = ops.deriv_t_at_phi(mu_corr, rho0_zero, rho1_zero) ...
            + ops.deriv_x_at_phi(psi_corr, zeros_x, zeros_x) ...
            - vareps * lap_mu_corr;

    Ap = Ap_grid(:);
end

% ---------------------------------------------------------------------------

function z = apply_Pinv(r_vec, problem, vareps)
% APPLY_PINV  Apply the separable DCT preconditioner P^{-1}.
%
%   P corresponds to eigenvalues lambda_x + lambda_t + eps^2 * lambda_x^2,
%   which is the exact AA* operator only when boundary corrections vanish
%   (eps=0 or the DC mode in t).

    r   = reshape(r_vec, problem.nt, problem.nx);
    lam = problem.lambda_x + problem.lambda_t + (vareps^2) .* problem.lambda_x.^2;

    z_hat      = mirt_dctn(r) ./ lam;
    z_hat(1,1) = 0;                   % fix gauge
    z          = mirt_idctn(z_hat);
    z          = z(:);
end
