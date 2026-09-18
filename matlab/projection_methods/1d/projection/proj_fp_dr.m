function x_out = proj_fp_dr(x_in, problem, cfg)
% PROJ_FP_DR  Project onto the Fokker-Planck constraint via Douglas-Rachford.
%
%   x_out = proj_fp_dr(x_in, problem, cfg)
%
%   Splits the FP equation  d_t rho + d_x m = eps * d_xx rho  by introducing
%   the effective flux  j = m - eps * d_x(interp_t(rho)):
%
%     C1 = { (rho, m, j) :  d_t rho + d_x j = 0 }          (pure continuity)
%     C2 = { (rho, m, j) :  j = m - eps * d_x(interp_t(rho)) }   (flux definition)
%
%   FP is equivalent to C1 ∩ C2.  DR alternates projections:
%
%     proj C1 : eps=0 continuity projection treating j as the momentum.
%               2D DCT solve, eigenvalues lambda_x + lambda_t.
%               m is unchanged (not constrained by C1).
%
%     proj C2 : Helmholtz solve in x, one per time slice.
%               KKT gives  (2I - eps^2 * d_xx) lam = r  (decoupled in t).
%               1D DCT in x, eigenvalues 2 + eps^2 * lambda_x.
%               Updates (rho, m, j) via the multiplier lam.
%
%   Inputs:
%     x_in.rho  (ntm x nx)   staggered density
%     x_in.mx   (nt  x nxm)  staggered momentum
%     problem               problem struct (needs ops, lambda_x, lambda_t, ...)
%     cfg                   config struct; uses:
%                             cfg.vareps        regularization
%                             cfg.proj_tol      DR stopping tolerance (default 1e-6)
%                             cfg.proj_max_iter DR max outer iterations (default 30)
%
%   Output:
%     x_out.rho  (ntm x nx)
%     x_out.mx   (nt  x nxm)

    ops    = problem.ops;
    rho0   = problem.rho0;
    rho1   = problem.rho1;
    nt     = problem.nt;
    vareps = cfg.vareps;

    dr_tol      = 1e-6;
    dr_max_iter = 30;
    if isfield(cfg, 'proj_tol'),      dr_tol      = cfg.proj_tol;      end
    if isfield(cfg, 'proj_max_iter'), dr_max_iter = cfg.proj_max_iter; end

    zeros_x = zeros(nt, 1);

    mu  = x_in.rho;   % (ntm x nx)
    psi = x_in.mx;    % (nt  x nxm)

    %% --- Initialize j as the effective flux from the given (rho, m) ---
    j = psi - vareps * ops.deriv_x_at_m(ops.interp_t_at_phi(mu, rho0, rho1));

    % DR state z = (rho, m, j) — all three components
    z_rho = mu;
    z_m   = psi;
    z_j   = j;

    % y will hold the most recent proj_C1 iterate (which lies in C1)
    y_rho = mu;
    y_m   = psi;

    %% --- Douglas-Rachford iteration ---
    for iter = 1:dr_max_iter

        %% Step 1: y = proj_C1(z)
        %  Enforce  d_t rho + d_x j = 0  (m unchanged).
        %  This is the eps=0 continuity projection with j playing the role of m.
        [y_rho, y_j] = proj_c1(z_rho, z_j, problem, ops, rho0, rho1, zeros_x);
        y_m = z_m;   % m is not constrained by C1

        %% DR reflect: w = 2y - z
        w_rho = 2*y_rho - z_rho;
        w_m   = 2*y_m   - z_m;
        w_j   = 2*y_j   - z_j;

        %% Step 2: proj_C2(w)
        %  Enforce  j = m - eps * d_x(interp_t(rho)).
        [pc2_rho, pc2_m, pc2_j] = proj_c2(w_rho, w_m, w_j, problem, ops, ...
                                            vareps, rho0, rho1, zeros_x);

        %% DR update: z_{k+1} = z_k + proj_C2(2y - z) - y
        z_rho = z_rho + pc2_rho - y_rho;
        z_m   = z_m   + pc2_m   - y_m;
        z_j   = z_j   + pc2_j   - y_j;

        %% Convergence: check FP residual of current y iterate
        fp_res = fp_residual(y_rho, y_m, problem, ops, vareps, rho0, rho1, zeros_x);
        if fp_res < dr_tol
            break;
        end
    end

    x_out.rho = y_rho;
    x_out.mx  = y_m;
end

% ---------------------------------------------------------------------------

function [rho_out, j_out] = proj_c1(rho_in, j_in, problem, ops, rho0, rho1, zeros_x)
% PROJ_C1  Project (rho, j) onto the eps=0 continuity equation d_t rho + d_x j = 0.
%
%   Solves the 2D Poisson equation  (-Delta) phi = f  via DCT,
%   where f = d_t rho + d_x j is the continuity residual.

    % Continuity residual (eps=0)
    f = ops.deriv_t_at_phi(rho_in, rho0, rho1) ...
      + ops.deriv_x_at_phi(j_in, zeros_x, zeros_x);

    % 2D DCT solve: eigenvalues lambda_x + lambda_t (no eps^2 term)
    lambda       = problem.lambda_x + problem.lambda_t;
    phi_hat      = mirt_dctn(f) ./ lambda;
    phi_hat(1,1) = 0;
    phi          = mirt_idctn(phi_hat);

    % Apply A* corrections
    rho_out = rho_in + ops.deriv_t_at_rho(phi);
    j_out   = j_in   + ops.deriv_x_at_m(phi);
end

% ---------------------------------------------------------------------------

function [rho_out, m_out, j_out] = proj_c2(rho_in, m_in, j_in, problem, ops, ...
                                             vareps, rho0, rho1, zeros_x)
% PROJ_C2  Project (rho, m, j) onto the flux definition  j = m - eps * d_x(interp_t(rho)).
%
%   KKT conditions give the multiplier lam satisfying:
%     (2I - eps^2 * d_xx) lam = r,    r = j - m + eps * d_x(interp_t(rho))
%
%   This decouples in t: for each time slice, it is a 1D Helmholtz equation.
%   Solved via DCT in x with eigenvalues  2 + eps^2 * lambda_x.
%
%   Updates:
%     rho += eps * interp_t_at_rho( d_x_phi(lam) )   [d_x_phi = backward diff]
%     m   += lam
%     j   -= lam

    % RHS: r = j - m + eps * G * S_full(rho)   lives on (nt x nxm)
    r = j_in - m_in + vareps * ops.deriv_x_at_m(ops.interp_t_at_phi(rho_in, rho0, rho1));

    % 1D Helmholtz solve in x, decoupled across time slices:
    %   (2 + eps^2 * lambda_x(k)) lam_hat(:,k) = r_hat(:,k)
    % lambda_x is (1 x nx); r is (nt x nxm). Use lambda_x(1:nxm) as approximation
    % (same eigenvalues as phi-grid; exact in continuous limit; accurate for fine grids).
    lambda_h = 2 + vareps^2 * problem.lambda_x(1, 1:problem.nx-1);   % (1 x nxm)

    r_hat   = dct(r');          % DCT in x: (nxm x nt)
    lam_hat = r_hat ./ lambda_h';   % broadcast: divide each of nt rows by (nxm x 1)
    lam     = idct(lam_hat)';   % (nt x nxm)

    % Apply KKT updates
    % rho: rho = rho_in - eps * S_lin^T * G^T * lam
    %         = rho_in + eps * interp_t_at_rho( d_x_phi(lam) )
    %   [since G^T = -d_x_phi, so -eps * S_lin^T * G^T = eps * interp_t_at_rho ∘ d_x_phi]
    rho_out = rho_in + vareps * ops.interp_t_at_rho( ...
                                    ops.deriv_x_at_phi(lam, zeros_x, zeros_x));
    m_out   = m_in + lam;
    j_out   = j_in - lam;
end

% ---------------------------------------------------------------------------

function res = fp_residual(rho, m, problem, ops, vareps, rho0, rho1, zeros_x)
% FP_RESIDUAL  ||d_t rho + d_x m - eps * d_xx rho||  (weighted L2 norm).

    lap_rho = ops.deriv_x_at_phi( ...
                  ops.deriv_x_at_m( ...
                      ops.interp_t_at_phi(rho, rho0, rho1)), ...
                  zeros_x, zeros_x);

    f   = ops.deriv_t_at_phi(rho, rho0, rho1) ...
        + ops.deriv_x_at_phi(m, zeros_x, zeros_x) ...
        - vareps * lap_rho;

    res = sqrt(problem.dt * problem.dx) * norm(f(:));
end
