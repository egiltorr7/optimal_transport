function x_out = proj_fokker_planck_imex(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_IMEX  Project onto the FP constraint (IMEX / backward-Euler diffusion).
%
%   Drop-in replacement for proj_fokker_planck_banded using backward-Euler
%   time discretisation for the diffusion term instead of Crank-Nicolson.
%
%   Discretisation of the FP constraint:
%     [D_t x I  -  eps (U_t x K_x)] rho_tilde  +  (I x D_x) b  =  d
%
%   where  U_t rho_tilde = [rho_tilde; 0]  (nt x nx),  i.e. for each
%   cell-centre time k the diffusion uses rho_tilde_k (end-of-step value)
%   except at the last cell-centre where it uses rho1 (nu, the BC).
%
%   Consequence for the RHS:
%     d = (1/dt)(e_1 x mu - e_nt x nu) + eps * e_nt x K_x*nu
%   Only the Laplacian of nu appears; mu does NOT contribute a diffusion BC.
%
%   Adjoint of the IMEX FP operator for rho:
%     A_rho^T phi = -Dt_bwd * phi  -  eps * (K_x phi)(1:ntm, :)
%   (U_t^T takes the first ntm rows of K_x phi -- no time interpolation matrix)
%
%   Requires:
%     problem.imex_proj    precomputed by precomp_imex_proj(problem, vareps)
%     problem.lambda_t     precomputed by setup_problem
%
%   Grid alignment (identical to proj_fokker_planck_banded):
%     x_in.rho  (ntm x nx)   staggered density
%     x_in.mx   (nt  x nxm)  staggered momentum

    ops    = problem.ops;
    rho0   = problem.rho0;
    rho1   = problem.rho1;
    nt     = problem.nt;
    ntm    = nt - 1;
    vareps = cfg.vareps;
    ip     = problem.imex_proj;

    zeros_x = zeros(nt, 1);

    mu  = x_in.rho;   % (ntm x nx)
    psi = x_in.mx;    % (nt  x nxm)

    %% --- FP residual  f = D_t mu - eps*K_x*[mu; rho1] + D_x psi  (nt x nx) ---
    % BE interpolation: U_t * rho_tilde = [rho_tilde; 0] for interior,
    % plus the BC rho1 at the last row (moved to RHS in the constraint,
    % but included here so f correctly measures the full residual).
    rho_interp = [mu; rho1];   % (nt x nx)

    laplacian_rho = ops.deriv_x_at_phi( ...
                        ops.deriv_x_at_m(rho_interp), ...
                        zeros_x, zeros_x);

    f = ops.deriv_t_at_phi(mu, rho0, rho1) ...
      + ops.deriv_x_at_phi(psi, zeros_x, zeros_x) ...
      - vareps * laplacian_rho;

    if norm(f(:)) * sqrt(problem.dt * problem.dx) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- DCT in x (each row of f independently) ---
    f_hat = dct(f')';    % (nt x nx)

    phi_hat = zeros(nt, problem.nx);

    % j=1: DC mode, lambda_x=0 => T_1 = M0 is singular.
    % Invert via DCT in t; (j=1, l=1) is the gauge freedom -> zero.
    f1_t          = dct(f_hat(:, 1));
    phi1_t        = zeros(nt, 1);
    phi1_t(2:end) = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:, 1) = idct(phi1_t);

    % j=2..nx: SPD tridiagonal solve with precomputed LU factors
    for j = 2:problem.nx
        phi_hat(:, j) = ip.Tj_U{j} \ (ip.Tj_L{j} \ (ip.Tj_P{j} * f_hat(:, j)));
    end

    %% --- IDCT in x -> phi in physical space (nt x nx) ---
    phi = idct(phi_hat')';

    %% --- Apply A^T to get corrections ---
    % A_rho^T phi = -Dt_bwd*phi - eps * U_t^T * K_x * phi
    % U_t^T phi   = phi(1:ntm, :)   (first ntm rows; no matrix multiply needed)
    dphi_dx = ops.deriv_x_at_m(phi);
    Kx_phi  = ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x);   % (nt x nx)

    x_out.rho = mu  + ops.deriv_t_at_rho(phi) + vareps * Kx_phi(1:ntm, :);
    x_out.mx  = psi + dphi_dx;
end
