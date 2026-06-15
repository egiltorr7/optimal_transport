function x_out = proj_fokker_planck_expsemi(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI  Project onto the FP constraint using the exact
%   heat semigroup S_eps^dt = exp(eps*Delta*dt) for the diffusion step.
%
%   Drop-in replacement for proj_fokker_planck_banded / proj_fokker_planck_imex.
%   Instead of approximating exp(eps*Delta*dt) by Padé(1,1) (CN) or backward
%   Euler (IMEX), it applies the exact semigroup via DCT in x:
%
%       S_eps^dt rho = IDCT_x( exp(-eps * lambda_x * dt) .* DCT_x(rho) )
%
%   The FP constraint becomes:
%       (rho_k - S_eps^dt rho_{k-1}) / dt  +  D_x psi_{k-1/2}  =  0
%
%   This eliminates the ε-dependent approximation error in the diffusion step,
%   making the time-discretisation error O(dt) and ε-independent.
%
%   Algorithm:
%     1. Compute FP residual f = (mu_curr - S*mu_prev)/dt + D_x psi  (nt x nx)
%     2. DCT in x -> f_hat (nt x nx)
%     3. j=1 (DC mode, c=1, T singular): gauge fix via DCT-in-t (same as banded)
%     4. j=2..nx: tridiagonal solve T_j phi_j = f_j (LU from expsemi_proj)
%     5. IDCT in x -> phi  (nt x nx)
%     6. Apply -A_rho^T phi in DCT space, then IDCT -> adj_rho
%        x_out.rho = mu + adj_rho   (= mu - A_rho^T phi)
%        x_out.mx  = psi + D_x^T phi
%
%   Requires:
%     problem.expsemi_proj   precomputed by precomp_expsemi_proj(problem, vareps)
%     problem.lambda_t       precomputed by setup_problem
%
%   Grid alignment (identical to other FP projections):
%     x_in.rho  (ntm x nx)   staggered density   (times dt, 2dt, ..., (nt-1)*dt)
%     x_in.mx   (nt  x nxm)  staggered momentum

    ops    = problem.ops;
    rho0   = problem.rho0;   % (1 x nx)
    rho1   = problem.rho1;   % (1 x nx)
    nt     = problem.nt;
    ntm    = nt - 1;
    dt     = problem.dt;
    ep     = problem.expsemi_proj;
    c_vals = ep.c_vals;   % (1 x nx): exp(-vareps * lambda_x * dt)

    zeros_x = zeros(nt, 1);

    mu  = x_in.rho;   % (ntm x nx)
    psi = x_in.mx;    % (nt  x nxm)

    %% --- FP residual  f = (mu_curr - S*mu_prev)/dt + D_x psi  (nt x nx) ---
    mu_prev = [rho0; mu];    % (nt x nx): rho at times 0, 1*dt, ..., (nt-1)*dt
    mu_curr = [mu; rho1];    % (nt x nx): rho at times 1*dt, ..., (nt-1)*dt, T

    S_prev = apply_semigroup(mu_prev, c_vals);   % (nt x nx)

    f = (mu_curr - S_prev) / dt ...
      + ops.deriv_x_at_phi(psi, zeros_x, zeros_x);

    if norm(f(:)) * sqrt(problem.dt * problem.dx) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- DCT in x ---
    f_hat = dct(f')';    % (nt x nx)

    phi_hat = zeros(nt, problem.nx);

    % j=1: DC mode (c=1, T_1 = pure time Laplacian M0, singular).
    % Invert via DCT-in-t with gauge fix (j=1, l=1) -> zero.
    f1_t          = dct(f_hat(:, 1));
    phi1_t        = zeros(nt, 1);
    phi1_t(2:end) = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:, 1) = idct(phi1_t);

    % j=2..nx: SPD tridiagonal solve with precomputed LU
    for j = 2:problem.nx
        phi_hat(:, j) = ep.Tj_U{j} \ (ep.Tj_L{j} \ (ep.Tj_P{j} * f_hat(:, j)));
    end

    %% --- IDCT in x -> phi in physical space ---
    phi = idct(phi_hat')';   % (nt x nx)

    %% --- Apply -A_rho^T phi in DCT space, then IDCT ---
    % In mode j: (-A_rho^T phi)_k = (c_j phi_{k+1} - phi_k) / dt
    % so x_out.rho = mu - A_rho^T phi  (projection formula)
    phi_hat_curr = phi_hat(1:ntm, :);            % (ntm x nx)
    phi_hat_next = phi_hat(2:nt,  :);            % (ntm x nx)
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / dt;   % (ntm x nx)
    adj_rho      = idct(adj_rho_hat')';          % (ntm x nx)

    x_out.rho = mu  + adj_rho;
    x_out.mx  = psi + ops.deriv_x_at_m(phi);
end

function S_rho = apply_semigroup(rho, c_vals)
% Apply exact heat semigroup: S_rho = IDCT_x(c_vals .* DCT_x(rho))
% rho:    (m x nx) in physical space
% c_vals: (1 x nx) semigroup coefficients per x-mode
% S_rho:  (m x nx) in physical space
    rho_hat = dct(rho')';                    % (m x nx)
    S_rho   = idct((rho_hat .* c_vals)')';  % (m x nx)
end
