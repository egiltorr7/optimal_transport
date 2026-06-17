function x_out = proj_fokker_planck_expsemi(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI  Project onto the FP constraint via ETD scheme.
%
%   Drop-in replacement for proj_fokker_planck_banded / proj_fokker_planck_imex.
%   Applies the ETD (exponential time differencing) discretisation of FP:
%
%       (rho_{j+1} - c_k * rho_j) / dt  +  phi_k * D_x psi_{j+1/2}  =  0
%
%   where c_k = exp(-eps*lambda_x(k)*dt) and phi_k = (1-c_k)/(eps*lambda_x(k)*dt).
%   The c_k factor is the exact semigroup; the phi_k weight is the ETD correction
%   for the momentum forcing (vs. phi_k=1 in the naive expsemi scheme).
%
%   Algorithm:
%     1. Compute S*rho_prev in DCT space using c_k coefficients
%     2. Compute D_x psi in physical space, DCT it
%     3. Form ETD residual: f_hat = (rho_curr - S*rho_prev)_hat/dt + phi_k .* (D_x psi)_hat
%     4. k=1 (DC mode, c=1, phi=1, T singular): gauge fix via DCT-in-t
%     5. k=2..nx: tridiagonal solve T_k phi = f_hat (precomputed LU)
%     6. IDCT -> phi in physical space
%     7. rho update: adj_rho_hat = (c_k * phi_hat_next - phi_hat_curr)/dt, IDCT
%        x_out.rho = mu  + adj_rho
%        x_out.mx  = psi + D_x^T (Phi_op * phi)
%        where Phi_op applies phi_k weighting in DCT space (adjoint of step 3)
%
%   Requires:
%     problem.expsemi_proj   precomputed by precomp_expsemi_proj(problem, vareps)
%     problem.lambda_t       precomputed by setup_problem
%
%   Grid alignment (identical to other FP projections):
%     x_in.rho  (ntm x nx)   staggered density   (times dt, 2dt, ..., (nt-1)*dt)
%     x_in.mx   (nt  x nxm)  staggered momentum

    ops      = problem.ops;
    rho0     = problem.rho0;   % (1 x nx)
    rho1     = problem.rho1;   % (1 x nx)
    nt       = problem.nt;
    ntm      = nt - 1;
    dt       = problem.dt;
    ep       = problem.expsemi_proj;
    c_vals   = ep.c_vals;    % (1 x nx): exp(-alpha_k)
    phi_vals = ep.phi_vals;  % (1 x nx): (1-c_k)/alpha_k

    zeros_x = zeros(nt, 1);

    mu  = x_in.rho;   % (ntm x nx)
    psi = x_in.mx;    % (nt  x nxm)

    %% --- ETD FP residual in DCT space ---
    % rho part: (rho_curr - S*rho_prev)/dt  applied in DCT-x
    mu_prev   = [rho0; mu];                          % (nt x nx)
    mu_curr   = [mu;   rho1];                        % (nt x nx)
    S_prev    = apply_semigroup(mu_prev, c_vals);    % (nt x nx)
    f_rho_hat = dct(((mu_curr - S_prev) / dt)')';   % (nt x nx)

    % m part: D_x psi in physical space, then DCT-x, then scale by phi_k
    Dxm_hat = dct(ops.deriv_x_at_phi(psi, zeros_x, zeros_x)')'; % (nt x nx)

    f_hat = f_rho_hat + phi_vals .* Dxm_hat;   % ETD residual in DCT space

    if norm(f_hat(:)) * sqrt(dt * problem.dx) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- Solve T_k phi_k = f_hat(:,k) for each DCT mode k ---
    phi_hat = zeros(nt, problem.nx);

    % k=1: DC mode (lambda_x=0, c=1, phi=1, T_1 singular -> DCT-in-t)
    f1_t          = dct(f_hat(:, 1));
    phi1_t        = zeros(nt, 1);
    phi1_t(2:end) = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:, 1) = idct(phi1_t);

    % k=2..nx: SPD tridiagonal solve with precomputed LU
    for k = 2:problem.nx
        phi_hat(:, k) = ep.Tj_U{k} \ (ep.Tj_L{k} \ (ep.Tj_P{k} * f_hat(:, k)));
    end

    %% --- rho update: adj_rho_hat = (c_k * phi_hat_next - phi_hat_curr)/dt ---
    phi_hat_curr = phi_hat(1:ntm, :);
    phi_hat_next = phi_hat(2:nt,  :);
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / dt;   % (ntm x nx)
    adj_rho      = idct(adj_rho_hat')';

    x_out.rho = mu + adj_rho;

    %% --- m update: delta_m = D_x^T (Phi_op * phi) ---
    % Phi_op applies phi_k weighting in DCT space, then IDCT (adjoint of phi scaling)
    phi_weighted = idct((phi_vals .* phi_hat)')';   % (nt x nx)
    x_out.mx = psi + ops.deriv_x_at_m(phi_weighted);
end

function S_rho = apply_semigroup(rho, c_vals)
% Apply exact heat semigroup: S_rho = IDCT_x(c_vals .* DCT_x(rho))
% rho:    (m x nx) in physical space
% c_vals: (1 x nx) semigroup coefficients per x-mode
% S_rho:  (m x nx) in physical space
    rho_hat = dct(rho')';                    % (m x nx)
    S_rho   = idct((rho_hat .* c_vals)')';  % (m x nx)
end
