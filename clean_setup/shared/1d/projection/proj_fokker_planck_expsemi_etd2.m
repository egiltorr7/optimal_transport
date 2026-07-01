function x_out = proj_fokker_planck_expsemi_etd2(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_ETD2  Project onto the FP constraint via ETD2.
%
%   ETD2 variant of proj_fokker_planck_expsemi.  D_x m is approximated as
%   linear in time over each interval, and the integrating-factor integral
%   is computed exactly:
%
%       (rho_{j+1} - c_k*rho_j)/dt  +  phi1_k*(D_x m)^j  +  phi2_k*(D_x m)^{j+1}  =  0
%
%   where phi1_k, phi2_k are the ETD2 weights (see precomp_expsemi_proj_etd2).
%   At large eps, phi2 >> phi1, correctly weighting the end-of-interval
%   momentum (fixing the ETD1 midpoint bias that grows with eps).
%
%   Grid (differs from ETD1 only in the time dimension of mx):
%     x_in.rho  (ntm   x nx)   density at interior integer times t_1,...,t_{nt-1}
%     x_in.mx   (nt+1  x nxm)  momentum at ALL integer times t_0,...,t_nt
%
%   Requires:
%     problem.expsemi_proj_etd2   precomputed by precomp_expsemi_proj_etd2
%     problem.lambda_t            precomputed by setup_problem

    ops  = problem.ops;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    nt   = problem.nt;
    ntm  = nt - 1;
    dt   = problem.dt;
    ep   = problem.expsemi_proj_etd2;
    c_vals    = ep.c_vals;     % (1 x nx)
    phi1_vals = ep.phi1_vals;  % (1 x nx)
    phi2_vals = ep.phi2_vals;  % (1 x nx)

    zeros_xp1 = zeros(nt+1, 1);   % zero-flux BCs at spatial walls, all nt+1 times

    mu  = x_in.rho;   % (ntm x nx)
    psi = x_in.mx;    % (nt+1 x nxm)

    %% --- ETD2 FP residual in DCT-x space ---
    mu_prev   = [rho0; mu];                           % (nt x nx)
    mu_curr   = [mu;   rho1];                         % (nt x nx)
    S_prev    = apply_semigroup(mu_prev, c_vals);
    f_rho_hat = dct(((mu_curr - S_prev) / dt)')';     % (nt x nx)

    % D_x m at all nt+1 integer times, in DCT-x space
    Dxm_all   = dct(ops.deriv_x_at_phi(psi, zeros_xp1, zeros_xp1)')';  % (nt+1 x nx)
    Dxm_start = Dxm_all(1:nt, :);      % (nt x nx): start of intervals j=1..nt
    Dxm_end   = Dxm_all(2:nt+1, :);    % (nt x nx): end of intervals   j=1..nt

    f_hat = f_rho_hat + phi1_vals .* Dxm_start + phi2_vals .* Dxm_end;

    if norm(f_hat(:)) * sqrt(dt * problem.dx) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- Solve T_k lambda_k = f_hat(:,k) for each DCT-x mode k ---
    phi_hat = zeros(nt, problem.nx);

    % k=1: DC mode (lambda_x=0, T_1 singular) -- DCT-in-t, same as ETD1
    f1_t          = dct(f_hat(:, 1));
    phi1_t        = zeros(nt, 1);
    phi1_t(2:end) = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:, 1) = idct(phi1_t);

    % k=2..nx: symmetric tridiagonal solve (LU precomputed)
    for k = 2:problem.nx
        phi_hat(:, k) = ep.Tj_U{k} \ (ep.Tj_L{k} \ (ep.Tj_P{k} * f_hat(:, k)));
    end

    %% --- rho update: identical to ETD1 ---
    phi_hat_curr = phi_hat(1:ntm, :);
    phi_hat_next = phi_hat(2:nt,  :);
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / dt;
    adj_rho      = idct(adj_rho_hat')';
    x_out.rho    = mu + adj_rho;

    %% --- m update at all nt+1 integer times ---
    % lambda has nt entries (one per FP interval, 1-indexed j=1..nt).
    % m at time j appears as "end" of interval j   (coeff phi2, lambda^j)
    %                     and "start" of interval j+1 (coeff phi1, lambda^{j+1}).
    % m at t_0   -> only start of interval 1:  phi1 * lambda^1
    % m at t_j   -> phi2 * lambda^j + phi1 * lambda^{j+1}   (j=1..nt-1)
    % m at t_nt  -> only end of interval nt:   phi2 * lambda^nt
    v_hat = zeros(nt+1, problem.nx);
    v_hat(1, :)    = phi1_vals .* phi_hat(1, :);
    v_hat(2:nt, :) = phi2_vals .* phi_hat(1:ntm, :) + phi1_vals .* phi_hat(2:nt, :);
    v_hat(nt+1, :) = phi2_vals .* phi_hat(nt, :);

    phi_weighted = idct(v_hat')';       % (nt+1 x nx)
    x_out.mx = psi + ops.deriv_x_at_m(phi_weighted);
end

function S_rho = apply_semigroup(rho, c_vals)
    rho_hat = dct(rho')';
    S_rho   = idct((rho_hat .* c_vals)')';
end
