function x_out = proj_fokker_planck_expsemi_nonuniform(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_NONUNIFORM  ETD projection on a non-uniform time grid.
%
%   Drop-in replacement for proj_fokker_planck_expsemi that uses
%   problem.dt_vec (nt x 1) instead of a scalar dt.
%
%   Requires problem.expsemi_nonuniform_proj set by
%   precomp_expsemi_proj_nonuniform(problem, vareps).

    ops      = problem.ops;
    rho0     = problem.rho0;   % (1 x nx)
    rho1     = problem.rho1;   % (1 x nx)
    nt       = problem.nt;
    ntm      = nt - 1;
    dt_vec   = problem.dt_vec; % (nt x 1)
    ep       = problem.expsemi_nonuniform_proj;
    c_mat    = ep.c_mat;       % (nt x nx)
    phi_mat  = ep.phi_mat;     % (nt x nx)

    mu  = x_in.rho;   % (ntm x nx)
    psi = x_in.mx;    % (nt  x nxm)

    zeros_x = zeros(nt, 1);

    %% ETD residual in DCT-x space
    mu_prev     = [rho0; mu];    % (nt x nx)
    mu_curr     = [mu;   rho1];  % (nt x nx)

    % Apply spatial DCT — semigroup is diagonal in this space
    rho_hat_prev = dct(mu_prev')';   % (nt x nx)
    rho_hat_curr = dct(mu_curr')';   % (nt x nx)

    % ETD semigroup: S_prev_hat(n,k) = c_mat(n,k) * rho_hat_prev(n,k)
    S_prev_hat  = c_mat .* rho_hat_prev;

    % Residual: (rho_curr - S*rho_prev) / dt_n  +  phi_n * D_x m
    f_rho_hat = (rho_hat_curr - S_prev_hat) ./ dt_vec;   % dt_vec broadcast

    Dxm_hat = dct(ops.deriv_x_at_phi(psi, zeros_x, zeros_x)')';

    f_hat = f_rho_hat + phi_mat .* Dxm_hat;

    %% Solve T_k phi_k = f_hat(:,k) for modes k>=2
    % k=1 (DC mode) is handled analytically below: T_1 is singular on
    % non-uniform grids (null vector = dt_vec, not the constant vector).
    phi_hat = zeros(nt, problem.nx);
    for k = 2:problem.nx
        phi_hat(:, k) = ep.Tj_U{k} \ (ep.Tj_L{k} \ (ep.Tj_P{k} * f_hat(:, k)));
    end

    %% rho update: adjoint of (rho_n - c_{n}*rho_{n-1})/dt_n
    % KKT for mu(j): adj_mu(j) = c_{j+1}/dt_{j+1} * phi(j+1) - 1/dt_j * phi(j)
    adj_rho_hat = c_mat(2:nt, :) .* phi_hat(2:nt, :) ./ dt_vec(2:nt) ...
                - phi_hat(1:ntm, :) ./ dt_vec(1:ntm);

    % DC mode (k=1): direct mass-conservation projection.
    % Enforce DC(x_out.rho(n,:)) = DC(rho0) for all interior stagger points.
    % phi_hat(:,1) = 0 so adj_rho_hat(:,1) from the formula above is 0;
    % override it with the correct correction.
    rho_dc = rho_hat_prev(1, 1);
    adj_rho_hat(:, 1) = rho_dc - rho_hat_curr(1:ntm, 1);

    adj_rho = idct(adj_rho_hat')';

    x_out.rho = mu + adj_rho;

    %% m update: D_x^T (phi_mat * phi_hat in DCT space)
    phi_weighted = idct((phi_mat .* phi_hat)')';   % (nt x nx)
    x_out.mx = psi + ops.deriv_x_at_m(phi_weighted);
end
