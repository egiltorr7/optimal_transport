function x_out = proj_fokker_planck_expsemi_backslash(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_BACKSLASH  ETD FP projection via mode-by-mode
%   sparse backslash.
%
%   x_out = proj_fokker_planck_expsemi_backslash(x_in, problem, cfg)
%
%   Same ETD (exact heat semigroup) formulation as proj_fokker_planck_expsemi
%   -- identical residual and update steps -- but each (kx,ky) tridiagonal
%   system  T_{kx,ky} * phi_hat(:,kx,ky) = f_hat(:,kx,ky)  is solved
%   independently with MATLAB's sparse \ (UMFPACK LU with partial pivoting)
%   in a sequential double loop over modes, instead of the batched Thomas
%   sweep in thomas_solve.
%
%   Purpose: naive CPU baseline for timing/correctness comparison against
%   proj_fokker_planck_expsemi and proj_fokker_planck_expsemi_gpu.
%     - Numerically stable for all eps (partial pivoting, no sign hazard).
%     - Cannot run on GPU (sparse \ is CPU-only).
%     - Expected to be ~10-100x slower than the batched Thomas solve, and
%       gets worse as nx*ny grows (one sparse build + factorize per mode).
%
%   Requires:
%     problem.expsemi_proj   precomputed by precomp_expsemi_proj(problem,vareps)
%   Set cfg.use_gpu = false when using this projection.
%
%   Grid (same as all other 2D projections):
%     x_in.rho  (ntm x nx  x ny)
%     x_in.mx   (nt  x nxm x ny)
%     x_in.my   (nt  x nx  x nym)

    ops = problem.ops;
    nt  = problem.nt;   ntm = nt - 1;
    nx  = problem.nx;
    ny  = problem.ny;
    ep  = problem.expsemi_proj;

    % Gather from GPU if needed (sparse \ is CPU-only); no-op if already CPU.
    mu       = to_cpu(x_in.rho);
    psi_x    = to_cpu(x_in.mx);
    psi_y    = to_cpu(x_in.my);
    rho0     = to_cpu(problem.rho0);
    rho1     = to_cpu(problem.rho1);
    c_vals   = to_cpu(ep.c_vals);
    phi_vals = to_cpu(ep.phi_vals);
    lower_all = to_cpu(ep.lower_all);
    main_all  = to_cpu(ep.main_all);
    upper_all = to_cpu(ep.upper_all);
    lambda_t  = to_cpu(problem.lambda_t);

    rho0_3d = reshape(rho0, 1, nx, ny);
    rho1_3d = reshape(rho1, 1, nx, ny);
    zeros_x = zeros(nt, ny);
    zeros_y = zeros(nt, nx);

    %% --- ETD FP residual in DCT-xy space ---
    mu_prev = cat(1, rho0_3d, mu);
    mu_curr = cat(1, mu, rho1_3d);

    mu_prev_hat = dct2_xy(mu_prev, nt, nx, ny);
    mu_curr_hat = dct2_xy(mu_curr, nt, nx, ny);
    f_rho_hat   = (mu_curr_hat - c_vals .* mu_prev_hat) / problem.dt;

    div_psi = ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
            + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y);
    div_psi_hat = dct2_xy(div_psi, nt, nx, ny);

    f_hat = f_rho_hat + phi_vals .* div_psi_hat;

    if norm(f_hat(:)) * sqrt(problem.dt * problem.dx * problem.dy) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- Mode-by-mode sparse backslash (kx,ky) != (1,1) ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;   % DC mode handled separately below

    phi_hat = zeros(nt, nx, ny);

    % Precompute sparse index vectors (same for every mode)
    i_diag  = (1:nt)';
    i_lower = (2:nt)';    j_lower = (1:nt-1)';
    i_upper = (1:nt-1)';  j_upper = (2:nt)';

    for ky = 1:ny
        for kx = 1:nx
            if kx == 1 && ky == 1
                continue;   % DC mode: singular T, handled separately below
            end

            lo = lower_all(:, kx, ky);   % (nt-1 x 1)
            ma = main_all(:,  kx, ky);   % (nt   x 1)
            up = upper_all(:, kx, ky);   % (nt-1 x 1)

            T = sparse([i_diag;  i_lower; i_upper], ...
                       [i_diag;  j_lower; j_upper], ...
                       [ma;      lo;      up],  nt, nt);

            phi_hat(:, kx, ky) = T \ rhs(:, kx, ky);
        end
    end

    %% --- DC mode (kx=1,ky=1): singular T, solve via 1-D DCT in time ---
    f1_t           = dct(f_hat(:, 1, 1));
    phi1_t         = zeros(nt, 1);
    phi1_t(2:end)  = f1_t(2:end) ./ lambda_t(2:end);
    phi_hat(:,1,1) = idct(phi1_t);

    %% --- rho update ---
    phi_hat_curr = phi_hat(1:ntm, :, :);
    phi_hat_next = phi_hat(2:nt,  :, :);
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / problem.dt;
    adj_rho      = idct2_xy(adj_rho_hat, ntm, nx, ny);

    x_out.rho = mu + adj_rho;

    %% --- mx / my update ---
    phi_weighted = idct2_xy(phi_vals .* phi_hat, nt, nx, ny);
    x_out.mx = psi_x + ops.deriv_x_at_m(phi_weighted);
    x_out.my = psi_y + ops.deriv_y_at_m(phi_weighted);
end

% ---------------------------------------------------------------------------
% Gathers a gpuArray to the host; returns non-gpuArray inputs unchanged
% (plain gather() errors on those).
% ---------------------------------------------------------------------------

function x = to_cpu(x)
    if isa(x, 'gpuArray')
        x = gather(x);
    end
end

% ---------------------------------------------------------------------------
% 2D DCT utilities -- identical to proj_fokker_planck_expsemi (built-in
% dct/idct, no twiddle precompute).
% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f, m, nx, ny)
    f2 = permute(f, [2, 1, 3]);
    f2 = reshape(f2, nx, m * ny);
    f2 = dct(f2);
    f2 = reshape(f2, nx, m, ny);
    f2 = permute(f2, [2, 1, 3]);

    f3 = permute(f2, [3, 1, 2]);
    f3 = reshape(f3, ny, m * nx);
    f3 = dct(f3);
    f3 = reshape(f3, ny, m, nx);
    f_hat = permute(f3, [2, 3, 1]);
end

function f = idct2_xy(f_hat, m, nx, ny)
    f3 = permute(f_hat, [3, 1, 2]);
    f3 = reshape(f3, ny, m * nx);
    f3 = idct(f3);
    f3 = reshape(f3, ny, m, nx);
    f2 = permute(f3, [2, 3, 1]);

    f2 = permute(f2, [2, 1, 3]);
    f2 = reshape(f2, nx, m * ny);
    f2 = idct(f2);
    f2 = reshape(f2, nx, m, ny);
    f  = permute(f2, [2, 1, 3]);
end
