function x_out = proj_fokker_planck_backslash(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_BACKSLASH  FP projection via mode-by-mode sparse backslash.
%
%   x_out = proj_fokker_planck_backslash(x_in, problem, cfg)
%
%   Drop-in replacement for proj_fokker_planck_banded / proj_fokker_planck_spike2.
%   Produces the same result but solves each (kx,ky) tridiagonal system
%   independently using MATLAB's sparse \ (UMFPACK LU with partial pivoting)
%   in a sequential double loop over modes.
%
%   Purpose: naive CPU baseline for timing and correctness comparison.
%     - Numerically stable for all eps (partial pivoting, no sign hazard).
%     - Cannot run on GPU (sparse \ is CPU-only).
%     - Expected to be ~10-100x slower than the batched Thomas/Spike solvers.
%
%   Requires problem.banded_proj from precomp_banded_proj (same as banded).
%   Set cfg.use_gpu = false when using this projection.

    ops    = problem.ops;
    rho0   = problem.rho0;
    rho1   = problem.rho1;
    nt     = problem.nt;
    nx     = problem.nx;
    ny     = problem.ny;
    vareps = cfg.vareps;
    bp     = problem.banded_proj;

    % Gather from GPU if needed (sparse \ is CPU-only)
    mu    = gather(x_in.rho);
    psi_x = gather(x_in.mx);
    psi_y = gather(x_in.my);
    rho0  = gather(rho0);
    rho1  = gather(rho1);

    zeros_x = zeros(nt, ny);
    zeros_y = zeros(nt, nx);

    %% --- FP residual  f = d_t mu + d_x psi_x + d_y psi_y - eps*Delta mu ---
    mu_phi   = ops.interp_t_at_phi(mu, rho0, rho1);
    nabla_mu = ops.deriv_x_at_phi(ops.deriv_x_at_m(mu_phi), zeros_x, zeros_x) ...
             + ops.deriv_y_at_phi(ops.deriv_y_at_m(mu_phi), zeros_y, zeros_y);

    f = ops.deriv_t_at_phi(mu, rho0, rho1) ...
      + ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
      + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y) ...
      - vareps * nabla_mu;

    if norm(f(:)) * sqrt(problem.dt * problem.dx * problem.dy) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- 2D DCT in x and y ---
    f_hat = dct2_xy(f, nt, nx, ny);   % (nt x nx x ny)

    %% --- Mode-by-mode sparse backslash ---
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

            lo = bp.lower_all(:, kx, ky);   % (nt-1 x 1)
            ma = bp.main_all(:,  kx, ky);   % (nt   x 1)
            up = bp.upper_all(:, kx, ky);   % (nt-1 x 1)

            T = sparse([i_diag;  i_lower; i_upper], ...
                       [i_diag;  j_lower; j_upper], ...
                       [ma;      lo;      up],  nt, nt);

            phi_hat(:, kx, ky) = T \ f_hat(:, kx, ky);
        end
    end

    %% --- (kx=1,ky=1) DC mode: lxy=0 -> singular T -> DCT-t ---
    f1_t           = dct(f_hat(:, 1, 1));
    phi1_t         = zeros(nt, 1);
    phi1_t(2:end)  = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:,1,1) = idct(phi1_t);

    %% --- 2D IDCT back to physical space ---
    phi = idct2_xy(phi_hat, nt, nx, ny);   % (nt x nx x ny)

    %% --- Apply A* corrections ---
    dphi_dx   = ops.deriv_x_at_m(phi);
    dphi_dy   = ops.deriv_y_at_m(phi);

    nabla_phi = ops.interp_t_at_rho( ...
        ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x) + ...
        ops.deriv_y_at_phi(dphi_dy, zeros_y, zeros_y));

    x_out.rho = mu    + ops.deriv_t_at_rho(phi) + vareps .* nabla_phi;
    x_out.mx  = psi_x + dphi_dx;
    x_out.my  = psi_y + dphi_dy;
end

% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f, nt, nx, ny)
% DCT-II along dim 2 (x) and dim 3 (y) of a (nt x nx x ny) array.
    f_hat = permute(f, [2, 1, 3]);
    f_hat = reshape(f_hat, nx, nt*ny);
    f_hat = dct(f_hat);
    f_hat = reshape(f_hat, nx, nt, ny);
    f_hat = permute(f_hat, [3, 2, 1]);   % merged permute
    f_hat = reshape(f_hat, ny, nt*nx);
    f_hat = dct(f_hat);
    f_hat = reshape(f_hat, ny, nt, nx);
    f_hat = permute(f_hat, [2, 3, 1]);
end

function f = idct2_xy(f_hat, nt, nx, ny)
% IDCT-II along dim 2 (x) and dim 3 (y) of a (nt x nx x ny) array.
    f = permute(f_hat, [3, 1, 2]);
    f = reshape(f, ny, nt*nx);
    f = idct(f);
    f = reshape(f, ny, nt, nx);
    f = permute(f, [3, 2, 1]);   % merged permute
    f = reshape(f, nx, nt*ny);
    f = idct(f);
    f = reshape(f, nx, nt, ny);
    f = permute(f, [2, 1, 3]);
end
