function x_out = proj_fokker_planck_expsemi_backslash_block_gpu(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_BACKSLASH_BLOCK_GPU  ETD FP projection via ONE
%   big block-diagonal sparse gpuArray backslash, instead of nx*ny separate
%   small backslash calls.
%
%   x_out = proj_fokker_planck_expsemi_backslash_block_gpu(x_in, problem, cfg)
%
%   Same twiddle-based, permute-free DCT infrastructure and residual/update
%   steps as proj_fokker_planck_expsemi_gpu / proj_fokker_planck_expsemi_backslash_gpu.
%   The ONLY difference from proj_fokker_planck_expsemi_backslash_gpu: instead
%   of looping over nx*ny modes and building+solving a fresh small sparse
%   system per mode on every call, this uses ONE big block-diagonal sparse
%   matrix (ep.T_block, precomputed ONCE by precomp_expsemi_proj_block --
%   the diagonals never change across ADMM iterations, only the RHS does)
%   and does a SINGLE backslash call per projection call.
%
%   Why this and not "cache the factorization" (what Thomas already does via
%   main_T_mod): MATLAB's decomposition/lu objects do NOT support sparse
%   gpuArray ("Parallel Computing Toolbox does not support sparse matrices
%   on the GPU" for that caching mechanism) -- so there is no way to reuse a
%   cached LU factorization across calls the way Thomas does. What IS
%   available: reduce the NUMBER of separate host-orchestrated sparse-solve
%   calls from nx*ny down to 1. proj_fokker_planck_expsemi_backslash_gpu
%   pays a per-mode overhead nx*ny times per call; this pays it once.
%
%   Requires:
%     problem.expsemi_proj   precomputed by precomp_expsemi_proj_block(problem,vareps)
%     Fields used: c_vals, phi_vals, T_block, tw_x, w_x, iw_x, itw_x, tw_y,
%     w_y, iw_y, itw_y (same DCT twiddles as proj_fokker_planck_expsemi_gpu).
%
%   Grid:
%     x_in.rho  (ntm x nx  x ny)
%     x_in.mx   (nt  x nxm x ny)
%     x_in.my   (nt  x nx  x nym)

    ops = problem.ops;
    nt  = problem.nt;   ntm = nt - 1;
    nx  = problem.nx;
    ny  = problem.ny;
    ep  = problem.expsemi_proj;
    c_vals   = ep.c_vals;
    phi_vals = ep.phi_vals;

    mu    = x_in.rho;
    psi_x = x_in.mx;
    psi_y = x_in.my;

    rho0_3d = reshape(problem.rho0, 1, nx, ny);
    rho1_3d = reshape(problem.rho1, 1, nx, ny);
    zeros_x = zeros(nt, ny, 'like', mu);
    zeros_y = zeros(nt, nx, 'like', mu);

    %% --- ETD FP residual in DCT-xy space ---
    mu_prev = cat(1, rho0_3d, mu);
    mu_curr = cat(1, mu, rho1_3d);

    mu_prev_hat = dct2_xy(mu_prev, ep);
    mu_curr_hat = dct2_xy(mu_curr, ep);
    f_rho_hat   = (mu_curr_hat - c_vals .* mu_prev_hat) / problem.dt;

    div_psi = ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
            + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y);
    div_psi_hat = dct2_xy(div_psi, ep);

    f_hat = f_rho_hat + phi_vals .* div_psi_hat;

    if norm(f_hat(:)) * sqrt(problem.dt * problem.dx * problem.dy) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- ONE big block-diagonal sparse solve for ALL modes at once ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;   % DC mode: T_block's DC block has main=1 -> phi≈0, same as Thomas

    M = nx * ny;
    rhs_block = reshape(rhs, nt * M, 1);
    phi_block = ep.T_block \ rhs_block;
    phi_hat   = reshape(phi_block, nt, nx, ny);

    %% --- DC mode (kx=1,ky=1): singular T, solve via 1-D DCT in time ---
    f1_col         = f_hat(:, 1, 1);
    f1_t           = dct_rows_gpu(f1_col');
    phi1_t         = zeros(1, nt, 'like', f_hat);
    phi1_t(2:end)  = f1_t(2:end) ./ reshape(problem.lambda_t(2:end), 1, []);
    phi_hat(:,1,1) = idct_rows_gpu(phi1_t)';

    %% --- rho update ---
    phi_hat_curr = phi_hat(1:ntm, :, :);
    phi_hat_next = phi_hat(2:nt,  :, :);
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / problem.dt;
    adj_rho      = idct2_xy(adj_rho_hat, ep);

    x_out.rho = mu + adj_rho;

    %% --- mx / my update ---
    phi_weighted = idct2_xy(phi_vals .* phi_hat, ep);
    x_out.mx = psi_x + ops.deriv_x_at_m(phi_weighted);
    x_out.my = psi_y + ops.deriv_y_at_m(phi_weighted);
end

% ---------------------------------------------------------------------------
% Same DCT/twiddle helpers as proj_fokker_planck_expsemi_gpu (copied verbatim
% so this file is self-contained and independently timeable).
% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f, ep)
    Nx = size(f, 2);
    xe = cat(2, f, f(:, Nx:-1:1, :));
    V  = fft(xe, [], 2);
    f_hat = real(V(:, 1:Nx, :) .* ep.tw_x) .* ep.w_x;

    Ny = size(f_hat, 3);
    ye = cat(3, f_hat, f_hat(:, :, Ny:-1:1));
    V  = fft(ye, [], 3);
    f_hat = real(V(:, :, 1:Ny) .* ep.tw_y) .* ep.w_y;
end

function f = idct2_xy(f_hat, ep)
    [m, nx_sz, Ny] = size(f_hat);
    Z  = f_hat .* ep.iw_y;
    U  = Z .* ep.itw_y;
    ye = cat(3, U, zeros(m, nx_sz, Ny, 'like', U));
    f  = real(ifft(ye, [], 3)) * (2*Ny);
    f  = f(:, :, 1:Ny);

    [m, Nx, ny_sz] = size(f);
    Z  = f .* ep.iw_x;
    U  = Z .* ep.itw_x;
    xe = cat(2, U, zeros(m, Nx, ny_sz, 'like', U));
    f  = real(ifft(xe, [], 2)) * (2*Nx);
    f  = f(:, 1:Nx, :);
end

function X = dct_rows_gpu(x)
    [~, N] = size(x);
    xe  = [x, x(:, N:-1:1)];
    V   = fft(xe, [], 2);
    k   = 0:N-1;
    tw  = exp(-1i * pi * k / (2*N));
    raw = real(V(:, 1:N) .* tw);
    w   = [1/sqrt(N), sqrt(2/N) * ones(1, N-1)];
    X   = raw .* (w / 2);
end

function x = idct_rows_gpu(X)
    [m, N] = size(X);
    w  = [1/sqrt(N), sqrt(2/N) * ones(1, N-1)];
    Z  = X .* w;
    k  = 0:N-1;
    U  = complex(Z) .* exp(1i * pi * k / (2*N));
    x  = real(ifft([U, zeros(m, N, 'like', U)], [], 2)) * (2*N);
    x  = x(:, 1:N);
end
