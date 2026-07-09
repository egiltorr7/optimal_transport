function x_out = proj_fokker_planck_expsemi_gpu(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_GPU  2D FP projection via ETD.  GPU-optimised.
%
%   Optimisations over proj_fokker_planck_expsemi:
%     - Thomas solver uses (nx*ny x nt) memory layout → coalesced GPU column
%       access instead of strided slice access
%     - DCT uses direct fft(A,[],2) / fft(A,[],3) on the 3D array, eliminating
%       the four permute() calls per 2-D DCT
%     - Twiddle factors and orthonormalisation weights are precomputed in
%       precomp_expsemi_proj and cast to GPU once before the solve loop
%
%   Requires:
%     problem.expsemi_proj   precomputed by precomp_expsemi_proj(problem,vareps)
%     Fields used from ep:
%       c_vals, phi_vals              (1 x nx x ny)
%       lower_T, main_T, upper_T     (M x ntm/nt), M = nx*ny
%       tw_x, w_x, itw_x             (1 x nx x 1)
%       tw_y, w_y, itw_y             (1 x 1 x ny)
%
%   Grid:
%     x_in.rho  (ntm x nx  x ny)
%     x_in.mx   (nt  x nxm x ny)
%     x_in.my   (nt  x nx  x nym)

    ops  = problem.ops;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;
    ny   = problem.ny;
    ep   = problem.expsemi_proj;
    c_vals   = ep.c_vals;
    phi_vals = ep.phi_vals;

    mu    = x_in.rho;
    psi_x = x_in.mx;
    psi_y = x_in.my;

    rho0_3d = reshape(rho0, 1, nx, ny);
    rho1_3d = reshape(rho1, 1, nx, ny);
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

    %% --- Batched Thomas solve (kx,ky) != (1,1) ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;

    phi_hat = thomas_solve(ep, rhs, nt, nx, ny);

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
% 2-D DCT-II along x (dim 2) then y (dim 3) — no permutes.
% Uses precomputed twiddles from ep.
% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f, ep)
    % DCT along x (dim 2)
    Nx = size(f, 2);
    xe = cat(2, f, f(:, Nx:-1:1, :));
    V  = fft(xe, [], 2);
    f_hat = real(V(:, 1:Nx, :) .* ep.tw_x) .* ep.w_x;

    % DCT along y (dim 3)
    Ny = size(f_hat, 3);
    ye = cat(3, f_hat, f_hat(:, :, Ny:-1:1));
    V  = fft(ye, [], 3);
    f_hat = real(V(:, :, 1:Ny) .* ep.tw_y) .* ep.w_y;
end

function f = idct2_xy(f_hat, ep)
    % IDCT along y (dim 3) first
    [m, nx_sz, Ny] = size(f_hat);
    Z  = f_hat .* ep.iw_y;
    U  = Z .* ep.itw_y;
    ye = cat(3, U, zeros(m, nx_sz, Ny, 'like', U));
    f  = real(ifft(ye, [], 3)) * (2*Ny);
    f  = f(:, :, 1:Ny);

    % IDCT along x (dim 2)
    [m, Nx, ny_sz] = size(f);
    Z  = f .* ep.iw_x;
    U  = Z .* ep.itw_x;
    xe = cat(2, U, zeros(m, Nx, ny_sz, 'like', U));
    f  = real(ifft(xe, [], 2)) * (2*Nx);
    f  = f(:, 1:Nx, :);
end

% ---------------------------------------------------------------------------
% Thomas (TDMA) with (M x nt) layout — coalesced column access on GPU.
% ep.lower_T, ep.main_T, ep.upper_T are (M x ntm/nt), M = nx*ny.
% f_hat is (nt x nx x ny); result phi_hat is (nt x nx x ny).
% ---------------------------------------------------------------------------

function phi_hat = thomas_solve(ep, f_hat, nt, nx, ny)
    M = nx * ny;
    d = reshape(permute(f_hat, [2, 3, 1]), M, nt);
    b = ep.main_T;

    for i = 2:nt
        w       = ep.lower_T(:, i-1) ./ b(:, i-1);
        b(:, i) = b(:, i) - w .* ep.upper_T(:, i-1);
        d(:, i) = d(:, i) - w .* d(:, i-1);
    end

    phi_r        = zeros(M, nt, 'like', f_hat);
    phi_r(:, nt) = d(:, nt) ./ b(:, nt);
    for i = nt-1:-1:1
        phi_r(:, i) = (d(:, i) - ep.upper_T(:, i) .* phi_r(:, i+1)) ./ b(:, i);
    end

    phi_hat = permute(reshape(phi_r, nx, ny, nt), [3, 1, 2]);
end

% ---------------------------------------------------------------------------
% FFT-based orthonormal DCT-II/III along rows — GPU-compatible.
% Used only for the 1-D time DCT of the DC spatial mode.
% ---------------------------------------------------------------------------

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
