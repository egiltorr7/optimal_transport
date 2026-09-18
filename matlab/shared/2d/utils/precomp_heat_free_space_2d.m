function heat = precomp_heat_free_space_2d(problem, cfg)
% PRECOMP_HEAT_FREE_SPACE_2D  Precompute 2D heat kernels for R^2 (Gaussian convolution).
%
%   heat = precomp_heat_free_space_2d(problem, cfg)
%
%   Reference process: standard Brownian motion on R^2 (free space).
%   The 2D Gaussian kernel is separable:
%
%     K_t(x,y) = K_t^x(x) * K_t^y(y)
%
%   so convolution is applied as two successive 1D FFT convolutions:
%   first along dimension 1 (x), then along dimension 2 (y).
%
%   One-shot propagation (same rationale as 1D):
%     phi_0 and psi_T are concentrated on [0,1]^2 (narrow Gaussians),
%     so a single zero-padded convolution with K_T is accurate.
%     Step-by-step propagation compounds zero-padding truncation errors
%     as the function spreads outside [0,1]^2.
%
%   Inputs:
%     problem   struct with fields: nx, ny, dx, dy, dt, nt
%     cfg       struct with fields: vareps
%
%   Output:
%     heat.apply_full    phi_out = heat.apply_full(phi_in)
%                        Apply K_T (full time T=1) — used in Sinkhorn loop.
%                        phi_in / phi_out are (nx x ny).
%     heat.apply_time    phi_out = heat.apply_time(phi_in, t)
%                        Apply K_t for arbitrary t — used for trajectory recovery.
%     heat.name          'free_space_2d'

    dx     = problem.dx;
    dy     = problem.dy;
    nx     = problem.nx;
    ny     = problem.ny;
    vareps = cfg.vareps;

    % Precompute full-time kernel K_T (used in every Sinkhorn iteration)
    sigma2_T = 2 * vareps * 1.0;   % T=1
    [ker_fft_x_T, n_pad_x_T, n_fft_x_T] = build_kernel_1d(sigma2_T, dx, nx);
    [ker_fft_y_T, n_pad_y_T, n_fft_y_T] = build_kernel_1d(sigma2_T, dy, ny);

    heat.apply_full = @(phi) apply_fs_2d(phi, ...
        ker_fft_x_T, ker_fft_y_T, nx, ny, ...
        n_pad_x_T, n_fft_x_T, n_pad_y_T, n_fft_y_T);
    heat.apply_time = @(phi, t) apply_free_space_2d_t(phi, vareps, t, dx, dy, nx, ny);
    heat.name       = 'free_space_2d';
end

% -------------------------------------------------------------------------

function [ker_fft, n_pad, n_fft] = build_kernel_1d(sigma2, dz, nz)
% Build and FFT-transform the 1D Gaussian kernel for variance sigma2.
    n_pad = ceil(6 * sqrt(sigma2) / dz);
    n_fft = 2^nextpow2(nz + 2*n_pad);
    jj    = ((0:n_fft-1)' - floor(n_fft/2));
    ker   = exp(-((jj * dz).^2) / (2 * sigma2));
    ker   = ker / sum(ker);
    ker_fft = fft(ifftshift(ker));
end

function phi_out = apply_fs_dim1(phi_in, ker_fft, nx, n_pad, n_fft)
% Apply precomputed 1D Gaussian kernel along dimension 1 (x-axis) for each column.
    ny_in = size(phi_in, 2);
    sig = zeros(n_fft, ny_in);
    sig(n_pad+1 : n_pad+nx, :) = phi_in;
    conv_result = real(ifft(fft(sig, n_fft, 1) .* ker_fft, n_fft, 1));
    phi_out = max(conv_result(n_pad+1 : n_pad+nx, :), 0);
end

function phi_out = apply_fs_dim2(phi_in, ker_fft, ny, n_pad, n_fft)
% Apply precomputed 1D Gaussian kernel along dimension 2 (y-axis) for each row.
    nx_in = size(phi_in, 1);
    sig = zeros(nx_in, n_fft);
    sig(:, n_pad+1 : n_pad+ny) = phi_in;
    conv_result = real(ifft(fft(sig, n_fft, 2) .* ker_fft', n_fft, 2));
    phi_out = max(conv_result(:, n_pad+1 : n_pad+ny), 0);
end

function phi_out = apply_fs_2d(phi_in, ker_fft_x, ker_fft_y, nx, ny, ...
                                n_pad_x, n_fft_x, n_pad_y, n_fft_y)
% Separable 2D Gaussian convolution: first along x, then along y.
    tmp     = apply_fs_dim1(phi_in,  ker_fft_x, nx, n_pad_x, n_fft_x);
    phi_out = apply_fs_dim2(tmp,     ker_fft_y, ny, n_pad_y, n_fft_y);
end

function phi_out = apply_free_space_2d_t(phi_in, vareps, t, dx, dy, nx, ny)
% Apply K_t for an arbitrary time t (kernels recomputed each call).
    if t == 0
        phi_out = max(phi_in, 0);
        return;
    end
    sigma2 = 2 * vareps * t;
    [ker_fft_x, n_pad_x, n_fft_x] = build_kernel_1d(sigma2, dx, nx);
    [ker_fft_y, n_pad_y, n_fft_y] = build_kernel_1d(sigma2, dy, ny);
    phi_out = apply_fs_2d(phi_in, ker_fft_x, ker_fft_y, nx, ny, ...
                          n_pad_x, n_fft_x, n_pad_y, n_fft_y);
end
