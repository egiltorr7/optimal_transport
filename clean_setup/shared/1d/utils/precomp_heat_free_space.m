function heat = precomp_heat_free_space(problem, cfg)
% PRECOMP_HEAT_FREE_SPACE  Precompute heat kernels for R^n (Gaussian convolution).
%
%   heat = precomp_heat_free_space(problem, cfg)
%
%   Reference process: standard Brownian motion on R (free space).
%
%   Why one-shot instead of step-by-step
%   -------------------------------------
%   In the Sinkhorn loop we need to propagate phi_0 forward over the full
%   time T and psi_T backward over T.  phi_0 and psi_T are determined by
%   the marginal conditions, so they are concentrated near the support of
%   rho0 / rho1 (narrow Gaussians on [0,1]).  A single convolution of a
%   narrow function with K_T is accurate with zero-padding because the
%   function is nearly zero outside [0,1].
%
%   Step-by-step propagation (nt small steps) is WRONG for large eps: after
%   k steps phi_k has spread wide and is no longer concentrated on [0,1],
%   so zero-padding it before step k+1 truncates real values and the error
%   compounds over nt steps.
%
%   Inputs:
%     problem   struct with fields: nx, dx, dt, nt
%     cfg       struct with fields: vareps
%
%   Output:
%     heat.apply_full    phi_out = heat.apply_full(phi_in)
%                        Apply K_T (full time T=1) — used in Sinkhorn loop.
%     heat.apply_time    phi_out = heat.apply_time(phi_in, t)
%                        Apply K_t for arbitrary t — used for trajectory recovery.
%     heat.name          'free_space'

    dx     = problem.dx;
    nx     = problem.nx;
    vareps = cfg.vareps;

    % Precompute the full-time kernel K_T (used in every Sinkhorn iteration)
    sigma2_T = 2 * vareps * 1.0;   % variance for t = T = 1
    [ker_fft_T, n_pad_T, n_fft_T] = build_kernel(sigma2_T, dx, nx);

    heat.apply_full = @(phi) apply_fs(phi(:), ker_fft_T, nx, n_pad_T, n_fft_T);
    heat.apply_time = @(phi, t) apply_free_space_t(phi(:), vareps, t, dx, nx);
    heat.name       = 'free_space';
end

% -------------------------------------------------------------------------

function [ker_fft, n_pad, n_fft] = build_kernel(sigma2, dx, nx)
% Build and FFT-transform the Gaussian kernel for variance sigma2.
    n_pad = ceil(6 * sqrt(sigma2) / dx);
    n_fft = 2^nextpow2(nx + 2*n_pad);
    jj    = ((0:n_fft-1)' - floor(n_fft/2));
    ker   = exp(-((jj * dx).^2) / (2 * sigma2));
    ker   = ker / sum(ker);
    ker_fft = fft(ifftshift(ker));
end

function phi_out = apply_fs(phi_in, ker_fft, nx, n_pad, n_fft)
% Apply a precomputed Gaussian kernel via zero-padded FFT convolution.
    sig = zeros(n_fft, 1);
    sig(n_pad+1 : n_pad+nx) = phi_in;
    phi_conv = real(ifft(fft(sig) .* ker_fft));
    phi_out  = max(phi_conv(n_pad+1 : n_pad+nx), 0);
end

function phi_out = apply_free_space_t(phi_in, vareps, t, dx, nx)
% Apply K_t for an arbitrary time t (kernel recomputed each call).
    if t == 0
        phi_out = max(phi_in, 0);
        return;
    end
    sigma2 = 2 * vareps * t;
    [ker_fft, n_pad, n_fft] = build_kernel(sigma2, dx, nx);
    phi_out = apply_fs(phi_in, ker_fft, nx, n_pad, n_fft);
end
