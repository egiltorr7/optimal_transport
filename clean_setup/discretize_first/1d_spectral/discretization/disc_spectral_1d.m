function ops = disc_spectral_1d(problem)
% DISC_SPECTRAL_1D  Spectral (FFT) operators for 1D OT / SB on [0,1].
%
%   ops = disc_spectral_1d(problem)
%
%   All variables (rho, mx) live on the collocated (nt x nx) grid at
%   cell-center positions x_j = (j - 0.5)/nx, j = 1,...,nx.
%   Time discretization (FE or BE) is handled separately in the projection;
%   this function provides only the spatial spectral operators.
%
%   Domain: [0, 1] periodic (L = 1).
%   Wavenumbers: omega_k = 2*pi * [0,1,...,nx/2,-nx/2+1,...,-1]
%
%   ops fields:
%     wavenumbers  (1 x nx)   angular wavenumbers omega_k
%     fft_x        @(u)       FFT along x (dim 2): (nt x nx) -> complex
%     ifft_x       @(u_hat)   IFFT along x (dim 2): complex -> real
%     deriv_x      @(u)       spectral d/dx:  u -> real(IFFT(i*omega .* FFT(u)))
%     deriv_xx     @(u)       spectral d^2/dx^2: u -> real(IFFT(-omega^2 .* FFT(u)))

    nx = problem.nx;

    % Angular wavenumbers for FFT on [0,1] (L=1)
    k_idx = [0 : floor(nx/2), -floor((nx-1)/2) : -1];   % (1 x nx)
    omega  = 2 * pi * k_idx;                              % (1 x nx)

    % For even nx the Nyquist mode (k=nx/2) is shared between +/- frequencies.
    % Multiplying by 1i*omega maps it to a purely imaginary value that
    % real(ifft) silently discards, making deriv_x o deriv_x != deriv_xx.
    % Zero it out so all spectral operators are mutually consistent.
    if mod(nx, 2) == 0
        omega(nx/2 + 1) = 0;
    end

    ops.wavenumbers = omega;

    ops.fft_x  = @(u) fft(u, [], 2);
    ops.ifft_x = @(u_hat) real(ifft(u_hat, [], 2));

    ops.deriv_x  = @(u) real(ifft( bsxfun(@times, fft(u, [], 2),  1i * omega), [], 2));
    ops.deriv_xx = @(u) real(ifft( bsxfun(@times, fft(u, [], 2), -omega.^2),   [], 2));

end
