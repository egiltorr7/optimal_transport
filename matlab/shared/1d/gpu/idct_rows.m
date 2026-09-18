function x = idct_rows(X)
% IDCT_ROWS  Inverse orthonormal DCT-II (= DCT-III) applied to each row.  GPU-compatible.
%
%   x = idct_rows(X)   X: (m x N) real matrix of DCT-II coefficients
%
%   Equivalent to idct(X')' but implemented via FFT so it works on gpuArray.
%
%   Algorithm:
%     Z = X .* w          -- w=[1/sqrt(N), sqrt(2/N)*...], (WD)^{-1} = (WD)^T
%     U = Z .* tw         -- tw = exp(i*pi*k/(2N))
%     x = real(ifft([U, zeros(m,N)], [], 2)) * 2N  -- zero-pad IFFT
%     x = x(:, 1:N)

    [m, N] = size(X);
    w      = [1/sqrt(N), sqrt(2/N) * ones(1, N-1)];
    Z      = X .* w;
    k      = 0:N-1;
    U      = complex(Z) .* exp(1i * pi * k / (2*N));
    x      = real(ifft([U, zeros(m, N, 'like', U)], [], 2)) * (2*N);
    x      = x(:, 1:N);
end
