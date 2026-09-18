function X = dct_rows(x)
% DCT_ROWS  Orthonormal DCT-II applied to each row of x.  GPU-compatible.
%
%   X = dct_rows(x)   x: (m x N) real matrix
%
%   Equivalent to dct(x')' but implemented via FFT so it works on gpuArray
%   without the Signal Processing Toolbox.
%
%   Algorithm:
%     xe = [x, x(:,N:-1:1)]          -- symmetric extension (m x 2N)
%     V  = fft(xe, [], 2)             -- 2N-point row FFT
%     raw = real(V(:,1:N) .* tw)      -- tw = exp(-i*pi*k/(2N)), k=0..N-1
%     X  = raw .* (w/2)               -- orthonormal scaling, w=[1/sqrt(N), sqrt(2/N)*...]

    [~, N] = size(x);
    xe     = [x, x(:,N:-1:1)];
    V      = fft(xe, [], 2);
    k      = 0:N-1;
    tw     = exp(-1i * pi * k / (2*N));
    raw    = real(V(:, 1:N) .* tw);
    w      = [1/sqrt(N), sqrt(2/N) * ones(1, N-1)];
    X      = raw .* (w / 2);
end
