function D_mod = thomas_batch_precomp(D, e)
% THOMAS_BATCH_PRECOMP  Thomas forward sweep for K symmetric tridiagonal systems.
%
%   D_mod = thomas_batch_precomp(D, e)
%
%   D:     (nt x K)  diagonals, column k = diagonal of system k
%   e:     (1  x K)  constant off-diagonal per system (same for all nt-1 entries)
%   D_mod: (nt x K)  modified diagonal after forward elimination
%
%   Each system T_k has tridiagonal structure with diagonal D(:,k) and
%   constant off-diagonal e(k).  The forward sweep computes D_mod such
%   that thomas_batch_solve(D_mod, e, F) solves T_k * X(:,k) = F(:,k)
%   for all k simultaneously.
%
%   Fully vectorized over K: each step is an element-wise operation (1 x K).
%   GPU-compatible (works on gpuArray inputs).

    [nt, ~] = size(D);
    D_mod   = D;
    for k = 2:nt
        m          = e ./ D_mod(k-1, :);
        D_mod(k,:) = D_mod(k,:) - m .* e;
    end
end
