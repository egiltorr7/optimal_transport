function X = thomas_batch_solve(D_mod, e, F)
% THOMAS_BATCH_SOLVE  Back-substitute K tridiagonal systems precomputed by
%   thomas_batch_precomp.
%
%   X = thomas_batch_solve(D_mod, e, F)
%
%   D_mod: (nt x K)  precomputed modified diagonal (from thomas_batch_precomp)
%   e:     (1  x K)  constant off-diagonal per system
%   F:     (nt x K)  right-hand sides (K simultaneous systems)
%   X:     (nt x K)  solution
%
%   Vectorized over K; sequential in nt.  GPU-compatible via 'like' pattern.

    [nt, K] = size(F);

    % Forward sweep: eliminate lower off-diagonal from RHS
    F_mod      = zeros(nt, K, 'like', F);
    F_mod(1,:) = F(1,:);
    for k = 2:nt
        m          = e ./ D_mod(k-1,:);
        F_mod(k,:) = F(k,:) - m .* F_mod(k-1,:);
    end

    % Back substitution
    X       = zeros(nt, K, 'like', F);
    X(nt,:) = F_mod(nt,:) ./ D_mod(nt,:);
    for k = nt-1:-1:1
        X(k,:) = (F_mod(k,:) - e .* X(k+1,:)) ./ D_mod(k,:);
    end
end
