function H = level_hessian_fd(p_l, m_l, vareps, dx, h_rel)
% LEVEL_HESSIAN_FD  Central-difference Hessian of LEVEL_OBJECTIVE's
%   analytic gradient, treating [p_l, m_l] jointly as free variables.
%
%   H = level_hessian_fd(p_l, m_l, vareps, dx, h_rel)
%
%   Returns the full (2*nx-1) x (2*nx-1) symmetric Hessian, ordered
%   [p_l, m_l]. If p_l is actually a fixed boundary value at the calling
%   level (i.e. only m_l is free), the caller should extract the trailing
%   (nx-1) x (nx-1) m-m block instead of using this matrix directly.
%
%   Deliberately uses finite differences of the (already gradient-checked)
%   analytic gradient rather than a hand-derived closed-form Hessian: the
%   K/Fisher-information cross terms are easy to get subtly wrong by hand,
%   and this problem is small enough per time level (size ~2*nx) that an
%   O(nx) central-difference sweep per Newton iteration is cheap.
%
%   h_rel is a RELATIVE step (default 1e-6): the actual per-variable
%   perturbation is h_k = h_rel * max(|x_k|, 1e-8). The Fisher-information
%   term's curvature scales like 1/p^2, so densities can differ by orders
%   of magnitude in scale across a single level (near a floor value vs.
%   near the peak); a single fixed ABSOLUTE step is badly mis-scaled for
%   one end or the other of that range and injects avoidable truncation/
%   roundoff noise into H, which shows up as poor KKT conditioning.

    if nargin < 5, h_rel = 1e-6; end
    h_floor = 1e-8;

    np = numel(p_l);
    nm = numel(m_l);
    n  = np + nm;
    H  = zeros(n, n);

    x0 = [p_l, m_l];
    h  = h_rel * max(abs(x0), h_floor);   % per-variable absolute step

    for k = 1:n
        pp = p_l; mp = m_l;
        pm = p_l; mm = m_l;
        hk = h(k);
        if k <= np
            pp(k) = pp(k) + hk;
            pm(k) = pm(k) - hk;
        else
            j = k - np;
            mp(j) = mp(j) + hk;
            mm(j) = mm(j) - hk;
        end
        [~, gpp, gpm] = level_objective(pp, mp, vareps, dx);
        [~, gmp, gmm] = level_objective(pm, mm, vareps, dx);
        gplus  = [gpp, gpm];
        gminus = [gmp, gmm];
        H(:, k) = (gplus - gminus)' / (2 * hk);
    end

    H = 0.5 * (H + H');   % symmetrize away finite-difference asymmetry
end
