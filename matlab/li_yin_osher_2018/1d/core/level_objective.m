function [obj, grad_p, grad_m] = level_objective(p_l, m_l, vareps, dx)
% LEVEL_OBJECTIVE  Kinetic energy + Fisher information for ONE time level.
%
%   [obj, grad_p, grad_m] = level_objective(p_l, m_l, vareps, dx)
%
%   Implements, for a single time level, the summand of Li-Yin-Osher
%   (2018) Eq. (7):
%
%       obj = sum_e  m_l(e)^2 / g(e)  +  vareps^2 * d(e)^2 * g(e)
%
%       g(e) = (p_l(e) + p_l(e+1)) / 2          edge-averaged density
%       d(e) = (log p_l(e+1) - log p_l(e)) / dx  discrete log-gradient
%
%   Inputs:
%     p_l  (1 x nx)     node densities at this time level (must be > 0)
%     m_l  (1 x nx-1)   edge fluxes at this time level
%     vareps            Fisher-information / diffusion strength (beta)
%     dx                spatial grid spacing
%
%   Outputs:
%     obj              scalar
%     grad_p  (1 x nx)     d(obj)/d(p_l)   [only meaningful if p_l is free]
%     grad_m  (1 x nx-1)   d(obj)/d(m_l)
%
%   Note: obj -> +Inf as any p_l(j) -> 0 (Li-Yin-Osher Lemma 2), so any
%   caller doing a line search must keep p_l strictly positive.

    nx = numel(p_l);
    pL = p_l(1:nx-1);
    pR = p_l(2:nx);
    g  = 0.5 * (pL + pR);
    d  = (log(pR) - log(pL)) / dx;

    K_edge = (m_l.^2) ./ g;
    I_edge = vareps^2 * (d.^2) .* g;
    obj = sum(K_edge) + sum(I_edge);

    if nargout > 1
        grad_m = 2 * m_l ./ g;

        % Common d(K_edge)/dp_left = d(K_edge)/dp_right, since g is the
        % symmetric average of the two endpoint densities.
        dK_dg = -0.5 * (m_l.^2) ./ (g.^2);

        dI_dleft  = vareps^2 * ( -2 * d .* g ./ (dx * pL) + 0.5 * d.^2 );
        dI_dright = vareps^2 * (  2 * d .* g ./ (dx * pR) + 0.5 * d.^2 );

        left_contrib  = zeros(1, nx);
        right_contrib = zeros(1, nx);
        left_contrib(1:nx-1) = dK_dg + dI_dleft;
        right_contrib(2:nx)  = dK_dg + dI_dright;

        grad_p = left_contrib + right_contrib;
    end
end
