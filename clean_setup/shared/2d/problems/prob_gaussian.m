function prob = prob_gaussian()
% PROB_GAUSSIAN  Transport a 2D Gaussian from (0.35, 0.35) to (0.65, 0.65).
%
%   rho0_func(xx, yy) and rho1_func(xx, yy) accept broadcasting inputs
%   xx (nx x 1) and yy (1 x ny) and return (nx x ny).
%
%   Marginals are isotropic Gaussians with sigma=0.07, centred well away
%   from the domain boundary so that wall BCs have negligible effect.

    Normal2d = @(xx, yy, mux, muy, sig) ...
        exp(-((xx - mux).^2 + (yy - muy).^2) / (2*sig^2)) / (2*pi*sig^2);

    prob.name      = 'gaussian';
    prob.rho0_func = @(xx, yy) Normal2d(xx, yy, 0.35, 0.35, 0.07);
    prob.rho1_func = @(xx, yy) Normal2d(xx, yy, 0.65, 0.65, 0.07);
end
