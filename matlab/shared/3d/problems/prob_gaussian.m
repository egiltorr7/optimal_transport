function prob = prob_gaussian()
% PROB_GAUSSIAN  Transport a 3D Gaussian from mu0 to mu1.
%
%   rho0_func(xx, yy, zz) and rho1_func(xx, yy, zz) accept broadcasting
%   inputs xx (nx x 1 x 1), yy (1 x ny x 1), zz (1 x 1 x nz) and return
%   (nx x ny x nz).
%
%   Default marginals: isotropic Gaussians with sigma=0.07, centred well
%   away from domain boundaries so wall BCs have negligible effect.
%   For machine-zero at walls, use sigma <= 0.07 with mu at least 3*sigma
%   from [0,1]^3 boundaries.

    sig = 0.07;

    Normal3d = @(xx, yy, zz, mux, muy, muz, s) ...
        exp(-((xx - mux).^2 + (yy - muy).^2 + (zz - muz).^2) / (2*s^2)) ...
        / (2*pi*s^2)^(3/2);

    prob.name      = 'gaussian';
    prob.rho0_func = @(xx, yy, zz) Normal3d(xx, yy, zz, 0.3, 0.3, 0.3, sig);
    prob.rho1_func = @(xx, yy, zz) Normal3d(xx, yy, zz, 0.7, 0.7, 0.7, sig);
end
