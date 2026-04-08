function prob = prob_gaussian_pure_scaling()
% PROB_GAUSSIAN_PURE_SCALING  Spread a Gaussian in place: same centre, wider.
%
%   rho0 = N(0.5, 0.03^2)   (narrow)
%   rho1 = N(0.5, 0.10^2)   (wide)
%
%   No translation; the velocity field is purely radial (symmetric about
%   x=0.5).  The centre of mass must stay at 0.5 for all t.
%
%   For eps=0 (OT):
%     W_2^2 = (sigma1 - sigma0)^2 = 0.07^2 = 0.0049
%     sigma_t = (1-t)*0.03 + t*0.10  (linear interpolation of std-devs)
%     v(t,x) = (sigma1-sigma0)/sigma_t * (x - 0.5)
%
%   Analytical SB solution available via analytical_sb_gaussian_general
%   with mu0=mu1=0.5.

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'gaussian_pure_scaling';
    prob.rho0_func = @(xx) Normal(xx, 0.5, 0.03);
    prob.rho1_func = @(xx) Normal(xx, 0.5, 0.10);

    prob.mu0    = 0.5;   prob.sigma0 = 0.03;
    prob.mu1    = 0.5;   prob.sigma1 = 0.10;
end
