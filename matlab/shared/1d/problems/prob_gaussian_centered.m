function prob = prob_gaussian_centered(L)
% PROB_GAUSSIAN_CENTERED  Transport a Gaussian on [0,L], centered in the domain.
%
%   prob = prob_gaussian_centered(L)
%
%   Places mu0 = L/2 - 1/6 and mu1 = L/2 + 1/6 (same absolute half-distance
%   as prob_gaussian on [0,1]) with sigma = 0.15.  On large domains (L >= 4)
%   both Gaussians are more than 5*sigma_max away from both boundaries at all
%   times, so boundary effects are negligible for any eps.

    if nargin < 1, L = 1; end

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'gaussian_centered';
    prob.mu0       = L/2 - 1/6;
    prob.mu1       = L/2 + 1/6;
    prob.sigma     = 0.15;
    prob.rho0_func = @(xx) Normal(xx, prob.mu0, prob.sigma);
    prob.rho1_func = @(xx) Normal(xx, prob.mu1, prob.sigma);
end
