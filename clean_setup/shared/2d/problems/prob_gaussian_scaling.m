function prob = prob_gaussian_scaling()
% PROB_GAUSSIAN_SCALING  Transport a narrow Gaussian to a wider one.
%
%   rho0 = N(0.3, 0.04^2)   (narrow, left-of-centre)
%   rho1 = N(0.7, 0.08^2)   (twice as wide, right-of-centre)
%
%   Tests that the solver handles simultaneous translation and spreading.
%   Analytical SB solution available via analytical_sb_gaussian_general.

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'gaussian_scaling';
    prob.rho0_func = @(xx) Normal(xx, 0.3, 0.04);
    prob.rho1_func = @(xx) Normal(xx, 0.7, 0.08);

    % Store parameters for use with analytical_sb_gaussian_general
    prob.mu0    = 0.3;
    prob.sigma0 = 0.04;
    prob.mu1    = 0.7;
    prob.sigma1 = 0.08;
end
