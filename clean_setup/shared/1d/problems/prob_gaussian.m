function prob = prob_gaussian()
% PROB_GAUSSIAN  Transport a Gaussian from x=1/3 to x=2/3 on [0,1].

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'gaussian';
    prob.mu0       = 1/3;
    prob.mu1       = 2/3;
    prob.sigma     = 0.15;
    prob.rho0_func = @(xx) Normal(xx, prob.mu0, prob.sigma);
    prob.rho1_func = @(xx) Normal(xx, prob.mu1, prob.sigma);
end
