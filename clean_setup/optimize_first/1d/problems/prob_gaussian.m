function prob = prob_gaussian()
% PROB_GAUSSIAN  Transport a Gaussian from x=1/3 to x=2/3.

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'gaussian';
    prob.rho0_func = @(xx) Normal(xx, 1/3, 0.05);
    prob.rho1_func = @(xx) Normal(xx, 2/3, 0.05);
end
