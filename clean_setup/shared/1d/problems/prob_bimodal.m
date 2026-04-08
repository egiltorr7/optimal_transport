function prob = prob_bimodal()
% PROB_BIMODAL  Transport a bimodal distribution to a single Gaussian.

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'bimodal';
    prob.rho0_func = @(xx) Normal(xx, 0.25, 0.05) + Normal(xx, 0.75, 0.05);
    prob.rho1_func = @(xx) Normal(xx, 0.5,  0.08);
end
