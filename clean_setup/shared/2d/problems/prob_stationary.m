function prob = prob_stationary()
% PROB_STATIONARY  Trivial transport: rho0 = rho1 = N(0.5, 0.05^2).
%
%   For eps=0 (OT) the exact solution is:
%     rho(t,x) = rho0(x)  for all t,  m = 0,  KE = 0.
%   Any nonzero KE or momentum is a solver artefact.
%
%   For eps>0 (SB) the solution is the "looping" Gaussian SB between
%   identical marginals — the process first spreads (diffusion) then
%   contracts back.  The analytical solution is available via
%   analytical_sb_gaussian_general with mu0=mu1, sigma0=sigma1.

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'stationary';
    prob.rho0_func = @(xx) Normal(xx, 0.5, 0.05);
    prob.rho1_func = @(xx) Normal(xx, 0.5, 0.05);

    prob.mu0    = 0.5;   prob.sigma0 = 0.05;
    prob.mu1    = 0.5;   prob.sigma1 = 0.05;
end
