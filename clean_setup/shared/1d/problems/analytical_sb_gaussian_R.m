function rho_ana = analytical_sb_gaussian_R(problem, vareps)
% ANALYTICAL_SB_GAUSSIAN_R  Exact Schrödinger bridge solution on R for equal-variance Gaussians.
%
%   rho_ana = analytical_sb_gaussian_R(problem, vareps)
%
%   Same closed-form marginal as analytical_sb_gaussian, but returned as
%   probability MASS (PDF * dx) evaluated at cell centres, WITHOUT
%   renormalising over [0,1].  This is the correct comparison target when
%   using the free-space heat kernel (precomp_heat_free_space), where mass
%   can diffuse outside the domain.
%
%   Problem: N(1/3, sigma^2) -> N(2/3, sigma^2),  sigma = 0.05.
%
%   Marginal at time t:
%     rho(t,x) = N(mu_t, sigma_t^2)
%     mu_t      = (1-t)*mu0 + t*mu1
%     sigma_t^2 = sigma^2 + 2*alpha*t*(1-t)
%     alpha     = sqrt(sigma^4 + vareps^2) - sigma^2
%
%   Output:
%     rho_ana  (nt+1 x nx)   probability mass at edge times 0, dt, ..., T
%                            rho_ana(k,i) = N(xx(i); mu_t, sigma_t^2) * dx
%
%   Each entry is the raw PDF value at the cell centre — NOT multiplied by dx
%   and NOT renormalised over [0,1].  This matches the 'use_pdf_marginals'
%   convention in sinkhorn_hopf_cole: sum(rho_ana(k,:)) ~ 1/dx * P(X in [0,1]),
%   which drops below 1/dx when the Gaussian spreads outside [0,1] (e.g. for
%   vareps=1, sigma_t ~ 0.7 at t=0.5 so sum ~ 0.52/dx).

    mu0   = 1/3;   mu1 = 2/3;   sigma = 0.05;
    nt    = problem.nt;
    dt    = problem.dt;
    nx    = problem.nx;
    dx    = problem.dx;
    xx    = problem.xx;

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);
    alpha  = sqrt(sigma^4 + vareps^2) - sigma^2;

    t_grid  = (0:nt)' * dt;
    rho_ana = zeros(nt+1, nx);
    for k = 1:(nt+1)
        t_k   = t_grid(k);
        mu_k  = (1 - t_k)*mu0 + t_k*mu1;
        sig_k = sqrt(sigma^2 + 2*alpha*t_k*(1 - t_k));
        rho_ana(k,:) = Normal(xx, mu_k, sig_k);   % raw PDF value at each cell centre
    end
end
