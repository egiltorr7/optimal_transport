function [rho_ana, mx_ana] = analytical_sb_gaussian_general(problem, mu0, sigma0, mu1, sigma1, vareps)
% ANALYTICAL_SB_GAUSSIAN_GENERAL  Exact SB solution for any two Gaussians.
%
%   [rho_ana, mx_ana] = analytical_sb_gaussian_general(
%                           problem, mu0, sigma0, mu1, sigma1, vareps)
%
%   Computes the analytical Schrödinger bridge between
%     rho0 = N(mu0, sigma0^2)  and  rho1 = N(mu1, sigma1^2)
%   with FP regularisation:  d_t rho + d_x m = vareps * d_xx rho.
%
%   DERIVATION
%   ----------
%   The BM reference measure has kernel exp(-(x-y)^2 / (4*vareps)).
%   For any boundary Gaussians the SB plan is also Gaussian.  The joint
%   coupling covariance between (x_0, x_1) satisfies the fixed-point
%
%     c^2 + 2*vareps*c - sigma0^2*sigma1^2 = 0
%     =>  c = C01 = sqrt(sigma0^2*sigma1^2 + vareps^2) - vareps
%
%   Setting K = C01 + vareps = sqrt(sigma0^2*sigma1^2 + vareps^2), the
%   marginal variance at time t is (via tower property over the BM bridge):
%
%     sigma_t^2 = (1-t)^2*sigma0^2 + 2*t*(1-t)*K + t^2*sigma1^2
%
%   The velocity field that satisfies the FP equation for this Gaussian
%   flow is affine in x:
%
%     v(t,x) = (mu1-mu0) + b_t * (x - mu_t)
%     b_t    = (C01 - sigma0^2 + t*(sigma0^2 + sigma1^2 - 2*K)) / sigma_t^2
%     mu_t   = (1-t)*mu0 + t*mu1
%
%   Limiting cases:
%     vareps -> 0:  K -> sigma0*sigma1,  C01 -> sigma0*sigma1
%                   sigma_t -> (1-t)*sigma0 + t*sigma1  (McCann interpolation)
%                   b_t     -> (sigma1-sigma0) / sigma_t  (pure OT velocity)
%     sigma0 = sigma1 = sigma:
%                   K = sqrt(sigma^4 + vareps^2),  alpha = K - sigma^2
%                   sigma_t^2 = sigma^2 + 2*alpha*t*(1-t)
%                   b_t = (alpha*(1-2t) - vareps) / sigma_t^2
%                   (reduces to the formula in analytical_sb_gaussian)
%
%   Outputs (same grid convention as analytical_sb_gaussian):
%     rho_ana  (ntm x nx)   density   at times k*dt,        positions (i-0.5)*dx
%     mx_ana   (nt  x nxm)  momentum  at times (k-0.5)*dt,  positions j*dx

    nt  = problem.nt;   ntm = nt - 1;   dt = problem.dt;
    nx  = problem.nx;   nxm = nx - 1;   dx = problem.dx;

    Normal = @(x, mu, sig) exp(-0.5*((x - mu)/sig).^2) / (sqrt(2*pi)*sig);

    K   = sqrt(sigma0^2 * sigma1^2 + vareps^2);
    C01 = K - vareps;

    % --- rho: times k*dt (k=1..ntm), positions (i-0.5)*dx ---
    t_rho = (1:ntm)' * dt;
    x_rho = ((1:nx) - 0.5) * dx;

    mu_t   = (1 - t_rho)*mu0 + t_rho*mu1;
    sig2_t = (1-t_rho).^2 * sigma0^2 + 2*t_rho.*(1-t_rho)*K + t_rho.^2 * sigma1^2;
    sig_t  = sqrt(sig2_t);

    rho_ana = zeros(ntm, nx);
    for k = 1:ntm
        row = Normal(x_rho, mu_t(k), sig_t(k));
        rho_ana(k, :) = row / sum(row);
    end

    % --- mx: times (k-0.5)*dt (k=1..nt), positions j*dx ---
    t_mx = ((1:nt)' - 0.5) * dt;
    x_mx = (1:nxm) * dx;

    mu_t_mx   = (1 - t_mx)*mu0 + t_mx*mu1;
    sig2_t_mx = (1-t_mx).^2 * sigma0^2 + 2*t_mx.*(1-t_mx)*K + t_mx.^2 * sigma1^2;

    mx_ana = zeros(nt, nxm);
    for k = 1:nt
        row = Normal(x_mx, mu_t_mx(k), sqrt(sig2_t_mx(k)));
        row = row / sum(row);
        b_t = (C01 - sigma0^2 + t_mx(k)*(sigma0^2 + sigma1^2 - 2*K)) / sig2_t_mx(k);
        v_k = (mu1 - mu0) + b_t * (x_mx - mu_t_mx(k));
        mx_ana(k, :) = row .* v_k;
    end
end
