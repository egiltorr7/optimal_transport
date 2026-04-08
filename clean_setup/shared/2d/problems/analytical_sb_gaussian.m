function [rho_ana, mx_ana] = analytical_sb_gaussian(problem, vareps)
% ANALYTICAL_SB_GAUSSIAN  Exact Schrödinger bridge solution for equal-variance Gaussians.
%
%   [rho_ana, mx_ana] = analytical_sb_gaussian(problem, vareps)
%
%   Problem: N(1/3, sigma^2) -> N(2/3, sigma^2),  sigma = 0.05.
%   FP constraint: d_t rho + d_x m = vareps * d_xx rho.
%
%   The SB marginal is Gaussian at every time:
%     rho(t,x) = N(mu_t, sigma_t^2)
%     mu_t     = (1-t)*mu0 + t*mu1              (linear interpolation)
%     sigma_t^2 = sigma^2 + 2*alpha*t*(1-t)
%     alpha    = sqrt(sigma^4 + vareps^2) - sigma^2
%
%   The coupling covariance under the SB plan is
%     C01 = sqrt(sigma^4 + vareps^2) - vareps
%   (derived from the BM transition kernel exp(-(x1-x0)^2 / (4*vareps))).
%
%   The velocity field is affine in x:
%     v(t,x) = (mu1-mu0) + (alpha*(1-2t) - vareps) / sigma_t^2 * (x - mu_t)
%   which reduces to the constant v = mu1-mu0 when vareps -> 0 (OT case).
%
%   Outputs on the staggered grid (same convention as analytical_gaussian):
%     rho_ana  (ntm x nx)   density   at times k*dt,        positions (i-0.5)*dx
%     mx_ana   (nt  x nxm)  momentum  at times (k-0.5)*dt,  positions j*dx

    mu0 = 1/3;   mu1 = 2/3;   sigma = 0.05;

    nt  = problem.nt;   ntm = nt - 1;   dt = problem.dt;
    nx  = problem.nx;   nxm = nx - 1;   dx = problem.dx;

    Normal = @(x, mu, sig) exp(-0.5*((x - mu)/sig).^2) / (sqrt(2*pi)*sig);

    % alpha: excess variance amplitude due to diffusion
    alpha = sqrt(sigma^4 + vareps^2) - sigma^2;

    % --- rho: times k*dt (k=1..ntm), positions (i-0.5)*dx ---
    t_rho = (1:ntm)' * dt;
    x_rho = ((1:nx) - 0.5) * dx;

    mu_t  =  (1 - t_rho)*mu0 + t_rho*mu1;
    sig2_t = sigma^2 + 2*alpha * t_rho .* (1 - t_rho);
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
    sig2_t_mx = sigma^2 + 2*alpha * t_mx .* (1 - t_mx);
    sig_t_mx  = sqrt(sig2_t_mx);

    mx_ana = zeros(nt, nxm);
    for k = 1:nt
        row = Normal(x_mx, mu_t_mx(k), sig_t_mx(k));
        row = row / sum(row);
        % velocity: translational part + diffusion-driven spreading/squeezing
        v_k = (mu1 - mu0) + (alpha*(1 - 2*t_mx(k)) - vareps) / sig2_t_mx(k) ...
              * (x_mx - mu_t_mx(k));
        mx_ana(k, :) = row .* v_k;
    end
end
