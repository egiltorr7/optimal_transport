function [rho_ana, mx_ana] = analytical_gaussian(problem)
% ANALYTICAL_GAUSSIAN  Exact solution for Gaussian-to-Gaussian OT (eps=0).
%
%   [rho_ana, mx_ana] = analytical_gaussian(problem)
%
%   Problem: N(1/3, 0.05^2) -> N(2/3, 0.05^2)
%   Equal variances => pure translation; velocity is constant:
%     v(t, x) = mu1 - mu0  (independent of t and x)
%
%   The displacement interpolation gives:
%     rho(t, x) = N(mu_t, sigma_t^2),  mu_t = (1-t)*mu0 + t*mu1,
%                                       sigma_t = (1-t)*sigma0 + t*sigma1
%     m(t, x)   = rho(t, x) * v
%
%   Outputs are evaluated on the staggered grid and normalized consistently
%   with setup_problem (each time slice sums to 1).
%
%   Outputs:
%     rho_ana  (ntm x nx)   density  at times k*dt,       positions (i-0.5)*dx
%     mx_ana   (nt  x nxm)  momentum at times (k-0.5)*dt, positions j*dx

    mu0    = 1/3;   sigma0 = 0.05;
    mu1    = 2/3;   sigma1 = 0.05;

    nt  = problem.nt;   ntm = nt - 1;   dt = problem.dt;
    nx  = problem.nx;   nxm = nx - 1;   dx = problem.dx;

    Normal = @(x, mu, sig) exp(-0.5*((x - mu)/sig).^2) / (sqrt(2*pi)*sig);

    % --- rho: times k*dt (k=1..ntm), positions (i-0.5)*dx ---
    t_rho = (1:ntm)' * dt;
    x_rho = ((1:nx) - 0.5) * dx;

    mu_t  = (1 - t_rho)*mu0 + t_rho*mu1;
    sig_t = (1 - t_rho)*sigma0 + t_rho*sigma1;

    rho_ana = zeros(ntm, nx);
    for k = 1:ntm
        row = Normal(x_rho, mu_t(k), sig_t(k));
        rho_ana(k, :) = row / sum(row);
    end

    % --- mx: times (k-0.5)*dt (k=1..nt), positions j*dx ---
    t_mx = ((1:nt)' - 0.5) * dt;
    x_mx = (1:nxm) * dx;

    mu_t_mx  = (1 - t_mx)*mu0 + t_mx*mu1;
    sig_t_mx = (1 - t_mx)*sigma0 + t_mx*sigma1;

    v = mu1 - mu0;   % constant velocity (equal-sigma case)

    mx_ana = zeros(nt, nxm);
    for k = 1:nt
        row = Normal(x_mx, mu_t_mx(k), sig_t_mx(k));
        row = row / sum(row);
        mx_ana(k, :) = row * v;
    end
end
