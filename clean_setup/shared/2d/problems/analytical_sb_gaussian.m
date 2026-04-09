function [rho_ana, mx_ana, my_ana] = analytical_sb_gaussian(problem, vareps)
% ANALYTICAL_SB_GAUSSIAN  Exact Schrödinger bridge solution for equal-variance Gaussians (2D).
%
%   [rho_ana, mx_ana, my_ana] = analytical_sb_gaussian(problem, vareps)
%
%   Problem: N(1/3, sigma^2) -> N(2/3, sigma^2) in x, uniform in y.
%   FP constraint: d_t rho + d_x mx + d_y my = vareps * (d_xx + d_yy) rho.
%
%   Because the problem is separable and uniform in y:
%     rho(t,x,y) = rho_1d(t,x)   (same 1D SB solution, broadcast over y)
%     mx(t,x,y)  = mx_1d(t,x)    (no y-dependence)
%     my(t,x,y)  = 0              (no y-transport)
%
%   Outputs on the staggered grid:
%     rho_ana  (ntm x nx  x ny)   density
%     mx_ana   (nt  x nxm x ny)   x-momentum
%     my_ana   (nt  x nx  x nym)  y-momentum  (zero)

    mu0 = 1/3;   mu1 = 2/3;   sigma = 0.05;

    nt  = problem.nt;   ntm = nt - 1;   dt = problem.dt;
    nx  = problem.nx;   nxm = nx - 1;   dx = problem.dx;
    ny  = problem.ny;   nym = ny - 1;

    Normal = @(x, mu, sig) exp(-0.5*((x - mu)/sig).^2) / (sqrt(2*pi)*sig);

    alpha = sqrt(sigma^4 + vareps^2) - sigma^2;

    % --- rho: times k*dt (k=1..ntm), positions (i-0.5)*dx ---
    t_rho = (1:ntm)' * dt;
    x_rho = ((1:nx) - 0.5) * dx;

    mu_t   = (1 - t_rho)*mu0 + t_rho*mu1;
    sig2_t = sigma^2 + 2*alpha * t_rho .* (1 - t_rho);
    sig_t  = sqrt(sig2_t);

    rho_1d = zeros(ntm, nx);
    for k = 1:ntm
        row = Normal(x_rho, mu_t(k), sig_t(k));
        rho_1d(k, :) = row / (sum(row) * dx);   % sum * dx = 1  (integral normalization)
    end
    rho_ana = repmat(rho_1d, 1, 1, ny);   % (ntm x nx x ny)

    % --- mx: times (k-0.5)*dt (k=1..nt), positions j*dx ---
    t_mx = ((1:nt)' - 0.5) * dt;
    x_mx = (1:nxm) * dx;

    mu_t_mx   = (1 - t_mx)*mu0 + t_mx*mu1;
    sig2_t_mx = sigma^2 + 2*alpha * t_mx .* (1 - t_mx);
    sig_t_mx  = sqrt(sig2_t_mx);

    mx_1d = zeros(nt, nxm);
    for k = 1:nt
        row = Normal(x_mx, mu_t_mx(k), sig_t_mx(k));
        row = row / (sum(row) * dx);             % integral normalization, consistent with rho
        v_k = (mu1 - mu0) + (alpha*(1 - 2*t_mx(k)) - vareps) / sig2_t_mx(k) ...
              * (x_mx - mu_t_mx(k));
        mx_1d(k, :) = row .* v_k;
    end
    mx_ana = repmat(mx_1d, 1, 1, ny);   % (nt x nxm x ny)

    % --- my: zero (no y-transport) ---
    my_ana = zeros(nt, nx, nym);
end
