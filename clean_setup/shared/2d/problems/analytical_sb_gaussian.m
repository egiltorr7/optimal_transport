function [rho_ana, mx_ana, my_ana] = analytical_sb_gaussian(problem, vareps)
% ANALYTICAL_SB_GAUSSIAN  Exact Schrödinger bridge solution for 2D isotropic Gaussians.
%
%   [rho_ana, mx_ana, my_ana] = analytical_sb_gaussian(problem, vareps)
%
%   Problem: N(mu0, sigma^2*I) -> N(mu1, sigma^2*I) with equal isotropic covariance.
%   Because the covariance is isotropic and equal at both ends, the SB solution is:
%
%     rho(t,x,y) = N_2d( mu_t, sigma_t^2 * I )
%     mu_t       = (1-t)*mu0 + t*mu1
%     sigma_t^2  = sigma^2 + 2*alpha*t*(1-t),   alpha = sqrt(sigma^4 + eps^2) - sigma^2
%
%   Velocity field:
%     v_x(t,x,y) = (mu1_x - mu0_x) + [alpha*(1-2t) - eps]/sigma_t^2 * (x - mu_t_x)
%     v_y(t,x,y) = (mu1_y - mu0_y) + [alpha*(1-2t) - eps]/sigma_t^2 * (y - mu_t_y)
%
%   Outputs on the staggered grid:
%     rho_ana  (ntm x nx  x ny)   density       at times k*dt,       positions cell-centre
%     mx_ana   (nt  x nxm x ny)   x-momentum    at times (k-0.5)*dt, positions x-staggered
%     my_ana   (nt  x nx  x nym)  y-momentum    at times (k-0.5)*dt, positions y-staggered

    mu0_x = 0.35;  mu0_y = 0.35;
    mu1_x = 0.65;  mu1_y = 0.65;
    sigma = 0.07;

    nt  = problem.nt;   ntm = nt - 1;   dt = problem.dt;
    nx  = problem.nx;   nxm = nx - 1;   dx = problem.dx;
    ny  = problem.ny;   nym = ny - 1;   dy = problem.dy;

    alpha = sqrt(sigma^4 + vareps^2) - sigma^2;

    % --- rho: times k*dt (k=1..ntm), cell-centre positions ---
    t_r  = reshape((1:ntm)' * dt,           ntm, 1,  1 );
    x_r  = reshape(((1:nx) - 0.5) * dx,     1,   nx, 1 );
    y_r  = reshape(((1:ny) - 0.5) * dy,     1,   1,  ny);

    mu_t_x  = (1 - t_r)*mu0_x + t_r*mu1_x;   % (ntm x 1 x 1)
    mu_t_y  = (1 - t_r)*mu0_y + t_r*mu1_y;
    sig2_t  = sigma^2 + 2*alpha * t_r .* (1 - t_r);

    rho_ana = exp(-((x_r - mu_t_x).^2 + (y_r - mu_t_y).^2) ./ (2*sig2_t));
    rho_ana = rho_ana ./ (sum(sum(rho_ana, 2), 3) * dx * dy);   % (ntm x nx x ny)

    % --- mx: times (k-0.5)*dt (k=1..nt), x-staggered positions ---
    t_m   = reshape(((1:nt)' - 0.5) * dt,   nt,  1,  1 );
    x_mx  = reshape((1:nxm) * dx,            1,   nxm,1 );
    y_mx  = reshape(((1:ny) - 0.5) * dy,     1,   1,  ny);

    mu_t_x_m  = (1 - t_m)*mu0_x + t_m*mu1_x;
    mu_t_y_m  = (1 - t_m)*mu0_y + t_m*mu1_y;
    sig2_t_m  = sigma^2 + 2*alpha * t_m .* (1 - t_m);

    rho_mx  = exp(-((x_mx - mu_t_x_m).^2 + (y_mx - mu_t_y_m).^2) ./ (2*sig2_t_m));
    rho_mx  = rho_mx ./ (sum(sum(rho_mx, 2), 3) * dx * dy);
    v_x     = (mu1_x - mu0_x) + (alpha*(1 - 2*t_m) - vareps) ./ sig2_t_m .* (x_mx - mu_t_x_m);
    mx_ana  = rho_mx .* v_x;   % (nt x nxm x ny)

    % --- my: times (k-0.5)*dt (k=1..nt), y-staggered positions ---
    x_my  = reshape(((1:nx) - 0.5) * dx,     1,   nx, 1  );
    y_my  = reshape((1:nym) * dy,             1,   1,  nym);

    rho_my  = exp(-((x_my - mu_t_x_m).^2 + (y_my - mu_t_y_m).^2) ./ (2*sig2_t_m));
    rho_my  = rho_my ./ (sum(sum(rho_my, 2), 3) * dx * dy);
    v_y     = (mu1_y - mu0_y) + (alpha*(1 - 2*t_m) - vareps) ./ sig2_t_m .* (y_my - mu_t_y_m);
    my_ana  = rho_my .* v_y;   % (nt x nx x nym)
end
