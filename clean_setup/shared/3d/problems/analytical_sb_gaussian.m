function [rho_ana, mx_ana, my_ana, mz_ana] = analytical_sb_gaussian(problem, vareps, mu0, mu1, sigma)
% ANALYTICAL_SB_GAUSSIAN  Exact SB solution for 3D isotropic Gaussians.
%
%   [rho, mx, my, mz] = analytical_sb_gaussian(problem, vareps, mu0, mu1, sigma)
%
%   Problem: N(mu0, sigma^2*I) -> N(mu1, sigma^2*I), equal isotropic covariance.
%   Solution:
%     rho(t,x,y,z) = N_3d( mu_t, sigma_t^2 * I )
%     mu_t         = (1-t)*mu0 + t*mu1
%     sigma_t^2    = sigma^2 + 2*alpha*t*(1-t),  alpha = sqrt(sigma^4 + eps^2) - sigma^2
%
%   Velocity field:
%     v_i(t,x) = (mu1_i - mu0_i) + [alpha*(1-2t) - eps]/sigma_t^2 * (xi - mu_t_i)
%
%   Outputs on the staggered grid:
%     rho_ana  (ntm x nx  x ny  x nz )   density at times k*dt
%     mx_ana   (nt  x nxm x ny  x nz )   x-momentum at half-integer times
%     my_ana   (nt  x nx  x nym x nz )   y-momentum
%     mz_ana   (nt  x nx  x ny  x nzm)   z-momentum
%
%   mu0, mu1: 3-element vectors [x, y, z]
%   sigma:    scalar isotropic standard deviation

    nt  = problem.nt;   ntm = nt - 1;   dt = problem.dt;
    nx  = problem.nx;   nxm = nx - 1;   dx = problem.dx;
    ny  = problem.ny;   nym = ny - 1;   dy = problem.dy;
    nz  = problem.nz;   nzm = nz - 1;   dz = problem.dz;
    dV  = dx * dy * dz;

    alpha = sqrt(sigma^4 + vareps^2) - sigma^2;

    Normal3d = @(x, y, z, mux, muy, muz, s2) ...
        exp(-((x - mux).^2 + (y - muy).^2 + (z - muz).^2) / (2*s2));

    % --- rho: times k*dt (k=1..ntm), cell-centre positions ---
    t_r = reshape((1:ntm)' * dt,       ntm, 1,  1,  1 );
    x_r = reshape(((1:nx) - 0.5) * dx, 1,   nx, 1,  1 );
    y_r = reshape(((1:ny) - 0.5) * dy, 1,   1,  ny, 1 );
    z_r = reshape(((1:nz) - 0.5) * dz, 1,   1,  1,  nz);

    mu_t_x = (1 - t_r)*mu0(1) + t_r*mu1(1);
    mu_t_y = (1 - t_r)*mu0(2) + t_r*mu1(2);
    mu_t_z = (1 - t_r)*mu0(3) + t_r*mu1(3);
    sig2_t  = sigma^2 + 2*alpha * t_r .* (1 - t_r);

    rho_raw = Normal3d(x_r, y_r, z_r, mu_t_x, mu_t_y, mu_t_z, sig2_t);
    nrm     = sum(rho_raw, [2,3,4]) * dV;
    rho_ana = rho_raw ./ nrm;   % (ntm x nx x ny x nz)

    % --- momentum at half-integer times ---
    t_m  = reshape(((1:nt)' - 0.5) * dt, nt, 1, 1, 1);
    mu_m_x = (1 - t_m)*mu0(1) + t_m*mu1(1);
    mu_m_y = (1 - t_m)*mu0(2) + t_m*mu1(2);
    mu_m_z = (1 - t_m)*mu0(3) + t_m*mu1(3);
    sig2_m  = sigma^2 + 2*alpha * t_m .* (1 - t_m);
    vel_coef = (alpha*(1 - 2*t_m) - vareps) ./ sig2_m;

    % mx: positions x-staggered
    x_mx = reshape((1:nxm) * dx, 1, nxm, 1, 1);
    y_mx = reshape(((1:ny) - 0.5) * dy, 1, 1, ny, 1);
    z_mx = reshape(((1:nz) - 0.5) * dz, 1, 1, 1, nz);
    rho_mx  = Normal3d(x_mx, y_mx, z_mx, mu_m_x, mu_m_y, mu_m_z, sig2_m);
    nrm_mx  = sum(rho_mx, [2,3,4]) * dV;
    rho_mx  = rho_mx ./ nrm_mx;
    v_x     = (mu1(1) - mu0(1)) + vel_coef .* (x_mx - mu_m_x);
    mx_ana  = rho_mx .* v_x;

    % my: positions y-staggered
    x_my = reshape(((1:nx) - 0.5) * dx, 1, nx, 1, 1);
    y_my = reshape((1:nym) * dy, 1, 1, nym, 1);
    z_my = reshape(((1:nz) - 0.5) * dz, 1, 1, 1, nz);
    rho_my  = Normal3d(x_my, y_my, z_my, mu_m_x, mu_m_y, mu_m_z, sig2_m);
    nrm_my  = sum(rho_my, [2,3,4]) * dV;
    rho_my  = rho_my ./ nrm_my;
    v_y     = (mu1(2) - mu0(2)) + vel_coef .* (y_my - mu_m_y);
    my_ana  = rho_my .* v_y;

    % mz: positions z-staggered
    x_mz = reshape(((1:nx) - 0.5) * dx, 1, nx, 1, 1);
    y_mz = reshape(((1:ny) - 0.5) * dy, 1, 1, ny, 1);
    z_mz = reshape((1:nzm) * dz, 1, 1, 1, nzm);
    rho_mz  = Normal3d(x_mz, y_mz, z_mz, mu_m_x, mu_m_y, mu_m_z, sig2_m);
    nrm_mz  = sum(rho_mz, [2,3,4]) * dV;
    rho_mz  = rho_mz ./ nrm_mz;
    v_z     = (mu1(3) - mu0(3)) + vel_coef .* (z_mz - mu_m_z);
    mz_ana  = rho_mz .* v_z;
end
