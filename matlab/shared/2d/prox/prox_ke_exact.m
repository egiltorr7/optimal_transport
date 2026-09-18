function x_out = prox_ke_exact(x_in, sigma, problem)
% PROX_KE_EXACT  Exact closed-form proximal operator for the kinetic energy (2D).
%
%   x_out = prox_ke_exact(x_in, sigma, problem)
%
%   Solves:
%     min_{rho,mx,my}  integral (mx^2+my^2)/(2*rho) dt dx dy
%                      + (1/(2*sigma))*||(rho,mx,my) - x_in||^2
%
%   via the exact cubic-root formula. Interpolates to cell centers (phi-locations),
%   applies the prox, then interpolates back to staggered locations.
%
%   Inputs:
%     x_in.rho  (ntm x nx  x ny)   proximal point density    (staggered)
%     x_in.mx   (nt  x nxm x ny)   proximal point x-momentum (staggered)
%     x_in.my   (nt  x nx  x nym)  proximal point y-momentum (staggered)
%     sigma                         proximal stepsize (= 1/gamma in ADMM)
%     problem                       problem struct
%
%   Output:
%     x_out.rho  (ntm x nx  x ny)
%     x_out.mx   (nt  x nxm x ny)
%     x_out.my   (nt  x nx  x nym)

    ops  = problem.ops;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    nt   = problem.nt;
    nx   = problem.nx;

    zeros_x = zeros(nt, problem.ny);   % x-wall BCs  (nt x ny)
    zeros_y = zeros(nt, nx);           % y-wall BCs  (nt x nx)

    % Interpolate proximal point to cell centers (phi-locations)
    rho_c = ops.interp_t_at_phi(x_in.rho, rho0, rho1);
    mx_c  = ops.interp_x_at_phi(x_in.mx, zeros_x, zeros_x);
    my_c  = ops.interp_y_at_phi(x_in.my, zeros_y, zeros_y);

    % Combined squared momentum at cell centers
    m2_c = mx_c.^2 + my_c.^2;

    % Exact prox via cubic formula (same as 1D with |m|^2 = mx^2 + my^2)
    rho_new = solve_cubic(1, 2*sigma - rho_c, sigma^2 - 2*sigma*rho_c, ...
                          -sigma*(sigma*rho_c + 0.5*m2_c));
    scale   = rho_new ./ (rho_new + sigma);
    mx_new  = scale .* mx_c;
    my_new  = scale .* my_c;

    % Enforce non-negativity
    neg_ind = (rho_new <= 1e-12);
    rho_new(neg_ind) = 0.0;
    mx_new(neg_ind)  = 0.0;
    my_new(neg_ind)  = 0.0;

    % Interpolate back to staggered locations
    x_out.rho = ops.interp_t_at_rho(rho_new);
    x_out.mx  = ops.interp_x_at_m(mx_new);
    x_out.my  = ops.interp_y_at_m(my_new);
end
