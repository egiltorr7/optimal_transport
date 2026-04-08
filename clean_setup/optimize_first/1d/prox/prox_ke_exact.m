function x_out = prox_ke_exact(x_in, sigma, problem)
% PROX_KE_EXACT  Exact closed-form proximal operator for the kinetic energy.
%
%   x_out = prox_ke_exact(x_in, sigma, problem)
%
%   Solves:
%     min_{rho,m}  integral |m|^2/(2*rho) dt dx  +  (1/(2*sigma))*||(rho,m) - x_in||^2
%
%   via the exact cubic-root formula. Interpolates to cell centers (phi-locations),
%   applies the prox, then interpolates back to staggered locations.
%
%   Inputs:
%     x_in.rho  (ntm x nx)  proximal point density   (staggered)
%     x_in.mx   (nt x nxm)  proximal point momentum  (staggered)
%     sigma                  proximal stepsize (= 1/gamma in ADMM)
%     problem                problem struct
%
%   Output:
%     x_out.rho  (ntm x nx)
%     x_out.mx   (nt  x nxm)

    ops  = problem.ops;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    nt   = problem.nt;

    zeros_x = zeros(nt, 1);

    % Interpolate proximal point to cell centers (phi-locations)
    rho_c = ops.interp_t_at_phi(x_in.rho, rho0, rho1);
    mx_c  = ops.interp_x_at_phi(x_in.mx, zeros_x, zeros_x);

    % Exact prox via cubic formula
    rho_new = solve_cubic(1, 2*sigma - rho_c, sigma^2 - 2*sigma*rho_c, ...
                          -sigma*(sigma*rho_c + 0.5*mx_c.^2));
    mx_new  = rho_new .* mx_c ./ (rho_new + sigma);

    % Enforce non-negativity
    neg_ind = (rho_new <= 1e-12);
    rho_new(neg_ind) = 0.0;
    mx_new(neg_ind)  = 0.0;

    % Interpolate back to staggered locations
    x_out.rho = ops.interp_t_at_rho(rho_new);
    x_out.mx  = ops.interp_x_at_m(mx_new);
end
