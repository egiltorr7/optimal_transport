function obj = compute_objective(rho, mx, my, problem)
% COMPUTE_OBJECTIVE  Evaluate the kinetic energy functional (2D).
%
%   obj = compute_objective(rho, mx, my, problem)
%
%   Computes  integral_0^1 integral_0^1 integral_0^1  (mx^2+my^2)/rho  dt dx dy
%   using staggered-grid interpolation to cell centers.

    ops  = problem.ops;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    dt   = problem.dt;
    dx   = problem.dx;
    dy   = problem.dy;
    nt   = problem.nt;
    nx   = problem.nx;

    zeros_x = zeros(nt, problem.ny);
    zeros_y = zeros(nt, nx);

    rho_c = ops.interp_t_at_phi(rho, rho0, rho1);
    mx_c  = ops.interp_x_at_phi(mx, zeros_x, zeros_x);
    my_c  = ops.interp_y_at_phi(my, zeros_y, zeros_y);

    ind = rho_c > 1e-10;
    val = zeros(size(rho_c));
    val(ind) = (mx_c(ind).^2 + my_c(ind).^2) ./ rho_c(ind);

    obj = sum(val(:)) * dt * dx * dy;
end
