function obj = compute_objective(rho, mx, problem)
% COMPUTE_OBJECTIVE  Evaluate the kinetic energy functional.
%
%   obj = compute_objective(rho, mx, problem)
%
%   Computes  integral_0^1 integral_0^1  |m|^2 / rho  dt dx
%   using the staggered-grid interpolation to cell centers.

    ops  = problem.ops;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    dt   = problem.dt;
    dx   = problem.dx;
    nt   = problem.nt;

    dirichlet_bc = zeros(nt, 1);

    rho_c = ops.interp_t_at_phi(rho, rho0, rho1);
    mx_c  = ops.interp_x_at_phi(mx, dirichlet_bc, dirichlet_bc);

    ind = rho_c > 1e-8;
    val = zeros(size(rho_c));
    val(ind) = (mx_c(ind).^2) ./ rho_c(ind);

    obj = sum(val(:)) * dt * dx;
end
