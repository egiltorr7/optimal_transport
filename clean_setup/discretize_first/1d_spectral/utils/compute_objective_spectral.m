function obj = compute_objective_spectral(rho, mx, problem)
% COMPUTE_OBJECTIVE_SPECTRAL  Kinetic energy on the collocated (nt x nx) grid.
%
%   obj = compute_objective_spectral(rho, mx, problem)
%
%   Computes  integral_0^1 integral_0^1  |m|^2 / rho  dt dx
%   using collocated variables (no staggered interpolation).
%
%   Drop-in replacement for compute_objective when using disc_spectral_1d.

    dt  = problem.dt;
    dx  = problem.dx;

    ind = rho > 1e-8;
    val = zeros(size(rho));
    val(ind) = mx(ind).^2 ./ rho(ind);

    obj = sum(val(:)) * dt * dx;
end
