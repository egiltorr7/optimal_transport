function x_out = prox_ke_cc_nonuniform(x_in, sigma, problem)
% PROX_KE_CC_NONUNIFORM  KE proximal operator for a non-uniform time grid.
%
%   Drop-in replacement for prox_ke_cc when using a non-uniform time grid.
%   The KE functional integrates over physical time:
%
%     G = sum_n dt_vec(n) * sum_x dx * m_n(x)^2 / (2 * rho_n(x))
%
%   so cell n's prox uses an effective step  sigma_eff(n) = sigma * dt_vec(n)/dt
%   instead of the uniform sigma.  Broadcasts (nt x 1) sigma_eff against
%   the (nt x nx) density and momentum matrices.
%
%   Inputs:
%     x_in.rho  (nt x nx)   cell-centred density
%     x_in.mx   (nt x nx)   cell-centred momentum
%     sigma                  proximal step-size (= 1/gamma in ADMM)
%     problem                problem struct; uses problem.dt and problem.dt_vec

    rho_c = x_in.rho;
    mx_c  = x_in.mx;

    dt     = problem.dt;       % scalar 1/nt (uniform reference step)
    dt_vec = problem.dt_vec;   % (nt x 1) actual non-uniform steps

    sigma_eff = sigma * dt_vec / dt;   % (nt x 1), broadcasts over x

    rho_new = solve_cubic(1, ...
                          2*sigma_eff - rho_c, ...
                          sigma_eff.^2 - 2*sigma_eff.*rho_c, ...
                          -sigma_eff.*(sigma_eff.*rho_c + 0.5*mx_c.^2));
    mx_new  = rho_new .* mx_c ./ (rho_new + sigma_eff);

    neg_ind = (rho_new <= 1e-12);
    rho_new(neg_ind) = 0.0;
    mx_new(neg_ind)  = 0.0;

    x_out.rho = rho_new;
    x_out.mx  = mx_new;
end
