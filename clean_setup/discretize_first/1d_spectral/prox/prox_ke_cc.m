function x_out = prox_ke_cc(x_in, sigma, problem)
% PROX_KE_CC  Exact proximal operator for KE on the collocated grid.
%
%   x_out = prox_ke_cc(x_in, sigma, problem)
%
%   Solves pointwise (rho and mx are collocated so the problem decouples):
%
%     min_{rho,m}  sum_{k,i} m_{k,i}^2 / (2*rho_{k,i}) * dt * dx
%                  + (1/(2*sigma)) * ||(rho,m) - x_in||^2
%
%   Inputs:
%     x_in.rho  (nt x nx)   proximal point density   (collocated)
%     x_in.mx   (nt x nx)   proximal point momentum  (collocated)
%     sigma                  proximal step-size (= 1/gamma in ADMM)
%     problem                struct (used for dt, dx scaling)
%
%   Output:
%     x_out.rho  (nt x nx)
%     x_out.mx   (nt x nx)

    rho_c = x_in.rho;
    mx_c  = x_in.mx;

    % Exact prox via cubic formula (same derivation as prox_ke_exact)
    rho_new = solve_cubic(1, 2*sigma - rho_c, sigma^2 - 2*sigma*rho_c, ...
                          -sigma*(sigma*rho_c + 0.5*mx_c.^2));
    mx_new  = rho_new .* mx_c ./ (rho_new + sigma);

    % Enforce non-negativity: only clip values that are truly negative
    % (numerical error from the cubic solve).  Do NOT use a positive
    % threshold — zeroing out small but positive Gaussian tails creates
    % a spatial step discontinuity that causes Gibbs oscillations when
    % the result is FFT'd in the spectral projection step.
    neg_ind = (rho_new < 0);
    rho_new(neg_ind) = 0.0;
    mx_new(neg_ind)  = 0.0;

    x_out.rho = rho_new;
    x_out.mx  = mx_new;
end
