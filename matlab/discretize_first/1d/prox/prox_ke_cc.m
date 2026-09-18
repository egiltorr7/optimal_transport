function x_out = prox_ke_cc(x_in, sigma, problem)
% PROX_KE_CC  Exact proximal operator for KE on the cell-centred grid.
%
%   x_out = prox_ke_cc(x_in, sigma, problem)
%
%   Solves:
%     min_{rho,m}  sum_{k,i} m_{k,i}^2 / (2*rho_{k,i}) * dt * dx
%                  + (1/(2*sigma)) * ||(rho,m) - x_in||^2
%
%   Because rho and m are collocated (both at (nt x nx) cell-centres),
%   the problem decouples pointwise and is solved via the exact cubic formula,
%   with no interpolation required.
%
%   Inputs:
%     x_in.rho  (nt x nx)   proximal point density   (cell-centred)
%     x_in.mx   (nt x nx)   proximal point momentum  (cell-centred)
%     sigma                  proximal step-size (= 1/gamma in ADMM)
%     problem                problem struct (used for dt, dx scaling)
%
%   Output:
%     x_out.rho  (nt x nx)
%     x_out.mx   (nt x nx)

    rho_c = x_in.rho;
    mx_c  = x_in.mx;

    % Exact prox via cubic formula (same derivation as prox_ke_exact,
    % but applied directly since variables are already collocated)
    rho_new = solve_cubic(1, 2*sigma - rho_c, sigma^2 - 2*sigma*rho_c, ...
                          -sigma*(sigma*rho_c + 0.5*mx_c.^2));
    mx_new  = rho_new .* mx_c ./ (rho_new + sigma);

    % Enforce non-negativity
    neg_ind = (rho_new <= 1e-12);
    rho_new(neg_ind) = 0.0;
    mx_new(neg_ind)  = 0.0;

    x_out.rho = rho_new;
    x_out.mx  = mx_new;
end
