function x_out = prox_ke_cc(x_in, sigma, problem)
% PROX_KE_CC  Exact proximal operator for KE on the cell-centred grid (2D).
%
%   x_out = prox_ke_cc(x_in, sigma, problem)
%
%   Solves:
%     min_{rho,mx,my}  sum_{k,i,j} (mx_{k,i,j}^2+my_{k,i,j}^2)/(2*rho_{k,i,j}) * dt*dx*dy
%                      + (1/(2*sigma)) * ||(rho,mx,my) - x_in||^2
%
%   Because rho, mx, my are all collocated (nt x nx x ny cell-centres),
%   the problem decouples pointwise. The cubic for rho is the same as in 1D
%   with |m|^2 = mx^2 + my^2, and mx_new, my_new scale proportionally.
%
%   Inputs:
%     x_in.rho  (nt x nx x ny)   proximal point density    (cell-centred)
%     x_in.mx   (nt x nx x ny)   proximal point x-momentum (cell-centred)
%     x_in.my   (nt x nx x ny)   proximal point y-momentum (cell-centred)
%     sigma                       proximal step-size (= 1/gamma in ADMM)
%     problem                     problem struct (unused here, kept for interface consistency)
%
%   Output:
%     x_out.rho  (nt x nx x ny)
%     x_out.mx   (nt x nx x ny)
%     x_out.my   (nt x nx x ny)

    rho_c = x_in.rho;
    mx_c  = x_in.mx;
    my_c  = x_in.my;

    m2_c = mx_c.^2 + my_c.^2;

    % Exact prox via cubic formula (same derivation as 1D with |m|^2 = mx^2+my^2)
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

    x_out.rho = rho_new;
    x_out.mx  = mx_new;
    x_out.my  = my_new;
end
