function x_out = prox_ke_cc(x_in, sigma, problem)
% PROX_KE_CC  Exact proximal operator for KE on the cell-centred grid (3D).
%
%   x_out = prox_ke_cc(x_in, sigma, problem)
%
%   Solves pointwise:
%     min_{rho,mx,my,mz}  (mx^2+my^2+mz^2)/(2*rho)
%                         + (1/(2*sigma))*||(rho,mx,my,mz) - x_in||^2
%
%   All variables are collocated on the (nt x nx x ny x nz) cell-centre grid,
%   so the problem decouples pointwise and is solved via the exact cubic formula.
%
%   Inputs:
%     x_in.rho  (nt x nx x ny x nz)   cell-centred density
%     x_in.mx   (nt x nx x ny x nz)   cell-centred x-momentum
%     x_in.my   (nt x nx x ny x nz)   cell-centred y-momentum
%     x_in.mz   (nt x nx x ny x nz)   cell-centred z-momentum
%     sigma                            proximal step-size (= 1/gamma)
%     problem                          problem struct (unused, kept for interface)
%
%   Output: x_out.rho, x_out.mx, x_out.my, x_out.mz (same grid as input)

    rho_c = x_in.rho;
    mx_c  = x_in.mx;
    my_c  = x_in.my;
    mz_c  = x_in.mz;

    m2_c = mx_c.^2 + my_c.^2 + mz_c.^2;

    rho_new = solve_cubic(1, 2*sigma - rho_c, sigma^2 - 2*sigma*rho_c, ...
                          -sigma*(sigma*rho_c + 0.5*m2_c));
    scale   = rho_new ./ (rho_new + sigma);
    mx_new  = scale .* mx_c;
    my_new  = scale .* my_c;
    mz_new  = scale .* mz_c;

    neg_ind = (rho_new <= 1e-12);
    rho_new(neg_ind) = 0;
    mx_new(neg_ind)  = 0;
    my_new(neg_ind)  = 0;
    mz_new(neg_ind)  = 0;

    x_out.rho = rho_new;
    x_out.mx  = mx_new;
    x_out.my  = my_new;
    x_out.mz  = mz_new;
end
