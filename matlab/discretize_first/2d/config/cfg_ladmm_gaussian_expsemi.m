function cfg = cfg_ladmm_gaussian_expsemi()
% CFG_LADMM_GAUSSIAN_EXPSEMI  Like cfg_ladmm_gaussian but replaces the
%   Crank–Nicolson banded projection with the ETD exact-semigroup projection.
%
%   The FP constraint is discretised as
%       (rho_{j+1} - exp(-eps*lxy*dt) * rho_j) / dt  +  phi_{kxky} * div(psi)  =  0
%   where lxy = lambda_x(kx) + lambda_y(ky) and phi = (1-c)/alpha is the
%   ETD weight.  This eliminates the Padé(1,1) approximation error of the
%   Crank–Nicolson scheme, improving accuracy for large eps.

    cfg            = cfg_ladmm_gaussian();
    cfg.name       = 'ladmm_gaussian_expsemi';
    cfg.projection = @proj_fokker_planck_expsemi;
end
