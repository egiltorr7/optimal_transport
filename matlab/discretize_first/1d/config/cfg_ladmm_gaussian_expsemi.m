function cfg = cfg_ladmm_gaussian_expsemi()
% CFG_LADMM_GAUSSIAN_EXPSEMI  Like cfg_ladmm_gaussian but uses the exact heat
%   semigroup exp(eps*Delta*dt) in the FP projection instead of Crank-Nicolson.
%
%   The FP constraint is discretised as:
%       (rho_k - S_eps^dt rho_{k-1}) / dt  +  D_x psi  =  0
%   where S_eps^dt is applied exactly per DCT mode via exp(-eps*lambda_x*dt).
%
%   This eliminates the Padé(1,1) approximation error and gives O(dt) accuracy
%   in the diffusion step that is independent of eps.

    cfg            = cfg_ladmm_gaussian();
    cfg.name       = 'ladmm_gaussian_expsemi';
    cfg.projection = @proj_fokker_planck_expsemi;
end
