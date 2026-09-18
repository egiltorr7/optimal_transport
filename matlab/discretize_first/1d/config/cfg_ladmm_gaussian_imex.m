function cfg = cfg_ladmm_gaussian_imex()
% CFG_LADMM_GAUSSIAN_IMEX  Like cfg_ladmm_gaussian but with IMEX (backward-Euler
%   diffusion) projection instead of Crank-Nicolson.
%
%   The FP constraint uses U_t = [I_{ntm}; 0] for the diffusion term,
%   so only the Laplacian of nu appears in the RHS (not mu).
%   tau condition is unchanged: ||A||^2 <= 1 (CN coupling A_fn is unmodified).

    cfg            = cfg_ladmm_gaussian();
    cfg.name       = 'ladmm_gaussian_imex';
    cfg.projection = @proj_fokker_planck_imex;
end
