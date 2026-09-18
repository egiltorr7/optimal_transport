function cfg = cfg_banded_proj()
% CFG_BANDED_PROJ  Reference: exact banded projection (DCT-x + tridiagonal-t).
%
%   Uses proj_fokker_planck_banded, which solves AA*phi = f exactly by
%   DCT in x and one tridiagonal solve per x-mode. Requires precomputation
%   via precomp_banded_proj. Exact for all eps; O(nt * nx) per call.
%
%   Pair with a prob_*.m and run via discretize_then_optimize.
%
%   NOTE: requires problem.banded_proj to be precomputed before ADMM.
%   The pipeline in discretize_then_optimize handles this automatically
%   when cfg.projection = @proj_fokker_planck_banded.

    cfg.name       = 'banded_proj';
    cfg.pipeline   = @discretize_then_optimize;
    cfg.disc       = @disc_staggered_1st;
    cfg.projection = @proj_fokker_planck_banded;
    cfg.prox_ke    = @prox_ke_exact;

    % Grid
    cfg.nt = 128;
    cfg.nx = 128;

    % ADMM parameters
    cfg.gamma    = 100;
    cfg.tau      = 101;     % proximal penalty; convergence needs tau > gamma * ||A||^2 <= gamma
    cfg.alpha    = 1.0;
    cfg.vareps   = 1e-1;
    cfg.max_iter = 10000;
    cfg.tol      = 1e-8;

end
