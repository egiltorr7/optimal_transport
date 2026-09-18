function cfg = cfg_spectral_proj()
% CFG_SPECTRAL_PROJ  Approach 1: spectral (DCT) approximate projection.
%
%   Uses proj_fokker_planck, which inverts AA* via the separable eigenvalue
%   approximation  lambda = lambda_x + lambda_t + eps^2 * lambda_x^2.
%   Exact for eps=0; approximate (but very fast) for eps>0.
%
%   Pair with a prob_*.m and run via discretize_then_optimize.

    cfg.name       = 'spectral_proj';
    cfg.pipeline   = @discretize_then_optimize;
    cfg.disc       = @disc_staggered_1st;
    cfg.projection = @proj_fokker_planck;
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
