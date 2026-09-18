function cfg = cfg_fp_pcg()
% CFG_FP_PCG  Approach 5: PCG projection with separable DCT preconditioner.
%
%   Uses proj_fp_pcg, which solves the normal equations AA*phi = f via
%   preconditioned CG. The preconditioner is the separable DCT approximation
%   (same as proj_fokker_planck), so it is exact for eps=0 and approximate
%   for eps>0. For large eps*nt, more PCG iterations are needed.
%
%   Pair with a prob_*.m and run via discretize_then_optimize.

    cfg.name       = 'fp_pcg';
    cfg.pipeline   = @discretize_then_optimize;
    cfg.disc       = @disc_staggered_1st;
    cfg.projection = @proj_fp_pcg;
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

    % PCG solver parameters
    cfg.proj_tol      = 1e-6;   % relative residual tolerance for PCG inner solve
    cfg.proj_max_iter = 50;     % max PCG iterations per projection call

end
