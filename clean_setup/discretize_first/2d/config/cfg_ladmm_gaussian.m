function cfg = cfg_ladmm_gaussian()
% CFG_LADMM_GAUSSIAN  Algorithm config: staggered grid, linearized ADMM (2D).
%
%   Pair with a prob_*.m to define the full experiment.

    cfg.name        = 'ladmm_gaussian';
    cfg.pipeline    = @discretize_then_optimize;
    cfg.disc        = @disc_staggered_1st;
    cfg.prox_ke     = @prox_ke_cc;
    cfg.projection  = @proj_fokker_planck_banded;

    % Grid
    cfg.nt = 64;
    cfg.nx = 128;
    cfg.ny = 128;

    % ADMM parameters
    cfg.gamma    = 100;       % penalty parameter γ
    cfg.tau      = 101;       % proximal penalty τ; convergence needs τ > γ*||A||² ≤ γ
    cfg.alpha    = 1.0;       % over-relaxation (1 = standard, (1,2) = over-relaxed)
    cfg.vareps   = 1e-2;      % Schrödinger bridge regularisation (0 = pure OT)
    cfg.max_iter = 10000;
    cfg.tol      = 1e-8;

end
