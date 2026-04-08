function cfg = cfg_staggered_gaussian()
% CFG_STAGGERED_GAUSSIAN  Algorithm config: staggered grid, exact prox, Fokker-Planck projection.
%
%   Pair with a prob_*.m to define the full experiment.

    cfg.name       = 'staggered_gaussian';
    cfg.pipeline   = @optimize_then_discretize;
    cfg.disc       = @disc_staggered_1st;
    % cfg.projection = @proj_fokker_planck;
    cfg.projection = @proj_fokker_planck_banded;
    cfg.prox_ke    = @prox_ke_exact;

    % Grid
    cfg.nt = 256;
    cfg.nx = 128;

    % ADMM parameters
    cfg.gamma    = 100;
    cfg.vareps   = 1e-1;    % Schrodinger bridge regularization (0 = pure OT)
    cfg.max_iter = 20000;
    cfg.tol      = 1e-8;

end
