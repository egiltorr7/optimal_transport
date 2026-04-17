function cfg = cfg_sinkhorn_gaussian()
% CFG_SINKHORN_GAUSSIAN  Algorithm config: 2D Sinkhorn / Hopf-Cole iteration.
%
%   Pair with prob_gaussian() to define the full experiment.

    cfg.name         = 'sinkhorn_gaussian';
    cfg.disc         = @disc_staggered_1st;      % required by setup_problem

    % Heat kernel choice (swap to compare):
    cfg.precomp_heat = @precomp_heat_neumann_2d;
    % cfg.precomp_heat     = @precomp_heat_free_space_2d;
    % cfg.use_pdf_marginals = true;   % use raw PDF marginals (for free-space / R^2 reference)

    % Grid
    cfg.nt = 32;
    cfg.nx = 64;
    cfg.ny = 64;

    % Sinkhorn parameters
    cfg.vareps   = 0.05;   % Schrodinger bridge regularisation epsilon
    cfg.max_iter = 500;
    cfg.tol      = 1e-8;

end
