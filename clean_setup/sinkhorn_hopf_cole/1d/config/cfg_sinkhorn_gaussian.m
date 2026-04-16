function cfg = cfg_sinkhorn_gaussian()
% CFG_SINKHORN_GAUSSIAN  Algorithm config: Sinkhorn / Hopf-Cole iteration.
%
%   Pair with a prob_*.m to define the full experiment.

    cfg.name             = 'sinkhorn_gaussian';
    cfg.disc             = @disc_staggered_1st;      % required by setup_problem

    cfg.precomp_heat     = @precomp_heat_neumann;

    % cfg.precomp_heat     = @precomp_heat_free_space; % swap to @precomp_heat_neumann to compare
    % cfg.use_pdf_marginals = true;   % use raw PDF values as marginals (correct for R reference)
                                    % set false (or omit) when using precomp_heat_neumann
    
    % Neumann 
    % cfg.precomp_heat     = @precomp_heat_neumann; 
    % cf.use_pdf_marginals = false;

    % Grid
    cfg.nt = 64;
    cfg.nx = 128;

    % Sinkhorn parameters
    cfg.vareps   = 1;        % Schrödinger bridge regularisation ε
    cfg.max_iter = 500;
    cfg.tol      = 1e-10;

end
