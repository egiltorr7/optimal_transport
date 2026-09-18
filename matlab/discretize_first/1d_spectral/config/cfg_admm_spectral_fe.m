function cfg = cfg_admm_spectral_fe()
% CFG_ADMM_SPECTRAL_FE  Config: spectral in x, forward Euler in t, ADMM.
%
%   Pair with a prob_*.m and setup_problem_spectral to define a full experiment.
%
%   Discretization:
%     Space  -- spectral (FFT), periodic [0,1]
%     Time   -- forward Euler (fully explicit):
%                 rho at interior times t_1,...,t_{nt-1}
%                 mx  at LEFT edges    t_0,...,t_{nt-1}
%
%   Solver: standard ADMM.
%     x-step: exact projection onto the discretised FP constraint
%     y-step: exact proximal operator for kinetic energy (prox_ke_spectral)

    cfg.name       = 'admm_spectral_fe';
    cfg.pipeline   = @discretize_then_optimize;
    cfg.disc       = @disc_spectral_1d;
    cfg.prox_ke    = @prox_ke_spectral;
    cfg.projection = @proj_fokker_planck_spectral;
    cfg.time_disc  = 'fe';

    % Grid
    cfg.nt = 128;
    cfg.nx = 128;

    % ADMM parameters
    cfg.gamma    = 100;    % penalty parameter γ
    cfg.alpha    = 1.0;    % over-relaxation (1 = standard, (1,2) = over-relaxed)
    cfg.vareps   = 0;      % Schrödinger bridge regularisation (0 = pure OT)
    cfg.max_iter = 5000;
    cfg.tol      = 1e-8;

end
