function cfg = cfg_admm_spectral_imex()
% CFG_ADMM_SPECTRAL_IMEX  Config: spectral in x, IMEX Euler in t, ADMM.
%
%   Discretization:
%     Space  -- spectral (FFT), periodic [0,1]
%     Time   -- IMEX (implicit diffusion, explicit divergence):
%                 rho at interior times t_1,...,t_{nt-1}
%                 mx  at LEFT edges    t_0,...,t_{nt-1}
%
%   IMEX treats the diffusion of rho implicitly (stable for large vareps*dt)
%   and the divergence of mx explicitly.  The constraint matrix L_k is
%   identical to the BE matrix; only the physical interpretation of mx time
%   points differs.  For vareps=0 (pure OT), IMEX reduces to FE.

    cfg.name       = 'admm_spectral_imex';
    cfg.pipeline   = @discretize_then_optimize;
    cfg.disc       = @disc_spectral_1d;
    cfg.prox_ke    = @prox_ke_spectral;
    cfg.projection = @proj_fokker_planck_spectral;
    cfg.time_disc  = 'imex';

    % Grid
    cfg.nt = 128;
    cfg.nx = 128;

    % ADMM parameters
    cfg.gamma    = 100;    % penalty parameter γ
    cfg.alpha    = 1.0;    % over-relaxation
    cfg.vareps   = 0;      % Schrödinger bridge regularisation (0 = pure OT)
    cfg.max_iter = 5000;
    cfg.tol      = 1e-8;

end
