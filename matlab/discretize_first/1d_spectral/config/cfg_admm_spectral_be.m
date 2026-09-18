function cfg = cfg_admm_spectral_be()
% CFG_ADMM_SPECTRAL_BE  Config: spectral in x, backward Euler in t, ADMM.
%
%   Discretization:
%     Space  -- spectral (FFT), periodic [0,1]
%     Time   -- backward Euler (fully implicit):
%                 rho at interior times t_1,...,t_{nt-1}
%                 mx  at RIGHT edges   t_1,...,t_nt
%
%   Backward Euler is unconditionally stable (no CFL restriction) at the
%   cost of a denser per-mode constraint matrix in the projection.

    cfg.name       = 'admm_spectral_be';
    cfg.pipeline   = @discretize_then_optimize;
    cfg.disc       = @disc_spectral_1d;
    cfg.prox_ke    = @prox_ke_spectral;
    cfg.projection = @proj_fokker_planck_spectral;
    cfg.time_disc  = 'be';

    % Grid
    cfg.nt = 256;
    cfg.nx = 256;

    % ADMM parameters
    cfg.gamma    = 1;
    cfg.alpha    = 1.0;
    cfg.vareps   = 0;
    cfg.max_iter = 5000;
    cfg.tol      = 1e-8;

end
