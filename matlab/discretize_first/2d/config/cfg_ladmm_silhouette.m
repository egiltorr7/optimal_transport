function cfg = cfg_ladmm_silhouette()
% CFG_LADMM_SILHOUETTE  Algorithm config for the mollified-silhouette transport.
%
%   Pair with prob_silhouette. Differs from cfg_ladmm_gaussian in two ways
%   that matter, both driven by vareps:
%
%   vareps = 1e-3, not 1.  vareps is the Schrodinger-bridge regularisation
%   (0 = pure OT). At vareps = 1 the shape diffuses into a featureless blob
%   by mid-time and reassembles, so almost none of the transport is visible
%   -- fine for a convergence study on a Gaussian, useless for a morph.
%   1e-3 is near-OT: the silhouette stays sharp and you see it move.
%
%   projection = banded (Thomas), not spike2.  The MATLAB notes put the
%   Thomas instability at vareps/dt > 1; here vareps/dt = 1e-3 * 32 = 0.032,
%   comfortably stable, so the cheaper solver is the right one. That
%   inequality is why cfg_ladmm_gaussian (vareps = 1, dt = 1/32, so
%   vareps/dt = 32) carries the spike2 alternative.

    cfg.name       = 'ladmm_silhouette';
    cfg.pipeline   = @discretize_then_optimize;
    cfg.disc       = @disc_staggered_1st;
    cfg.prox_ke    = @prox_ke_cc;
    cfg.projection = @proj_fokker_planck_banded;

    % Grid
    cfg.nt = 32;
    cfg.nx = 128;
    cfg.ny = 128;

    % ADMM parameters
    cfg.gamma    = 1;
    cfg.tau      = 1.05 * cfg.gamma;   % needs tau > gamma*||A||^2 <= gamma
    cfg.alpha    = 1.0;
    cfg.vareps   = 1e-3;
    cfg.max_iter = 10000;
    cfg.tol      = 1e-8;

    % No CUDA GPU on this Mac (gpuDeviceCount == 0), so CPU.
    cfg.use_gpu    = false;
    cfg.gpu_device = 1;
end
