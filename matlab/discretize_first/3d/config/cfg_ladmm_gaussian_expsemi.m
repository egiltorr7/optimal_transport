function cfg = cfg_ladmm_gaussian_expsemi()
% CFG_LADMM_GAUSSIAN_EXPSEMI  L-ADMM config for 3D Gaussian SB with ETD projection.
%
%   Matches the 2D counterpart but adds nz, uses disc_staggered_1st_3d, and
%   points to the 3D GPU ETD projection and KE proximal operator.

    cfg.name       = 'ladmm_gaussian_expsemi_3d';
    cfg.nt         = 32;
    cfg.nx         = 32;
    cfg.ny         = 32;
    cfg.nz         = 32;

    cfg.vareps     = 0.1;
    cfg.gamma      = 100;
    cfg.tau        = 1.05 * cfg.gamma;   % tau > gamma * ||A||^2 ~ gamma

    cfg.disc       = @disc_staggered_1st_3d;
    cfg.projection = @proj_fokker_planck_expsemi_gpu;
    cfg.prox_ke    = @prox_ke_cc;

    cfg.max_iter   = 10000;
    cfg.tol        = 1e-4;
    cfg.print_every = 2000;

    cfg.use_gpu    = false;
    cfg.gpu_device = 1;
end
