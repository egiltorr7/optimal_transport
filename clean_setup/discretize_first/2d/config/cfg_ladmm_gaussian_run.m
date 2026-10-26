function cfg = cfg_ladmm_gaussian_run()
    cfg            = cfg_ladmm_gaussian();
    cfg.nt         = 128;
    cfg.nx         = 256;
    cfg.ny         = 256;
    cfg.vareps     = 1;
    cfg.gamma      = 30;
    cfg.tau        = 31.5;
    cfg.projection = @proj_fokker_planck_banded;
    cfg.use_gpu    = true;
    cfg.gpu_device = 1;
end
