function cfg = cfg_ladmm_gaussian_expsemi_gpu()
% CFG_LADMM_GAUSSIAN_EXPSEMI_GPU  Expsemi-ADMM config with GPU acceleration.
%
%   Uses disc_staggered_1st_gpu (cuBLAS matrix ops),
%        proj_fokker_planck_expsemi_gpu (FFT-based DCT + batched Thomas).
%   Set cfg.use_gpu = true to trigger GPU data transfer in discretize_then_optimize.

    cfg           = cfg_ladmm_gaussian_expsemi();
    cfg.name      = 'ladmm_gaussian_expsemi_gpu';
    cfg.disc      = @disc_staggered_1st_gpu;
    cfg.projection = @proj_fokker_planck_expsemi_gpu;
    cfg.use_gpu   = true;
end
