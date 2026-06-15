function cfg = cfg_ladmm_gaussian_gpu()
% CFG_LADMM_GAUSSIAN_GPU  Standard Crank-Nicolson ADMM config with GPU acceleration.
%
%   Uses disc_staggered_1st_gpu (cuBLAS matrix ops),
%        proj_fokker_planck_banded_gpu (FFT-based DCT + batched Thomas).
%   Set cfg.use_gpu = true to trigger GPU data transfer in discretize_then_optimize.

    cfg            = cfg_ladmm_gaussian();
    cfg.name       = 'ladmm_gaussian_gpu';
    cfg.disc       = @disc_staggered_1st_gpu;
    cfg.projection = @proj_fokker_planck_banded_gpu;
    cfg.use_gpu    = true;
    cfg.gpu_device = 1;    % device index (1-based); override in script if needed
end
