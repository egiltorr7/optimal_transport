% CFG_LADMM_GAUSSIAN_RUN  Machine-local run overrides (template).
%
%   Copy this file to cfg_ladmm_gaussian_run.m (same directory) and edit.
%   That file is gitignored — it will not be committed.
%
%   test_sb_gaussian.m automatically uses cfg_ladmm_gaussian_run when it
%   exists, falling back to cfg_ladmm_gaussian otherwise.

function cfg = cfg_ladmm_gaussian_run()

    cfg = cfg_ladmm_gaussian();   % start from committed defaults

    % --- Override what you need ---

    % Grid (larger for production runs)
    cfg.nt = 64;
    cfg.nx = 128;
    cfg.ny = 128;

    % ADMM parameters
    % cfg.gamma    = 100;
    % cfg.tau      = 101;
    % cfg.vareps   = 1e-2;
    % cfg.max_iter = 10000;
    % cfg.tol      = 1e-8;

    % GPU
    cfg.use_gpu    = true;
    cfg.gpu_device = 1;    % check available devices with: gpuDeviceTable

end
