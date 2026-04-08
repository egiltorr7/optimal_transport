function cfg = cfg_fp_dr()
% CFG_FP_DR  Approach 2: Douglas-Rachford projection.
%
%   Splits the FP constraint into:
%     C1 = pure continuity d_t rho + d_x j = 0   (eps=0 DCT solve)
%     C2 = flux definition j = m - eps * d_x rho  (Helmholtz DCT solve in x)
%
%   proj_C1 uses a 2D DCT with eigenvalues lambda_x + lambda_t.
%   proj_C2 uses a 1D DCT in x with eigenvalues 2 + eps^2 * lambda_x.
%   Both are O(nt * nx) per call. The DR outer loop typically needs
%   O(10-30) iterations; cost per ADMM step scales with proj_max_iter.
%
%   Pair with a prob_*.m and run via discretize_then_optimize.

    cfg.name       = 'fp_dr';
    cfg.pipeline   = @discretize_then_optimize;
    cfg.disc       = @disc_staggered_1st;
    cfg.projection = @proj_fp_dr;
    cfg.prox_ke    = @prox_ke_exact;

    % Grid
    cfg.nt = 128;
    cfg.nx = 128;

    % ADMM parameters
    cfg.gamma    = 100;
    cfg.tau      = 101;
    cfg.alpha    = 1.0;
    cfg.vareps   = 1e-1;
    cfg.max_iter = 10000;
    cfg.tol      = 1e-8;

    % DR projection parameters
    cfg.proj_tol      = 1e-6;   % FP residual tolerance for DR inner loop
    cfg.proj_max_iter = 30;     % max DR iterations per projection call

end
