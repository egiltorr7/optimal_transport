function cfg = cfg_ladmm_pikachu(vareps, nt)
% CFG_LADMM_PIKACHU  ETD projection, GPU, for the Pikachu -> Raichu transport.
%
%   cfg = cfg_ladmm_pikachu(vareps, nt)
%
%   Pair with prob_mollified and the 128x128 cached marginals.
%
%   WHY ETD RATHER THAN CRANK-NICOLSON.  Two reasons, and the first is the
%   one that matters at vareps = 0:
%
%   1. ETD is EXACT at vareps = 0. Its weights are c = exp(-vareps*lxy*dt)
%      and phi = (1-c)/alpha with alpha = vareps*lxy*dt. At vareps = 0 the
%      precompute takes the alpha -> 0 branch (phi = 1, c = 1) and the
%      per-mode system collapses to main = 2/dt^2 + lxy, off = -1/dt^2 --
%      the pure continuity-equation projection, with no Pade error at all.
%      (precomp_expsemi_proj.m guards this explicitly: phi is initialised to
%      1 and only overwritten where alpha > 1e-14, so there is no 0/0.)
%
%   2. CN's weakness on this problem is POSITIVITY, not accuracy. The
%      mollified silhouettes are exactly zero over ~55-70% of the domain, so
%      any negative excursion from CN's amplification factor at high lambda
%      lands immediately next to a region of exact zeros, where the KE prox
%      clamps. ETD's semigroup factor exp(-alpha) is positive for every mode
%      and every vareps, so it cannot produce that.
%
%   GPU: uses proj_fokker_planck_expsemi_gpu, the coalesced-layout variant,
%   whenever use_gpu is true. discretize_then_optimize casts the extra _T
%   and twiddle fields to gpuArray for it (see its GPU setup block).

    arguments
        vareps (1,1) double
        nt     (1,1) double = 64
    end

    cfg.name     = sprintf('ladmm_pikachu_eps%g_nt%d', vareps, nt);
    cfg.pipeline = @discretize_then_optimize;
    cfg.disc     = @disc_staggered_1st;
    cfg.prox_ke  = @prox_ke_cc;

    % Grid. nx/ny must match the cached marginals
    % (pikachu_raichu_w0.04_128.mat); nt is free -- the cache stores only the
    % two SPATIAL marginals, so refining time needs no new cache.
    cfg.nt = nt;
    cfg.nx = 128;
    cfg.ny = 128;

    % ADMM. Convergence needs tau > gamma*||A||^2, and ||A||^2 <= 1, so any
    % tau > gamma works; 101/100 sits just above the bound.
    cfg.gamma    = 1;
    cfg.tau      = (101/100) * cfg.gamma;
    cfg.alpha    = 1.0;
    cfg.vareps   = vareps;
    cfg.max_iter = 20000;
    cfg.tol      = 1e-8;

    % GPU
    cfg.use_gpu    = true;
    cfg.gpu_device = 1;
    if cfg.use_gpu
        cfg.projection = @proj_fokker_planck_expsemi_gpu;
    else
        cfg.projection = @proj_fokker_planck_expsemi;
    end

    cfg.print_every = 100;
end
