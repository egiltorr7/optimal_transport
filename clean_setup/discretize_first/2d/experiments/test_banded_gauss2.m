% TEST_BANDED_GAUSS2_2D  Crank-Nicolson banded ADMM vs reference on
%   Gaussian-to-Gaussian SB with DIFFERENT means AND different standard
%   deviations at the two endpoints. Mirrors test_expsemi_gauss2.m but uses
%   the CN banded FP projection (proj_fokker_planck_banded) instead of ETD.
%
%   Saves one .mat per (grid, eps) and one sweep summary per grid.
%   No figures — post-process locally with postprocess_banded_gauss2.m.
%
%   Gaussian marginals:  sigma0=0.03, sigma1=0.05, mu0=(0.3,0.3), mu1=(0.7,0.7)
%     -> 6*max(sigma0,sigma1) = 0.3 to nearest wall  (exp(-18) ~ 1.5e-8)
%     -> machine zero at corners  (exp(-36) ~ 2e-16)
%
%   Reference:
%     eps >= EPS_THRESH  -> log-domain Sinkhorn (Hopf-Cole, Neumann BCs)
%     eps <  EPS_THRESH  -> analytical SB solution (exact for isotropic
%                           Gaussians with GENERAL, possibly different,
%                           variances -- see derivation in gauss2_sb_stag/cc
%                           in test_expsemi_gauss2.m)
%
%   Output files:
%     results/data/banded2d_gauss2_eps<e>_nt<N>_nx<N>.mat  (per run)
%     results/data/banded2d_sweep_gauss2_nt<N>_nx<N>.mat   (per grid)

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));
clear functions

dat_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'data', 'gauss2');
if ~exist(dat_dir, 'dir'), mkdir(dat_dir); end

%% -------------------------------------------------------------------------
%  Gaussian problem parameters (different sigma at each endpoint)
%  (machine zero at corners, negligible at walls)
% -------------------------------------------------------------------------
MU0    = [0.3, 0.3];
MU1    = [0.7, 0.7];
SIGMA0 = 0.03;
SIGMA1 = 0.05;

Normal2d = @(xx, yy, mux, muy, sig) ...
    exp(-((xx - mux).^2 + (yy - muy).^2) / (2*sig^2)) / (2*pi*sig^2);

prob_def           = prob_gaussian();
prob_def.rho0_func = @(xx, yy) Normal2d(xx, yy, MU0(1), MU0(2), SIGMA0);
prob_def.rho1_func = @(xx, yy) Normal2d(xx, yy, MU1(1), MU1(2), SIGMA1);


%% -------------------------------------------------------------------------
%  Grid refinement levels:  [NT, NX]  (NY = NX always, NT = 2*NX lockstep)
% -------------------------------------------------------------------------
GRIDS = [ 64,  32 ; ...
         128,  64 ; ...
         256, 128 ; ...
         512, 256 ];

N_GRIDS = size(GRIDS, 1);

%% -------------------------------------------------------------------------
%  Epsilon sweep and threshold
% -------------------------------------------------------------------------
EPS_SWEEP  = [1e-8, 1e-4, 0.01, 0.1, 1.0, 10.0, 100.0];
EPS_THRESH = 5e-3;   % below: analytical reference; above: Sinkhorn
NE         = numel(EPS_SWEEP);

TOL_WORK   = 1e-8;

%% -------------------------------------------------------------------------
%  Base solver config  (grid fields overwritten per level)
% -------------------------------------------------------------------------
cfg_bd            = cfg_ladmm_gaussian();
cfg_bd.max_iter   = 10000;
cfg_bd.tol        = TOL_WORK;
cfg_bd.gamma      = 0.1;
cfg_bd.tau        = 0.11;
cfg_bd.use_gpu    = true;
cfg_bd.gpu_device = 1;
cfg_bd.print_every = 2000;
cfg_bd.projection  = @proj_fokker_planck_banded;

%% =========================================================================
%  Main loop: grid levels -> epsilon
% =========================================================================
for g = 1:N_GRIDS
    NT = GRIDS(g, 1);
    NX = GRIDS(g, 2);
    NY = NX;

    cfg_bd.nt = NT;
    cfg_bd.nx = NX;
    cfg_bd.ny = NY;

    fprintf('\n========================================\n');
    fprintf('  Grid level %d/%d:  NT=%d  NX=NY=%d\n', g, N_GRIDS, NT, NX);
    fprintf('========================================\n');

    % Sweep-level accumulators
    sw_eps                  = EPS_SWEEP(:);
    sw_wall_bd              = nan(NE, 1);
    sw_wall_ref             = nan(NE, 1);
    sw_iters_bd             = nan(NE, 1);
    sw_converged_bd         = false(NE, 1);
    sw_ref_type             = cell(NE, 1);
    sw_ref_iters            = nan(NE, 1);
    sw_ref_converged        = false(NE, 1);
    sw_max_err_rho_stag     = nan(NE, 1);
    sw_max_err_rho_cc       = nan(NE, 1);
    sw_max_rel_err_rho_stag = nan(NE, 1);
    sw_max_rel_err_rho_cc   = nan(NE, 1);

    for i = 1:NE
        eps_i = EPS_SWEEP(i);
        fprintf('\n  eps = %.4g\n', eps_i);

        cfg_bd.vareps = eps_i;
        problem_i     = setup_problem(cfg_bd, prob_def);

        nt  = problem_i.nt;   ntm = nt - 1;
        nx  = problem_i.nx;
        ny  = problem_i.ny;
        dt  = problem_i.dt;
        dx  = problem_i.dx;
        dy  = problem_i.dy;
        dV  = dx * dy;

        t_stag = (1:ntm)' * dt;
        t_cc   = ((1:nt)' - 0.5) * dt;
        xx     = problem_i.xx;
        yy     = problem_i.yy;

        %% ---- Reference -------------------------------------------------
        use_anal = eps_i < EPS_THRESH;

        if use_anal
            fprintf('    Analytical reference\n');
            t_ref = tic;
            [rho_ref_stag, mx_ref_stag, my_ref_stag] = ...
                gauss2_sb_stag(problem_i, eps_i, MU0, MU1, SIGMA0, SIGMA1);
            rho_ref_cc = gauss2_sb_cc(problem_i, eps_i, MU0, MU1, SIGMA0, SIGMA1);
            ref_wall     = toc(t_ref);
            ref_type     = 'analytical';
            ref_iters    = NaN;
            ref_converged = true;
            rho_sk_full  = [];
        else
            cfg_sk.vareps            = eps_i;
            cfg_sk.max_iter          = 5000;
            cfg_sk.tol               = 1e-10;
            cfg_sk.precomp_heat      = @precomp_heat_neumann_2d;
            cfg_sk.use_pdf_marginals = true;
            if eps_i < 0.05
                cfg_sk.eps_init      = min(1.0, eps_i * 100);
                cfg_sk.anneal_factor = 4.0;
            end

            fprintf('    Sinkhorn  ');
            t_ref        = tic;
            res_sk       = sinkhorn_hopf_cole(problem_i, cfg_sk);
            ref_wall     = toc(t_ref);
            ref_type     = 'sinkhorn';
            ref_iters    = res_sk.iters;
            ref_converged = res_sk.converged;
            fprintf('iters=%d  conv=%d  wall=%.2fs\n', ...
                res_sk.iters, res_sk.converged, ref_wall);

            rho_sk_full   = res_sk.rho;                                   % (nt+1 x nx x ny)
            rho_ref_stag  = res_sk.rho(2:nt, :, :);                      % (ntm x nx x ny)
            rho_ref_cc    = 0.5*(res_sk.rho(1:nt,:,:) + res_sk.rho(2:nt+1,:,:));
            mx_ref_stag   = [];
            my_ref_stag   = [];
        end

        %% ---- Banded (CN) ADMM ------------------------------------------
        fprintf('    Banded    ');
        t0      = tic;
        res_bd  = discretize_then_optimize(cfg_bd, problem_i);
        bd_wall = toc(t0);
        fprintf('iters=%d  conv=%d  wall=%.2fs\n', ...
            res_bd.iters, res_bd.converged, bd_wall);

        %% ---- Extract ADMM fields ---------------------------------------
        rho_bd_stag  = res_bd.rho_stag;
        mx_bd_stag   = res_bd.mx_stag;
        my_bd_stag   = res_bd.my_stag;
        rho_bd_cc    = res_bd.rho_cc;
        mx_bd_cc     = res_bd.mx_cc;
        my_bd_cc     = res_bd.my_cc;
        res_x        = res_bd.res_x;
        res_y        = res_bd.res_y;
        res_primal   = res_bd.res_primal;
        iters_bd     = res_bd.iters;
        converged_bd = res_bd.converged;
        iter_times   = res_bd.iter_times;

        %% ---- Errors vs reference as function of time -------------------
        % Staggered (x-variable, interior integer times t_1..t_{ntm})
        err_rho_stag_t     = sqrt(dV * sum(sum((rho_bd_stag - rho_ref_stag).^2, 2), 3));
        nrm_rho_stag_t     = sqrt(dV * sum(sum(rho_ref_stag.^2, 2), 3));
        rel_err_rho_stag_t = err_rho_stag_t ./ nrm_rho_stag_t;

        % Cell-centre (y-variable, half-integer times)
        err_rho_cc_t     = sqrt(dV * sum(sum((rho_bd_cc - rho_ref_cc).^2, 2), 3));
        nrm_rho_cc_t     = sqrt(dV * sum(sum(rho_ref_cc.^2, 2), 3));
        rel_err_rho_cc_t = err_rho_cc_t ./ nrm_rho_cc_t;

        fprintf('    max rel err:  stag=%.3e  cc=%.3e\n', ...
            max(rel_err_rho_stag_t), max(rel_err_rho_cc_t));

        %% ---- Accumulate sweep ------------------------------------------
        sw_wall_bd(i)              = bd_wall;
        sw_wall_ref(i)             = ref_wall;
        sw_iters_bd(i)             = iters_bd;
        sw_converged_bd(i)         = converged_bd;
        sw_ref_type{i}             = ref_type;
        sw_ref_iters(i)            = ref_iters;
        sw_ref_converged(i)        = ref_converged;
        sw_max_err_rho_stag(i)     = max(err_rho_stag_t);
        sw_max_err_rho_cc(i)       = max(err_rho_cc_t);
        sw_max_rel_err_rho_stag(i) = max(rel_err_rho_stag_t);
        sw_max_rel_err_rho_cc(i)   = max(rel_err_rho_cc_t);

        %% ---- Save per-(grid, eps) file ---------------------------------
        fname = sprintf('banded2d_gauss2_eps%g_nt%d_nx%d.mat', eps_i, NT, NX);
        save(fullfile(dat_dir, fname), ...
            'eps_i', 'NT', 'NX', 'NY', 'nt', 'nx', 'ny', 'dt', 'dx', 'dy', 'dV', ...
            't_stag', 't_cc', 'xx', 'yy', 'MU0', 'MU1', 'SIGMA0', 'SIGMA1', ...
            'ref_type', 'ref_iters', 'ref_converged', 'ref_wall', ...
            'rho_ref_stag', 'rho_ref_cc', 'mx_ref_stag', 'my_ref_stag', ...
            'rho_sk_full', ...
            'bd_wall', 'iters_bd', 'converged_bd', ...
            'rho_bd_stag', 'mx_bd_stag', 'my_bd_stag', ...
            'rho_bd_cc',   'mx_bd_cc',   'my_bd_cc', ...
            'res_x', 'res_y', 'res_primal', 'iter_times', ...
            'err_rho_stag_t',     'nrm_rho_stag_t',     'rel_err_rho_stag_t', ...
            'err_rho_cc_t',       'nrm_rho_cc_t',       'rel_err_rho_cc_t');
        fprintf('    Saved: %s\n', fname);

        clear res_bd res_sk rho_sk_full ...
              rho_ref_stag rho_ref_cc mx_ref_stag my_ref_stag ...
              rho_bd_stag mx_bd_stag my_bd_stag rho_bd_cc mx_bd_cc my_bd_cc ...
              res_x res_y res_primal iter_times ...
              err_rho_stag_t nrm_rho_stag_t rel_err_rho_stag_t ...
              err_rho_cc_t   nrm_rho_cc_t   rel_err_rho_cc_t
    end   % eps loop

    %% ---- Save sweep summary per grid -----------------------------------
    sfname = sprintf('banded2d_sweep_gauss2_nt%d_nx%d.mat', NT, NX);
    save(fullfile(dat_dir, sfname), ...
        'sw_eps', 'NT', 'NX', 'NY', 'TOL_WORK', 'MU0', 'MU1', 'SIGMA0', 'SIGMA1', ...
        'sw_wall_bd', 'sw_wall_ref', ...
        'sw_iters_bd', 'sw_converged_bd', ...
        'sw_ref_type', 'sw_ref_iters', 'sw_ref_converged', ...
        'sw_max_err_rho_stag',     'sw_max_err_rho_cc', ...
        'sw_max_rel_err_rho_stag', 'sw_max_rel_err_rho_cc');
    fprintf('\nSweep summary saved: %s\n', sfname);

end   % grid loop

%% =========================================================================
%  Local functions: analytical SB for isotropic Gaussians with GENERAL
%  (possibly different) variances at the two endpoints. See derivation
%  comment in test_expsemi_gauss2.m -- identical formulas here.
% =========================================================================

function [rho_stag, mx_stag, my_stag] = gauss2_sb_stag(problem, vareps, mu0, mu1, sigma0, sigma1)
% Density and momentum at STAGGERED times (interior integer and half-integer).
    nt  = problem.nt;   ntm = nt - 1;   dt = problem.dt;
    nx  = problem.nx;   nxm = nx - 1;   dx = problem.dx;
    ny  = problem.ny;   nym = ny - 1;   dy = problem.dy;

    C = sqrt(sigma0^2*sigma1^2 + vareps^2) - vareps;

    % rho at interior integer times k*dt, k=1..ntm
    t_r   = reshape((1:ntm)' * dt,          ntm, 1,  1 );
    x_r   = reshape(((1:nx)  - 0.5) * dx,   1,   nx, 1 );
    y_r   = reshape(((1:ny)  - 0.5) * dy,   1,   1,  ny);
    mu_tx = (1-t_r)*mu0(1) + t_r*mu1(1);
    mu_ty = (1-t_r)*mu0(2) + t_r*mu1(2);
    sig2  = 2*vareps*t_r.*(1-t_r) + (1-t_r).^2*sigma0^2 + t_r.^2*sigma1^2 + 2*t_r.*(1-t_r)*C;
    rho_stag = exp(-((x_r-mu_tx).^2 + (y_r-mu_ty).^2) ./ (2*sig2));
    rho_stag = rho_stag ./ (sum(sum(rho_stag,2),3) * dx * dy);

    % mx at half-integer times (k-0.5)*dt, k=1..nt
    t_m   = reshape(((1:nt)' - 0.5) * dt,   nt,  1,  1 );
    x_mx  = reshape((1:nxm) * dx,            1,   nxm,1 );
    y_mx  = reshape(((1:ny)  - 0.5) * dy,    1,   1,  ny);
    mu_tx_m = (1-t_m)*mu0(1) + t_m*mu1(1);
    mu_ty_m = (1-t_m)*mu0(2) + t_m*mu1(2);
    sig2_m  = 2*vareps*t_m.*(1-t_m) + (1-t_m).^2*sigma0^2 + t_m.^2*sigma1^2 + 2*t_m.*(1-t_m)*C;
    sig2p_m = 2*vareps*(1-2*t_m) - 2*(1-t_m)*sigma0^2 + 2*t_m*sigma1^2 + 2*(1-2*t_m)*C;
    k_m     = (sig2p_m - 2*vareps) ./ (2*sig2_m);

    rho_mx  = exp(-((x_mx-mu_tx_m).^2 + (y_mx-mu_ty_m).^2) ./ (2*sig2_m));
    rho_mx  = rho_mx ./ (sum(sum(rho_mx,2),3) * dx * dy);
    vx      = (mu1(1)-mu0(1)) + k_m .* (x_mx-mu_tx_m);
    mx_stag = rho_mx .* vx;

    % my at half-integer times, y-staggered positions
    x_my  = reshape(((1:nx)  - 0.5) * dx,   1,   nx, 1  );
    y_my  = reshape((1:nym) * dy,            1,   1,  nym);
    rho_my  = exp(-((x_my-mu_tx_m).^2 + (y_my-mu_ty_m).^2) ./ (2*sig2_m));
    rho_my  = rho_my ./ (sum(sum(rho_my,2),3) * dx * dy);
    vy      = (mu1(2)-mu0(2)) + k_m .* (y_my-mu_ty_m);
    my_stag = rho_my .* vy;
end

function rho_cc = gauss2_sb_cc(problem, vareps, mu0, mu1, sigma0, sigma1)
% Density at cell-centre (half-integer) times — for y-variable comparison.
    nt = problem.nt;   dt = problem.dt;
    nx = problem.nx;   dx = problem.dx;
    ny = problem.ny;   dy = problem.dy;

    C = sqrt(sigma0^2*sigma1^2 + vareps^2) - vareps;

    t_r   = reshape(((1:nt)' - 0.5) * dt,  nt, 1,  1 );
    x_r   = reshape(((1:nx)  - 0.5) * dx,  1,  nx, 1 );
    y_r   = reshape(((1:ny)  - 0.5) * dy,  1,  1,  ny);
    mu_tx = (1-t_r)*mu0(1) + t_r*mu1(1);
    mu_ty = (1-t_r)*mu0(2) + t_r*mu1(2);
    sig2  = 2*vareps*t_r.*(1-t_r) + (1-t_r).^2*sigma0^2 + t_r.^2*sigma1^2 + 2*t_r.*(1-t_r)*C;
    rho_cc = exp(-((x_r-mu_tx).^2 + (y_r-mu_ty).^2) ./ (2*sig2));
    rho_cc = rho_cc ./ (sum(sum(rho_cc,2),3) * dx * dy);
end
