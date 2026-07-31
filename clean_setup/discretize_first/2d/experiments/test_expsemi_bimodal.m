% TEST_EXPSEMI_BIMODAL_2D  ETD GPU ADMM vs Sinkhorn reference on unimodal-to-
%   bimodal SB (mass-splitting transport), grid refinement.
%
%   Saves one .mat per (grid, eps) and one sweep summary per grid.
%   No figures — post-process locally with postprocess_expsemi_bimodal.m.
%
%   Marginals:
%     rho0: single Gaussian bump at (0.5, 0.3), sigma0=0.05
%     rho1: two equal-weight Gaussian bumps at (0.3,0.7) and (0.7,0.7),
%           sigma1=0.05 each -- splits into two symmetric peaks
%     -> 6*sigma = 0.3 to nearest wall for every bump  (exp(-18) ~ 1.5e-8)
%     -> bumps separated by 0.4 = 8*sigma1, well-resolved as distinct peaks
%
%   Reference: log-domain Sinkhorn (Hopf-Cole, Neumann BCs) for ALL eps --
%   no closed-form SB solution exists for mixture marginals, unlike the
%   Gaussian-to-Gaussian test cases.
%
%   Output files:
%     results/data/bimodal/expsemi2d_bimodal_eps<e>_nt<N>_nx<N>.mat  (per run)
%     results/data/bimodal/expsemi2d_sweep_bimodal_nt<N>_nx<N>.mat   (per grid)

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));
clear functions

dat_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'data', 'bimodal');
if ~exist(dat_dir, 'dir'), mkdir(dat_dir); end

%% -------------------------------------------------------------------------
%  Unimodal-to-bimodal problem parameters
%  (machine zero at corners, negligible at walls)
% -------------------------------------------------------------------------
MU0    = [0.5, 0.3];
MU1A   = [0.3, 0.7];
MU1B   = [0.7, 0.7];
SIGMA0 = 0.05;
SIGMA1 = 0.05;

Normal2d = @(xx, yy, mux, muy, sig) ...
    exp(-((xx - mux).^2 + (yy - muy).^2) / (2*sig^2)) / (2*pi*sig^2);

prob_def           = prob_gaussian();
prob_def.rho0_func = @(xx, yy) Normal2d(xx, yy, MU0(1), MU0(2), SIGMA0);
prob_def.rho1_func = @(xx, yy) 0.5*Normal2d(xx, yy, MU1A(1), MU1A(2), SIGMA1) ...
                              + 0.5*Normal2d(xx, yy, MU1B(1), MU1B(2), SIGMA1);


%% -------------------------------------------------------------------------
%  Grid refinement levels:  [NT, NX]  (NY = NX always, NT = 2*NX lockstep)
% -------------------------------------------------------------------------
GRIDS = [ 64,  32 ; ...
         128,  64 ; ...
         256, 128 ; ...
         512, 256 ];

N_GRIDS = size(GRIDS, 1);

%% -------------------------------------------------------------------------
%  Epsilon sweep (no threshold -- Sinkhorn is the reference for every eps)
% -------------------------------------------------------------------------
EPS_SWEEP  = [1e-8, 1e-4, 0.01, 0.1, 1.0, 10.0, 100.0];
NE         = numel(EPS_SWEEP);

TOL_WORK   = 1e-8;

%% -------------------------------------------------------------------------
%  Base solver config  (grid fields overwritten per level)
% -------------------------------------------------------------------------
cfg_es            = cfg_ladmm_gaussian_expsemi();
cfg_es.max_iter   = 10000;
cfg_es.gamma      = 0.1;
cfg_es.tau        = 0.11;
cfg_es.tol        = TOL_WORK;
cfg_es.use_gpu    = true;
cfg_es.gpu_device = 1;
cfg_es.print_every = 2000;
cfg_es.projection  = @proj_fokker_planck_expsemi;

%% =========================================================================
%  Main loop: grid levels -> epsilon
% =========================================================================
for g = 1:N_GRIDS
    NT = GRIDS(g, 1);
    NX = GRIDS(g, 2);
    NY = NX;

    cfg_es.nt = NT;
    cfg_es.nx = NX;
    cfg_es.ny = NY;

    fprintf('\n========================================\n');
    fprintf('  Grid level %d/%d:  NT=%d  NX=NY=%d\n', g, N_GRIDS, NT, NX);
    fprintf('========================================\n');

    % Sweep-level accumulators
    sw_eps                  = EPS_SWEEP(:);
    sw_wall_es              = nan(NE, 1);
    sw_wall_ref             = nan(NE, 1);
    sw_iters_es             = nan(NE, 1);
    sw_converged_es         = false(NE, 1);
    sw_ref_iters            = nan(NE, 1);
    sw_ref_converged        = false(NE, 1);
    sw_max_err_rho_stag     = nan(NE, 1);
    sw_max_err_rho_cc       = nan(NE, 1);
    sw_max_rel_err_rho_stag = nan(NE, 1);
    sw_max_rel_err_rho_cc   = nan(NE, 1);

    for i = 1:NE
        eps_i = EPS_SWEEP(i);
        fprintf('\n  eps = %.4g\n', eps_i);

        cfg_es.vareps = eps_i;
        problem_i     = setup_problem(cfg_es, prob_def);

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

        %% ---- Reference: Sinkhorn for every eps (no closed form) --------
        cfg_sk.vareps            = eps_i;
        cfg_sk.max_iter          = 10000;
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

        % Sinkhorn can fail to converge (or diverge to NaN/Inf) for some eps,
        % especially very small eps where entropic regularization becomes
        % numerically singular. Don't save the Sinkhorn density/error data in
        % that case -- a broken reference would silently corrupt the error
        % metrics. Still run and save the ADMM solver's own output below.
        sinkhorn_ok = res_sk.converged && ~any(isnan(res_sk.rho(:))) && ~any(isinf(res_sk.rho(:)));
        if sinkhorn_ok
            rho_sk_full   = res_sk.rho;                                   % (nt+1 x nx x ny)
            rho_ref_stag  = res_sk.rho(2:nt, :, :);                      % (ntm x nx x ny)
            rho_ref_cc    = 0.5*(res_sk.rho(1:nt,:,:) + res_sk.rho(2:nt+1,:,:));
        else
            fprintf('    Sinkhorn did not converge/diverged -- no reference density saved for this eps\n');
            rho_sk_full  = [];
            rho_ref_stag = [];
            rho_ref_cc   = [];
        end

        %% ---- ExpSemi ADMM ----------------------------------------------
        fprintf('    ExpSemi   ');
        t0      = tic;
        res_es  = discretize_then_optimize(cfg_es, problem_i);
        es_wall = toc(t0);
        fprintf('iters=%d  conv=%d  wall=%.2fs\n', ...
            res_es.iters, res_es.converged, es_wall);

        %% ---- Extract ADMM fields ---------------------------------------
        rho_es_stag  = res_es.rho_stag;
        mx_es_stag   = res_es.mx_stag;
        my_es_stag   = res_es.my_stag;
        rho_es_cc    = res_es.rho_cc;
        mx_es_cc     = res_es.mx_cc;
        my_es_cc     = res_es.my_cc;
        res_x        = res_es.res_x;
        res_y        = res_es.res_y;
        res_primal   = res_es.res_primal;
        iters_es     = res_es.iters;
        converged_es = res_es.converged;
        iter_times   = res_es.iter_times;

        %% ---- Errors vs reference as function of time -------------------
        % Only computable when Sinkhorn produced a valid reference; without
        % one we keep the ADMM solve but cannot compare it to anything.
        if sinkhorn_ok
            % Staggered (x-variable, interior integer times t_1..t_{ntm})
            err_rho_stag_t     = sqrt(dV * sum(sum((rho_es_stag - rho_ref_stag).^2, 2), 3));
            nrm_rho_stag_t     = sqrt(dV * sum(sum(rho_ref_stag.^2, 2), 3));
            rel_err_rho_stag_t = err_rho_stag_t ./ nrm_rho_stag_t;

            % Cell-centre (y-variable, half-integer times)
            err_rho_cc_t     = sqrt(dV * sum(sum((rho_es_cc - rho_ref_cc).^2, 2), 3));
            nrm_rho_cc_t     = sqrt(dV * sum(sum(rho_ref_cc.^2, 2), 3));
            rel_err_rho_cc_t = err_rho_cc_t ./ nrm_rho_cc_t;

            fprintf('    max rel err:  stag=%.3e  cc=%.3e\n', ...
                max(rel_err_rho_stag_t), max(rel_err_rho_cc_t));
        else
            err_rho_stag_t = [];  nrm_rho_stag_t = [];  rel_err_rho_stag_t = [];
            err_rho_cc_t   = [];  nrm_rho_cc_t   = [];  rel_err_rho_cc_t   = [];
            fprintf('    max rel err:  N/A (no valid Sinkhorn reference)\n');
        end

        %% ---- Accumulate sweep ------------------------------------------
        sw_wall_es(i)              = es_wall;
        sw_wall_ref(i)             = ref_wall;
        sw_iters_es(i)             = iters_es;
        sw_converged_es(i)         = converged_es;
        sw_ref_iters(i)            = ref_iters;
        sw_ref_converged(i)        = ref_converged;
        if sinkhorn_ok
            sw_max_err_rho_stag(i)     = max(err_rho_stag_t);
            sw_max_err_rho_cc(i)       = max(err_rho_cc_t);
            sw_max_rel_err_rho_stag(i) = max(rel_err_rho_stag_t);
            sw_max_rel_err_rho_cc(i)   = max(rel_err_rho_cc_t);
        end

        %% ---- Save per-(grid, eps) file ---------------------------------
        fname = sprintf('expsemi2d_bimodal_eps%g_nt%d_nx%d.mat', eps_i, NT, NX);
        save(fullfile(dat_dir, fname), ...
            'eps_i', 'NT', 'NX', 'NY', 'nt', 'nx', 'ny', 'dt', 'dx', 'dy', 'dV', ...
            't_stag', 't_cc', 'xx', 'yy', 'MU0', 'MU1A', 'MU1B', 'SIGMA0', 'SIGMA1', ...
            'ref_type', 'ref_iters', 'ref_converged', 'ref_wall', ...
            'rho_ref_stag', 'rho_ref_cc', 'rho_sk_full', ...
            'es_wall', 'iters_es', 'converged_es', ...
            'rho_es_stag', 'mx_es_stag', 'my_es_stag', ...
            'rho_es_cc',   'mx_es_cc',   'my_es_cc', ...
            'res_x', 'res_y', 'res_primal', 'iter_times', ...
            'err_rho_stag_t',     'nrm_rho_stag_t',     'rel_err_rho_stag_t', ...
            'err_rho_cc_t',       'nrm_rho_cc_t',       'rel_err_rho_cc_t');
        fprintf('    Saved: %s\n', fname);

        clear res_es res_sk rho_sk_full ...
              rho_ref_stag rho_ref_cc ...
              rho_es_stag mx_es_stag my_es_stag rho_es_cc mx_es_cc my_es_cc ...
              res_x res_y res_primal iter_times ...
              err_rho_stag_t nrm_rho_stag_t rel_err_rho_stag_t ...
              err_rho_cc_t   nrm_rho_cc_t   rel_err_rho_cc_t
    end   % eps loop

    %% ---- Save sweep summary per grid -----------------------------------
    sfname = sprintf('expsemi2d_sweep_bimodal_nt%d_nx%d.mat', NT, NX);
    save(fullfile(dat_dir, sfname), ...
        'sw_eps', 'NT', 'NX', 'NY', 'TOL_WORK', 'MU0', 'MU1A', 'MU1B', 'SIGMA0', 'SIGMA1', ...
        'sw_wall_es', 'sw_wall_ref', ...
        'sw_iters_es', 'sw_converged_es', ...
        'sw_ref_iters', 'sw_ref_converged', ...
        'sw_max_err_rho_stag',     'sw_max_err_rho_cc', ...
        'sw_max_rel_err_rho_stag', 'sw_max_rel_err_rho_cc');
    fprintf('\nSweep summary saved: %s\n', sfname);

end   % grid loop
