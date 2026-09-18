% TEST_EXPSEMI  3D ETD-ADMM for the Gaussian Schrödinger bridge.
%
%   Runs a grid-refinement study over several (NT, NX, NY, NZ) levels and
%   epsilon values.  All results are saved to .mat files for offline
%   post-processing; no figures are generated here.
%
%   Usage (headless server):
%     matlab -nodisplay -batch "run('test_expsemi.m')"
%
%   Output files (in results/data/):
%     expsemi3d_gaussian_eps<e>_nt<N>_nx<N>.mat   per-run data
%     expsemi3d_sweep_gaussian_nt<N>_nx<N>.mat    per-grid sweep summary

clear; close all;
set(groot, 'defaultFigureVisible', 'off');
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

dat_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'data');
if ~exist(dat_dir, 'dir'), mkdir(dat_dir); end

%% -------------------------------------------------------------------------
%  Problem parameters (machine-zero Gaussians: exp(-sigma^2 / 2) < 1e-15)
% -------------------------------------------------------------------------
SIGMA = 0.05;         % std dev; walls at 0/1 -> exp(-(0.3/0.05)^2/2) ~ 1e-16
MU0   = [0.3, 0.3, 0.3];
MU1   = [0.7, 0.7, 0.7];

EPS_SWEEP = [0.01, 0.1, 1.0, 10.0, 100.0];

% Grid levels: [NT, NX, NY, NZ]
GRIDS = [
    16, 16, 16, 16;
    32, 32, 32, 32;
    64, 32, 32, 32;
    64, 64, 64, 64;
];
NGRIDS = size(GRIDS, 1);

prob_def = prob_gaussian_custom(MU0, MU1, SIGMA);

%% -------------------------------------------------------------------------
%  Main loop
% -------------------------------------------------------------------------
for gi = 1:NGRIDS

    NT = GRIDS(gi, 1);
    NX = GRIDS(gi, 2);
    NY = GRIDS(gi, 3);
    NZ = GRIDS(gi, 4);

    fprintf('\n=== Grid: NT=%d  NX=%d  NY=%d  NZ=%d ===\n', NT, NX, NY, NZ);

    cfg_base        = cfg_ladmm_gaussian_expsemi();
    cfg_base.nt     = NT;
    cfg_base.nx     = NX;
    cfg_base.ny     = NY;
    cfg_base.nz     = NZ;
    cfg_base.use_gpu = true;

    NE = numel(EPS_SWEEP);
    sw_max_rel_err_rho_stag = nan(NE, 1);
    sw_iters_es             = nan(NE, 1);
    sw_converged_es         = false(NE, 1);
    sw_walltime_es          = nan(NE, 1);

    for ei = 1:NE
        eps_i = EPS_SWEEP(ei);
        fprintf('\n--- eps = %g ---\n', eps_i);

        cfg_i        = cfg_base;
        cfg_i.vareps = eps_i;

        problem = setup_problem(cfg_i, prob_def);
        nt = problem.nt;   ntm = nt - 1;
        nx = problem.nx;   ny  = problem.ny;   nz  = problem.nz;
        dt = problem.dt;   dx  = problem.dx;   dy  = problem.dy;   dz = problem.dz;
        dV = dx * dy * dz;
        t_stag = (1:ntm)' * dt;
        t_cc   = ((1:nt)' - 0.5) * dt;

        %% Reference solution (analytical is exact for isotropic Gaussians at all eps)
        ref_type = 'analytical';
        [rho_ref_stag, mx_ref_stag, my_ref_stag, mz_ref_stag] = ...
            analytical_sb_gaussian(problem, eps_i, MU0, MU1, SIGMA);
        ref_iters     = 0;
        ref_converged = true;
        ref_wall      = 0;

        %% ETD-ADMM solve
        fprintf('Running ETD-ADMM...\n');
        tic_es = tic;
        res_es = discretize_then_optimize(cfg_i, problem);
        fprintf('  iters=%d  converged=%d  wall=%.2fs\n', ...
            res_es.iters, res_es.converged, res_es.walltime);

        %% Extract fields
        rho_es_stag = res_es.rho_stag;
        mx_es_stag  = res_es.mx_stag;
        my_es_stag  = res_es.my_stag;
        mz_es_stag  = res_es.mz_stag;
        rho_es_cc   = res_es.rho_cc;
        mx_es_cc    = res_es.mx_cc;
        my_es_cc    = res_es.my_cc;
        mz_es_cc    = res_es.mz_cc;
        res_x       = res_es.res_x;
        res_y       = res_es.res_y;
        res_primal  = res_es.res_primal;
        iters_es    = res_es.iters;
        converged_es = res_es.converged;
        iter_times  = res_es.iter_times;

        %% Errors vs time (rho only)
        if strcmp(ref_type, 'analytical')
            err_rho_stag_t = sqrt(dV * sum((rho_es_stag - rho_ref_stag).^2, [2,3,4]));
            nrm_rho_stag_t = sqrt(dV * sum(rho_ref_stag.^2, [2,3,4]));
        else
            err_rho_stag_t = zeros(ntm, 1);
            nrm_rho_stag_t = ones(ntm, 1);
        end
        rel_err_rho_stag_t = err_rho_stag_t ./ max(nrm_rho_stag_t, 1e-14);

        max_rel_err = max(rel_err_rho_stag_t);
        fprintf('  max rel L2 err (rho stag): %.3e\n', max_rel_err);

        sw_max_rel_err_rho_stag(ei) = max_rel_err;
        sw_iters_es(ei)             = iters_es;
        sw_converged_es(ei)         = converged_es;
        sw_walltime_es(ei)          = res_es.walltime;

        %% Save per-run .mat
        fname = sprintf('expsemi3d_gaussian_eps%g_nt%d_nx%d', eps_i, NT, NX);
        fpath = fullfile(dat_dir, [fname '.mat']);
        save(fpath, ...
            'eps_i', 'NT', 'NX', 'NY', 'NZ', 'nt', 'nx', 'ny', 'nz', ...
            'dt', 'dx', 'dy', 'dz', 'dV', 't_stag', 't_cc', 'MU0', 'MU1', 'SIGMA', ...
            'ref_type', 'ref_iters', 'ref_converged', 'ref_wall', ...
            'rho_ref_stag', 'mx_ref_stag', 'my_ref_stag', 'mz_ref_stag', ...
            'rho_es_stag', 'mx_es_stag', 'my_es_stag', 'mz_es_stag', ...
            'rho_es_cc', 'mx_es_cc', 'my_es_cc', 'mz_es_cc', ...
            'res_x', 'res_y', 'res_primal', 'iter_times', ...
            'iters_es', 'converged_es', ...
            'err_rho_stag_t', 'nrm_rho_stag_t', 'rel_err_rho_stag_t');
        fprintf('Saved: %s\n', fpath);

        clear res_es rho_es_stag mx_es_stag my_es_stag mz_es_stag
        clear rho_es_cc mx_es_cc my_es_cc mz_es_cc
        clear rho_ref_stag mx_ref_stag my_ref_stag mz_ref_stag
    end

    %% Save per-grid sweep summary
    EPS_VALS = EPS_SWEEP;
    fname_sw = sprintf('expsemi3d_sweep_gaussian_nt%d_nx%d', NT, NX);
    save(fullfile(dat_dir, [fname_sw '.mat']), ...
        'NT', 'NX', 'NY', 'NZ', 'EPS_VALS', ...
        'sw_max_rel_err_rho_stag', 'sw_iters_es', 'sw_converged_es', 'sw_walltime_es');
    fprintf('Saved sweep summary: %s\n', fname_sw);

end

fprintf('\nAll done.\n');

%% =========================================================================
%  Helper: build prob_def with custom mu / sigma (overrides prob_gaussian)
% =========================================================================

function prob = prob_gaussian_custom(mu0, mu1, sigma)
    Normal3d = @(xx, yy, zz, mux, muy, muz, s) ...
        exp(-((xx - mux).^2 + (yy - muy).^2 + (zz - muz).^2) / (2*s^2)) ...
        / (2*pi*s^2)^(3/2);
    prob.name      = 'gaussian';
    prob.rho0_func = @(xx, yy, zz) Normal3d(xx, yy, zz, mu0(1), mu0(2), mu0(3), sigma);
    prob.rho1_func = @(xx, yy, zz) Normal3d(xx, yy, zz, mu1(1), mu1(2), mu1(3), sigma);
end
