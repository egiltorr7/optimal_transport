function outfile = run_pikachu_raichu(vareps, nt, opts)
% RUN_PIKACHU_RAICHU  Solve the Pikachu -> Raichu transport and export to .mat.
%
%   run_pikachu_raichu(0,    64)    % pure OT,          nt = 64
%   run_pikachu_raichu(1e-2, 128)   % Schrodinger bridge, nt = 128
%   run_pikachu_raichu(0, 64, outdir='...', max_iter=5000)
%
%   nx = ny = 128 always (set by the cached marginals). nt is free: the
%   cache holds only the two SPATIAL marginals, so refining time costs no
%   new cache and the nt = 64 / nt = 128 pair is a clean time-refinement
%   study at fixed space.
%
%   ONE vareps PER CALL, deliberately: MATLAB does not release GPU memory
%   between solves in a process, and a fresh process also avoids allocator
%   fragmentation. Use run_pikachu_raichu_all.sh to do both.
%
%   Marginals come from the PRECOMPUTED 128x128 cache
%   (images/mollified/pikachu_raichu_w0.04_128.mat) rather than from the
%   sprites, so the interp2 + bump-convolution setup is not repeated and
%   cannot drift between the two vareps runs.
%
%   NO REFERENCE-ERROR TRACKING. There is no closed form for this problem,
%   so nothing is compared against one -- and nothing needed removing:
%   ladmm_solve never tracked a reference error in the first place. What it
%   records per iteration is res_x, res_y, res_primal and (added for this
%   work) res_dual. The dual residual does cost one extra A^T application
%   per iteration; that is the only diagnostic overhead here.
%
%   Writes <outdir>/pikachu_raichu_eps<E>_<nx>.mat with everything needed to
%   plot in Python -- see the `saved` struct at the bottom for the field list.

    arguments
        vareps          (1,1) double
        nt              (1,1) double = 64
        opts.outdir     (1,:) char   = ''
        opts.max_iter   (1,1) double = -1      % -1 = use the config's value
        opts.cache       (1,:) char   = ''
        opts.tag         (1,:) char   = ''
        opts.print_every (1,1) double = -1     % -1 = use the config's value
    end

    cfg = cfg_ladmm_pikachu(vareps, nt);
    if opts.max_iter   > 0, cfg.max_iter   = opts.max_iter;   end
    if opts.print_every > 0, cfg.print_every = opts.print_every; end

    if isempty(opts.cache)
        opts.cache = fullfile(fileparts(which('prob_mollified')), 'images', ...
                              'mollified', 'pikachu_raichu_w0.04_128.mat');
    end
    if isempty(opts.outdir)
        opts.outdir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'pikachu');
    end
    if ~exist(opts.outdir, 'dir'), mkdir(opts.outdir); end

    prob    = prob_mollified(opts.cache);
    problem = setup_problem(cfg, prob);

    fprintf('\n=== Pikachu -> Raichu | vareps = %g | nt = %d ===\n', vareps, nt);
    fprintf('grid  nt=%d nx=%d ny=%d   gamma=%g tau=%g   max_iter=%d tol=%g\n', ...
            cfg.nt, cfg.nx, cfg.ny, cfg.gamma, cfg.tau, cfg.max_iter, cfg.tol);
    fprintf('proj  %s   marginals: %s\n', func2str(cfg.projection), opts.cache);
    if cfg.use_gpu
        g = gpuDevice(cfg.gpu_device);
        fprintf('gpu   %s, %.1f GB free of %.1f GB\n', g.Name, ...
                g.AvailableMemory/2^30, g.TotalMemory/2^30);
    end

    t0     = tic;
    result = cfg.pipeline(cfg, problem);
    el     = toc(t0);

    fprintf(['done  %d iters in %.1f s (%.1f ms/iter) | converged=%d ' ...
             'res_x(end)=%.3e\n'], result.iters, el, ...
            1000*el/max(result.iters,1), result.converged, result.res_x(end));
    fprintf('      primal %.3e -> %.3e   dual %.3e -> %.3e\n', ...
            result.res_primal(1), result.res_primal(end), ...
            result.res_dual(1),   result.res_dual(end));

    % discretize_then_optimize already gathers x/y off the GPU and returns
    % FLAT fields (rho_cc, rho_stag, ...), not nested x/y structs -- gather()
    % here anyway so this still works if that ever changes.
    g = @(v) gather(v);

    saved = struct();
    % densities and momenta, cell-centre grid (nt x nx x ny) -- what you plot
    saved.rho_cc   = g(result.rho_cc);
    saved.mx_cc    = g(result.mx_cc);
    saved.my_cc    = g(result.my_cc);
    % staggered / FP-feasible density (ntm x nx x ny)
    saved.rho_stag = g(result.rho_stag);
    % marginals actually used
    saved.rho0     = g(problem.rho0);
    saved.rho1     = g(problem.rho1);
    % residual histories (n_iters x 1)
    saved.res_primal = g(result.res_primal);
    saved.res_dual   = g(result.res_dual);
    saved.res_x      = g(result.res_x);
    saved.res_y      = g(result.res_y);
    saved.iter_times = g(result.iter_times);
    % grid
    saved.xx = g(problem.xx);   saved.yy = g(problem.yy);
    saved.dx = problem.dx;      saved.dy = problem.dy;   saved.dt = problem.dt;
    saved.nt = cfg.nt;  saved.nx = cfg.nx;  saved.ny = cfg.ny;
    % run metadata -- record what this was measured with, beside the numbers
    saved.vareps     = vareps;
    saved.gamma      = cfg.gamma;
    saved.tau        = cfg.tau;
    saved.alpha      = cfg.alpha;
    saved.tol        = cfg.tol;
    saved.max_iter   = cfg.max_iter;
    saved.iters      = result.iters;
    saved.converged  = result.converged;
    saved.time_per_iter = result.time_per_iter;
    saved.gpu_mem_post_mb = result.gpu_mem_post_mb;
    saved.walltime   = el;
    saved.projection = func2str(cfg.projection);
    saved.cache_file = opts.cache;
    saved.created    = datestr(now, 'yyyy-mm-dd HH:MM:SS'); %#ok<TNOW1,DATST>

    stem    = sprintf('pikachu_raichu_eps%g_nt%d_nx%d%s', ...
                      vareps, cfg.nt, cfg.nx, opts.tag);
    outfile = fullfile(opts.outdir, [stem '.mat']);
    % -v7.3 = HDF5, which scipy.io cannot read; use -v7 so Python can load it
    % with scipy.io.loadmat. The arrays here are ~4 MB each, well under the
    % 2 GB limit that would force -v7.3.
    save(outfile, '-struct', 'saved', '-v7');
    fprintf('saved %s\n', outfile);
end
