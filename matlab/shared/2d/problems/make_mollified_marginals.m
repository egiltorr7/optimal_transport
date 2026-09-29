function make_mollified_marginals(opts)
% MAKE_MOLLIFIED_MARGINALS  Build and cache the mollified silhouette marginals.
%
%   make_mollified_marginals()                 % defaults: w=0.04, 128 and 256
%   make_mollified_marginals(widths=[0.04], resolutions=[128 256])
%
%   Writes, per (w, n), into images/mollified/:
%     <name>_w<W>_<n>.mat   rho0, rho1 and the grid  -- load with prob_mollified
%     rho0_<a>_w<W>_<n>.png } exactly n x n greyscale, for looking at
%     rho1_<b>_w<W>_<n>.png }
%
%   WHY w = 0.04 IS THE DEFAULT.  The controlling quantity is cells per
%   mollification radius, w*n, not w or n alone. Measured on these sprites:
%
%     cells  peak cell-to-cell jump   DCT energy above 3n/4
%      2.56          38.2%                  3.0e-05     under-resolved
%      5.12          18.6%                  2.6e-07     ok
%     10.24           9.3%                  1.3e-09     ok
%
%   w = 0.04 gives 5.1 cells at n=128 and 10.2 at n=256, so BOTH grids
%   resolve the SAME continuum density -- which is what makes a refinement
%   study between them mean anything. w = 0.02 is only safe at n = 256.
%
%   This matters because of the FP projection, not the DCT: the DCT in
%   proj_fokker_planck_banded is an exact diagonalization, not a truncation,
%   so it does not ring. The hazard is Crank-Nicolson's positivity -- at
%   high lambda its amplification factor turns negative, a steep edge
%   excites exactly those modes, and with floor_frac = 0 any negative
%   excursion lands next to a region of exact zeros where the KE prox
%   clamps. Widening the mollifier is the cheap way to stay out of that.

    arguments
        opts.widths      (1,:) double = 0.04
        opts.resolutions (1,:) double = [128 256]
        opts.margin      (1,1) double = 0.08
        opts.img0        (1,:) char   = ''
        opts.img1        (1,:) char   = ''
        opts.label0      (1,:) char   = 'pikachu'
        opts.label1      (1,:) char   = 'raichu'
        opts.name        (1,:) char   = 'pikachu_raichu'
    end

    here  = fileparts(mfilename('fullpath'));
    imdir = fullfile(here, 'images');
    if isempty(opts.img0), opts.img0 = fullfile(imdir, [opts.label0 '.png']); end
    if isempty(opts.img1), opts.img1 = fullfile(imdir, [opts.label1 '.png']); end
    outdir = fullfile(imdir, 'mollified');
    if ~exist(outdir, 'dir'), mkdir(outdir); end

    for w = opts.widths
        if opts.margin <= w
            error('make_mollified_marginals:margin', ...
                  'margin (%.4g) must exceed w (%.4g)', opts.margin, w);
        end
        prob = prob_silhouette(img0 = opts.img0, img1 = opts.img1, ...
                               mollify_w = w, margin = opts.margin, ...
                               floor_frac = 0, name = opts.name);
        for n = opts.resolutions
            cfg.nt = 8; cfg.nx = n; cfg.ny = n; cfg.name = 'cache';
            cfg.disc = @disc_staggered_1st;
            p = setup_problem(cfg, prob);

            tag  = sprintf('w%.2f_%d', w, n);
            S = struct('rho0', p.rho0, 'rho1', p.rho1, ...
                       'xx', p.xx, 'yy', p.yy, 'dx', p.dx, 'dy', p.dy, ...
                       'nx', n, 'ny', n, 'mollify_w', w, 'margin', opts.margin, ...
                       'name', opts.name, 'img0', opts.img0, 'img1', opts.img1, ...
                       'created', datestr(now, 'yyyy-mm-dd HH:MM:SS')); %#ok<TNOW1,DATST>
            matname = fullfile(outdir, sprintf('%s_%s.mat', opts.name, tag));
            save(matname, '-struct', 'S');

            labs = {opts.label0, opts.label1};
            for k = 0:1
                r = S.(sprintf('rho%d', k));
                bnd = [r(1,:), r(end,:), r(:,1).', r(:,end).'];
                jmp = max(max(abs(diff(r,1,1)))) / max(r(:));
                fprintf(['%s n=%3d w=%.2f (%.2f cells)  rho%d %-8s  ' ...
                         'max=%.4f mass=%.6f bnd=%.1e jump=%.1f%%\n'], ...
                        opts.name, n, w, w*n, k, labs{k+1}, max(r(:)), ...
                        sum(r(:))*p.dx*p.dy, max(abs(bnd)), 100*jmp);
                imwrite(flipud((r/max(r(:))).'), ...
                        fullfile(outdir, sprintf('rho%d_%s_%s.png', k, labs{k+1}, tag)));
            end
            fprintf('   -> %s\n', matname);
        end
    end
end
