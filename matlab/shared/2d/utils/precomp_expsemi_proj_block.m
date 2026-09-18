function ep = precomp_expsemi_proj_block(problem, vareps)
% PRECOMP_EXPSEMI_PROJ_BLOCK  Standalone precompute for
%   proj_fokker_planck_expsemi_backslash_block_gpu.
%
%   ep = precomp_expsemi_proj_block(problem, vareps)
%
%   Computes ONLY what the block method needs. Does NOT call
%   precomp_expsemi_proj and does NOT carry Thomas-specific fields
%   (lower_T, main_T, upper_T, main_T_mod) or the raw per-mode diagonal
%   arrays (lower_all, main_all, upper_all) as outputs -- the block method
%   has no use for any of those. Keeping them around (as an earlier version
%   of this function did, by calling precomp_expsemi_proj first) roughly
%   doubles the GPU memory footprint for no benefit: the diagonals are used
%   here only transiently, to assemble ep.T_block, then discarded.
%
%   Output struct fields:
%     ep.c_vals    (1 x nx x ny)         semigroup coefficients c(kx,ky)
%     ep.phi_vals  (1 x nx x ny)         ETD weights phi(kx,ky)
%     ep.T_block   (nt*M x nt*M) sparse  block-diagonal tridiagonal system,
%                  M = nx*ny independent nt x nt blocks, one per (kx,ky)
%                  mode -- built ONCE here since the diagonals depend only
%                  on vareps/dt/grid, never the current iterate.
%     ep.tw_x, ep.w_x, ep.iw_x, ep.itw_x   (1 x nx x 1) DCT-x twiddles
%     ep.tw_y, ep.w_y, ep.iw_y, ep.itw_y   (1 x 1 x ny) DCT-y twiddles
%     (identical twiddle math to precomp_expsemi_proj / proj_fokker_planck_expsemi_gpu)
%
%   Mode linear index p = kx + (ky-1)*nx (MATLAB column-major flatten of the
%   trailing (nx,ny) dims -- matches reshape(X, nt_or_ntm, M) for any
%   (nt_or_ntm, nx, ny) array X). Block b = p-1 occupies global rows/cols
%   b*nt+1 : (b+1)*nt. The DC mode (p=1) is included like any other block
%   (main diagonal forced to 1, same as Thomas) -- the zeroed RHS at that
%   mode gives phi≈0 there too, matching thomas_solve's uniform treatment.

    nt  = problem.nt;   ntm = nt - 1;
    nx  = problem.nx;
    ny  = problem.ny;
    dt  = problem.dt;
    M   = nx * ny;

    %% --- Semigroup coefficients (same formula as precomp_expsemi_proj) ---
    lxy = reshape(problem.lambda_x, 1, nx, 1) + reshape(problem.lambda_y, 1, 1, ny);
    alpha_vals = vareps * lxy * dt;
    c_vals     = exp(-alpha_vals);

    phi_vals     = ones(1, nx, ny);
    nz           = alpha_vals > 1e-14;
    phi_vals(nz) = -expm1(-alpha_vals(nz)) ./ alpha_vals(nz);

    ep.c_vals   = c_vals;
    ep.phi_vals = phi_vals;

    %% --- Tridiagonal diagonals: LOCAL only, used to build T_block below ---
    c2     = c_vals .^ 2;
    phi2_l = phi_vals .^ 2 .* lxy;

    main_mid = (1 + c2) / dt^2 + phi2_l;
    main_all = repmat(main_mid, nt, 1, 1);
    main_all(1,  :, :) = 1/dt^2    + phi2_l;
    main_all(nt, :, :) = c2 / dt^2 + phi2_l;

    off_diag  = -c_vals / dt^2;
    lower_all = repmat(off_diag, ntm, 1, 1);
    upper_all = lower_all;

    main_all(:, 1, 1) = 1;   % DC mode: singular T -> main=1, same as Thomas

    %% --- Assemble ONE block-diagonal sparse matrix, all M modes at once ---
    main_M  = reshape(main_all,  nt,  M);
    lower_M = reshape(lower_all, ntm, M);
    upper_M = reshape(upper_all, ntm, M);

    block_off = (0:M-1) * nt;   % (1 x M) -- global offset of block p's first row/col

    main_idx  = repmat((1:nt)', 1, M) + repmat(block_off, nt, 1);
    lower_row = repmat((2:nt)',   1, M) + repmat(block_off, ntm, 1);
    lower_col = repmat((1:nt-1)', 1, M) + repmat(block_off, ntm, 1);
    upper_row = repmat((1:nt-1)', 1, M) + repmat(block_off, ntm, 1);
    upper_col = repmat((2:nt)',   1, M) + repmat(block_off, ntm, 1);

    rows = [main_idx(:);  lower_row(:);  upper_row(:)];
    cols = [main_idx(:);  lower_col(:);  upper_col(:)];
    vals = [main_M(:);    lower_M(:);    upper_M(:)];

    N = nt * M;
    ep.T_block = sparse(rows, cols, vals, N, N);

    %% --- DCT twiddle factors (same as precomp_expsemi_proj) ---
    kx = reshape(0:nx-1, 1, nx, 1);
    ep.tw_x  = exp(-1i * pi * kx / (2*nx));
    ep.w_x   = reshape([1/sqrt(nx), sqrt(2/nx)*ones(1,nx-1)]/2, 1, nx, 1);
    ep.iw_x  = 2 * ep.w_x;
    ep.itw_x = conj(ep.tw_x);

    ky = reshape(0:ny-1, 1, 1, ny);
    ep.tw_y  = exp(-1i * pi * ky / (2*ny));
    ep.w_y   = reshape([1/sqrt(ny), sqrt(2/ny)*ones(1,ny-1)]/2, 1, 1, ny);
    ep.iw_y  = 2 * ep.w_y;
    ep.itw_y = conj(ep.tw_y);
end
