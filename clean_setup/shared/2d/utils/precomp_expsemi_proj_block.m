function ep = precomp_expsemi_proj_block(problem, vareps)
% PRECOMP_EXPSEMI_PROJ_BLOCK  Precompute for
%   proj_fokker_planck_expsemi_backslash_block_gpu.
%
%   ep = precomp_expsemi_proj_block(problem, vareps)
%
%   Strict superset of precomp_expsemi_proj: same fields, plus one more:
%
%     ep.T_block  (nt*M x nt*M) sparse -- M = nx*ny independent nt x nt
%                 tridiagonal blocks stacked block-diagonally, one per
%                 (kx,ky) mode. Built ONCE here, not per ADMM iteration,
%                 since main_all/lower_all/upper_all depend only on
%                 vareps/dt/grid, never on the current iterate -- only the
%                 RHS changes call to call.
%
%   Mode linear index p = kx + (ky-1)*nx (MATLAB column-major flatten of the
%   trailing (nx,ny) dims -- matches reshape(X, nt, M) for any (nt,nx,ny)
%   array X). Block b = p-1 occupies global rows/cols b*nt+1 : (b+1)*nt.
%   The DC mode (p=1) is included like any other block (main_all(:,1,1)=1,
%   off-diagonals left as computed) -- same uniform treatment thomas_solve
%   already relies on; the zeroed RHS at that mode gives phi≈0 there too.
%
%   Because this is a superset of precomp_expsemi_proj's fields, it is a
%   safe drop-in replacement anywhere the extra ep.T_block field is simply
%   unused (all other expsemi_* projections ignore it).
%
%   Requires Signal Processing Toolbox is NOT needed here -- pure sparse
%   index arithmetic, no dct/fft calls.

    ep = precomp_expsemi_proj(problem, vareps);

    nt  = problem.nt;   ntm = nt - 1;
    nx  = problem.nx;
    ny  = problem.ny;
    M   = nx * ny;

    main_M  = reshape(ep.main_all,  nt,  M);
    lower_M = reshape(ep.lower_all, ntm, M);
    upper_M = reshape(ep.upper_all, ntm, M);

    block_off = (0:M-1) * nt;   % (1 x M) -- global offset of block p's first row/col

    % Main diagonal: nt x M entries, global row == global col
    main_idx = repmat((1:nt)', 1, M) + repmat(block_off, nt, 1);

    % Lower diagonal: connects local row i (2..nt) to local col i-1
    lower_row = repmat((2:nt)',   1, M) + repmat(block_off, ntm, 1);
    lower_col = repmat((1:nt-1)', 1, M) + repmat(block_off, ntm, 1);

    % Upper diagonal: connects local row i (1..nt-1) to local col i+1
    upper_row = repmat((1:nt-1)', 1, M) + repmat(block_off, ntm, 1);
    upper_col = repmat((2:nt)',   1, M) + repmat(block_off, ntm, 1);

    rows = [main_idx(:);  lower_row(:);  upper_row(:)];
    cols = [main_idx(:);  lower_col(:);  upper_col(:)];
    vals = [main_M(:);    lower_M(:);    upper_M(:)];

    N = nt * M;
    ep.T_block = sparse(rows, cols, vals, N, N);
end
