function bp = precomp_banded_proj_pcr(problem, vareps)
% PRECOMP_BANDED_PROJ_PCR  Precompute for proj_fokker_planck_pcr (PCR tridiagonal solver).
%
%   bp = precomp_banded_proj_pcr(problem, vareps)
%
%   Builds the same tridiagonal diagonals as precomp_banded_proj, plus:
%     bp.a_all   (nt x nx x ny)  extended lower diagonal, a_all(1,:,:) = 0
%     bp.c_all   (nt x nx x ny)  extended upper diagonal, c_all(nt,:,:) = 0
%     bp.n_levels  scalar          log2(nt) — number of PCR reduction levels
%
%   Requires nt to be a power of 2.
%
%   Also pre-allocates the PCR work buffers (CPU doubles) so
%   proj_fokker_planck_pcr can reuse them each iteration:
%     bp.a_buf, bp.b_buf, bp.c_buf, bp.d_buf   (nt x nx x ny)
%     bp.a_new, bp.b_new, bp.c_new, bp.d_new   (nt x nx x ny)
%
%   The diagonals and work buffers are cast to gpuArray in
%   discretize_then_optimize when cfg.use_gpu is true.

    assert(mod(log2(problem.nt), 1) == 0, ...
        'precomp_banded_proj_pcr: nt=%d must be a power of 2.', problem.nt);

    % Reuse base precomputation (lower_all, main_all, upper_all)
    bp_base      = precomp_banded_proj(problem, vareps);
    bp.lower_all = bp_base.lower_all;
    bp.main_all  = bp_base.main_all;
    bp.upper_all = bp_base.upper_all;

    nt = problem.nt;
    nx = problem.nx;
    ny = problem.ny;

    % Extended diagonals: (nt x nx x ny) with zero-padding at boundaries
    bp.a_all = cat(1, zeros(1, nx, ny), bp.lower_all);  % a(1,:,:) = 0
    bp.c_all = cat(1, bp.upper_all, zeros(1, nx, ny));  % c(nt,:,:) = 0

    % DC mode (kx=1,ky=1): singular T, handled separately.
    % Set a=0, b=1, c=0 so PCR returns phi=0 for zero RHS.
    bp.a_all(:, 1, 1) = 0;
    bp.main_all(:, 1, 1) = 1;
    bp.c_all(:, 1, 1) = 0;

    bp.n_levels = round(log2(nt));

    % Pre-allocated work buffers (reused each ADMM iteration)
    bp.a_buf = zeros(nt, nx, ny);
    bp.b_buf = zeros(nt, nx, ny);
    bp.c_buf = zeros(nt, nx, ny);
    bp.d_buf = zeros(nt, nx, ny);
    bp.a_new = zeros(nt, nx, ny);
    bp.b_new = zeros(nt, nx, ny);
    bp.c_new = zeros(nt, nx, ny);
    bp.d_new = zeros(nt, nx, ny);
end
