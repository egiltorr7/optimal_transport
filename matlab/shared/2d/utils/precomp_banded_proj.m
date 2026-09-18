function bp = precomp_banded_proj(problem, vareps)
% PRECOMP_BANDED_PROJ  Precompute tridiagonal diagonals for proj_fokker_planck_banded (2D).
%
%   bp = precomp_banded_proj(problem, vareps)
%
%   The FP operator AA* decouples under DCT-xy into nx*ny tridiagonal systems
%   indexed by (kx, ky) with combined eigenvalue lxy = lambda_x(kx) + lambda_y(ky):
%
%     T_{kx,ky} = M0 + lxy * diag(M1d) + lxy^2 * M2
%
%   where M0, M1d, M2 are nt x nt time-direction matrices.
%   Since T is tridiagonal, we precompute all diagonals vectorised over (kx, ky):
%
%     bp.lower_all  (nt-1 x nx x ny)   lower diagonals for all modes
%     bp.main_all   (nt   x nx x ny)   main  diagonals (mode (1,1) set to 1)
%     bp.upper_all  (nt-1 x nx x ny)   upper diagonals for all modes
%
%   Mode (1,1) has lxy=0 and a singular T; it is handled separately via DCT-t
%   in proj_fokker_planck_banded.  Its main diagonal is set to 1 so the batched
%   Thomas solver produces 0 when given a zero RHS for that mode.
%
%   All output arrays are regular (CPU) doubles.  Cast to gpuArray in
%   discretize_then_optimize if cfg.use_gpu is set.

    nt  = problem.nt;   ntm = nt - 1;
    nx  = problem.nx;
    ny  = problem.ny;
    dt  = problem.dt;

    % Time operators: (nt x ntm) and (ntm x nt)
    It_phi = 0.5 * toeplitz([1 1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]);
    Dt_phi =       toeplitz([1 -1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]) / dt;
    It_rho = 0.5 * toeplitz([1 zeros(1,ntm-1)], [1 1 zeros(1,ntm-1)]);
    Dt_rho =       toeplitz([-1 zeros(1,ntm-1)], [-1 1 zeros(1,ntm-1)]) / dt;

    % T building blocks (nt x nt, all tridiagonal)
    M0_bnd = -Dt_phi * Dt_rho;
    M1d    = ones(nt, 1);
    M1d(1) = 1 + vareps / dt;
    M1d(nt)= 1 - vareps / dt;
    M2_bnd = vareps^2 * (It_phi * It_rho);

    % Extract tridiagonal diagonals as column vectors
    lower_M0 = diag(M0_bnd, -1);   % (ntm x 1)
    main_M0  = diag(M0_bnd);       % (nt  x 1)
    upper_M0 = diag(M0_bnd,  1);   % (ntm x 1)
    lower_M2 = diag(M2_bnd, -1);
    main_M2  = diag(M2_bnd);
    upper_M2 = diag(M2_bnd,  1);

    % Combined spatial eigenvalue lxy = lambda_x(kx) + lambda_y(ky): (1 x nx x ny)
    lxy  = reshape(problem.lambda_x, 1, nx, 1) + reshape(problem.lambda_y, 1, 1, ny);
    lxy2 = lxy .^ 2;

    % Reshape diagonal vectors for broadcasting: (ntm/nt x 1 x 1)
    lower_M0 = reshape(lower_M0, ntm, 1, 1);
    upper_M0 = reshape(upper_M0, ntm, 1, 1);
    lower_M2 = reshape(lower_M2, ntm, 1, 1);
    upper_M2 = reshape(upper_M2, ntm, 1, 1);
    main_M0  = reshape(main_M0,  nt,  1, 1);
    main_M2  = reshape(main_M2,  nt,  1, 1);
    M1d      = reshape(M1d,      nt,  1, 1);

    % Vectorised diagonals for all (kx, ky) modes
    bp.lower_all = lower_M0 + lxy2 .* lower_M2;                    % (ntm x nx x ny)
    bp.main_all  = main_M0  + lxy  .* M1d + lxy2 .* main_M2;      % (nt  x nx x ny)
    bp.upper_all = upper_M0 + lxy2 .* upper_M2;                    % (ntm x nx x ny)

    % Mode (1,1): lxy=0, T=M0 is singular.  Set main diagonal to 1 so Thomas
    % gives 0 for a zero RHS; proj_fokker_planck_banded overwrites this mode.
    bp.main_all(:, 1, 1) = 1;
end
