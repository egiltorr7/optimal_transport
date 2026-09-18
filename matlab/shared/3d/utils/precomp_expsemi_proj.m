function ep = precomp_expsemi_proj(problem, vareps)
% PRECOMP_EXPSEMI_PROJ  Precompute batched Thomas diagonals for the 3D ETD
%   projection  proj_fokker_planck_expsemi_gpu (3D version).
%
%   ep = precomp_expsemi_proj(problem, vareps)
%
%   The 3D Fokker–Planck constraint decouples under DCT-xyz into nx*ny*nz
%   independent tridiagonal systems.  The combined spatial eigenvalue is
%
%       lxyz(kx,ky,kz) = lambda_x(kx) + lambda_y(ky) + lambda_z(kz)
%
%   ETD parameters per mode:
%       alpha = vareps * lxyz * dt,   c = exp(-alpha),   phi = (1-c)/alpha
%
%   Thomas diagonals for mode (kx,ky,kz) (nt x nt tridiagonal):
%       main(1)       = 1/dt^2      + phi^2 * lxyz
%       main(2..nt-1) = (1+c^2)/dt^2 + phi^2 * lxyz
%       main(nt)      = c^2/dt^2   + phi^2 * lxyz
%       off-diagonals = -c/dt^2
%
%   Mode (1,1,1): lxyz = 0 -> singular T.  main(:,1,1,1) set to 1.
%
%   Output fields:
%     ep.c_vals      (1 x nx x ny x nz)
%     ep.phi_vals    (1 x nx x ny x nz)
%     ep.lower_all   (ntm x nx x ny x nz)
%     ep.main_all    (nt  x nx x ny x nz)
%     ep.upper_all   (ntm x nx x ny x nz)
%     ep.tw_x        (1 x nx x 1  x 1 )   DCT-x twiddle factors
%     ep.w_x         (1 x nx x 1  x 1 )   DCT-x weights
%     ep.itw_x       (1 x nx x 1  x 1 )   IDCT-x twiddles (conj of tw_x)
%     ep.tw_y        (1 x 1  x ny x 1 )   DCT-y twiddles
%     ep.w_y         (1 x 1  x ny x 1 )
%     ep.itw_y       (1 x 1  x ny x 1 )
%     ep.tw_z        (1 x 1  x 1  x nz)   DCT-z twiddles
%     ep.w_z         (1 x 1  x 1  x nz)
%     ep.itw_z       (1 x 1  x 1  x nz)

    nt  = problem.nt;   ntm = nt - 1;
    nx  = problem.nx;
    ny  = problem.ny;
    nz  = problem.nz;
    dt  = problem.dt;

    % Combined spatial eigenvalue: (1 x nx x ny x nz) via broadcast
    lxyz = reshape(problem.lambda_x, 1, nx, 1,  1 ) ...
         + reshape(problem.lambda_y, 1, 1,  ny, 1 ) ...
         + reshape(problem.lambda_z, 1, 1,  1,  nz);

    alpha_vals = vareps * lxyz * dt;
    c_vals     = exp(-alpha_vals);

    phi_vals     = ones(1, nx, ny, nz);
    nz_mask      = alpha_vals > 1e-14;
    phi_vals(nz_mask) = (1 - c_vals(nz_mask)) ./ alpha_vals(nz_mask);

    ep.c_vals   = c_vals;
    ep.phi_vals = phi_vals;

    c2     = c_vals .^ 2;
    phi2_l = phi_vals .^ 2 .* lxyz;

    main_mid             = (1 + c2) / dt^2 + phi2_l;
    ep.main_all          = repmat(main_mid, nt, 1, 1, 1);
    ep.main_all(1,  :,:,:) = 1/dt^2      + phi2_l;
    ep.main_all(nt, :,:,:) = c2 / dt^2   + phi2_l;

    off_diag     = -c_vals / dt^2;
    ep.lower_all = repmat(off_diag, ntm, 1, 1, 1);
    ep.upper_all = ep.lower_all;

    ep.main_all(:, 1, 1, 1) = 1;

    % --- Precomputed Thomas forward sweep ---
    M = nx * ny * nz;
    lower_r = reshape(permute(ep.lower_all, [2, 3, 4, 1]), M, nt-1);
    main_r  = reshape(permute(ep.main_all,  [2, 3, 4, 1]), M, nt);
    upper_r = reshape(permute(ep.upper_all, [2, 3, 4, 1]), M, nt-1);
    for j = 2:nt
        w = lower_r(:, j-1) ./ main_r(:, j-1);
        main_r(:, j) = main_r(:, j) - w .* upper_r(:, j-1);
    end
    ep.main_T_mod = main_r;

    % --- Precomputed DCT twiddle factors ---
    kx = reshape(0:nx-1, 1, nx, 1, 1);
    ep.tw_x  = exp(-1i * pi * kx / (2*nx));
    ep.w_x   = reshape([1/sqrt(nx), sqrt(2/nx)*ones(1,nx-1)]/2, 1, nx, 1, 1);
    ep.iw_x  = 2 * ep.w_x;
    ep.itw_x = conj(ep.tw_x);

    ky = reshape(0:ny-1, 1, 1, ny, 1);
    ep.tw_y  = exp(-1i * pi * ky / (2*ny));
    ep.w_y   = reshape([1/sqrt(ny), sqrt(2/ny)*ones(1,ny-1)]/2, 1, 1, ny, 1);
    ep.iw_y  = 2 * ep.w_y;
    ep.itw_y = conj(ep.tw_y);

    kz = reshape(0:nz-1, 1, 1, 1, nz);
    ep.tw_z  = exp(-1i * pi * kz / (2*nz));
    ep.w_z   = reshape([1/sqrt(nz), sqrt(2/nz)*ones(1,nz-1)]/2, 1, 1, 1, nz);
    ep.iw_z  = 2 * ep.w_z;
    ep.itw_z = conj(ep.tw_z);
end
