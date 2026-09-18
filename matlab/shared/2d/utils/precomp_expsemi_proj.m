function ep = precomp_expsemi_proj(problem, vareps)
% PRECOMP_EXPSEMI_PROJ  Precompute batched Thomas diagonals for the 2D ETD
%   projection  proj_fokker_planck_expsemi.
%
%   ep = precomp_expsemi_proj(problem, vareps)
%
%   The 2D Fokker–Planck constraint with the exact heat semigroup decouples
%   under 2D DCT (DCT-xy) into nx*ny independent (nt x nt) tridiagonal
%   systems, one per spatial mode (kx, ky).  The combined eigenvalue is
%
%       lxy(kx,ky) = lambda_x(kx) + lambda_y(ky),
%
%   and the ETD parameters per mode are
%
%       alpha  = vareps * lxy * dt
%       c      = exp(-alpha)                     exact semigroup factor
%       phi    = (1 - c) / alpha   (phi->1 as alpha->0)   ETD weight
%
%   The per-mode tridiagonal T_{kx,ky} (nt x nt) has
%
%       main(1)        = 1/dt^2      + phi^2 * lxy
%       main(2..nt-1)  = (1+c^2)/dt^2 + phi^2 * lxy
%       main(nt)       = c^2/dt^2   + phi^2 * lxy
%       off-diagonals  = -c / dt^2                 (upper = lower)
%
%   Mode (kx=1, ky=1): lxy = 0, T = M0 is singular (time Laplacian).
%   Its main diagonal is set to 1 so the batched Thomas solver returns zero
%   when given a zero RHS; the projection handles this mode separately
%   via a DCT in time.
%
%   Output struct fields:
%     ep.c_vals     (1 x nx x ny)    semigroup coefficients c(kx,ky)
%     ep.phi_vals   (1 x nx x ny)    ETD weights phi(kx,ky)
%     ep.lower_all  (ntm x nx x ny)  lower diagonals for all modes
%     ep.main_all   (nt  x nx x ny)  main  diagonals for all modes
%     ep.upper_all  (ntm x nx x ny)  upper diagonals for all modes
%
%   Compatible with gpuArray: cast output fields after this call if needed.

    nt  = problem.nt;   ntm = nt - 1;
    nx  = problem.nx;
    ny  = problem.ny;
    dt  = problem.dt;

    % Combined spatial eigenvalue: (1 x nx x ny)
    lxy = reshape(problem.lambda_x, 1, nx, 1) + reshape(problem.lambda_y, 1, 1, ny);

    alpha_vals = vareps * lxy * dt;   % (1 x nx x ny)
    c_vals     = exp(-alpha_vals);    % (1 x nx x ny)

    % ETD weight: phi = (1 - exp(-alpha)) / alpha, with phi(0) = 1
    % Computed via expm1 to avoid catastrophic cancellation in 1 - exp(-alpha)
    % for small alpha (guarantees phi < 1 exactly, as the math requires).
    phi_vals       = ones(1, nx, ny);
    nz             = alpha_vals > 1e-14;
    phi_vals(nz)   = -expm1(-alpha_vals(nz)) ./ alpha_vals(nz);

    ep.c_vals   = c_vals;
    ep.phi_vals = phi_vals;

    % Batched Thomas diagonals vectorised over all (kx, ky) modes.
    c2     = c_vals .^ 2;          % (1 x nx x ny)
    phi2_l = phi_vals .^ 2 .* lxy; % (1 x nx x ny)

    % main_all: interior rows have value (1+c^2)/dt^2 + phi^2*lxy; endpoints differ.
    main_mid             = (1 + c2) / dt^2 + phi2_l;   % (1 x nx x ny)
    ep.main_all          = repmat(main_mid, nt, 1, 1);  % (nt x nx x ny)
    ep.main_all(1,  :,:) = 1/dt^2      + phi2_l;
    ep.main_all(nt, :,:) = c2 / dt^2   + phi2_l;

    % off-diagonals: constant -c/dt^2 for all time rows and all modes
    off_diag     = -c_vals / dt^2;                           % (1 x nx x ny)
    ep.lower_all = repmat(off_diag, ntm, 1, 1);              % (ntm x nx x ny)
    ep.upper_all = ep.lower_all;

    % Mode (1,1): lxy = 0 -> singular T; set main = 1 so Thomas gives 0 on 0 RHS.
    ep.main_all(:, 1, 1) = 1;

    % --- GPU-optimised Thomas layout: (M x nt/ntm) where M = nx*ny ---
    % Stored alongside the standard layout so proj_fokker_planck_expsemi_gpu can
    % use coalesced column access  b(:,i)  instead of strided  b(i,:,:).
    M = nx * ny;
    ep.lower_T = reshape(permute(ep.lower_all, [2,3,1]), M, ntm);
    ep.main_T  = reshape(permute(ep.main_all,  [2,3,1]), M, nt);
    ep.upper_T = reshape(permute(ep.upper_all, [2,3,1]), M, ntm);

    % --- Precomputed Thomas forward sweep (matches 1D thomas_batch_precomp) ---
    % Runs the forward elimination on main_T once so thomas_solve only modifies
    % the RHS, never the diagonal.  Avoids relying on gpuArray copy-on-write
    % (which can silently corrupt ep.main_T when b = ep.main_T is written to
    % inside a function that received ep as a by-value struct copy).
    main_T_mod = ep.main_T;
    for j = 2:nt
        w = ep.lower_T(:, j-1) ./ main_T_mod(:, j-1);
        main_T_mod(:, j) = main_T_mod(:, j) - w .* ep.upper_T(:, j-1);
    end
    ep.main_T_mod = main_T_mod;

    % --- Precomputed DCT twiddle factors (1 x nx x 1) and (1 x 1 x ny) ---
    % Eliminates exp() recomputation every projection call.
    kx = reshape(0:nx-1, 1, nx, 1);
    ep.tw_x  = exp(-1i * pi * kx / (2*nx));                     % (1 x nx x 1)
    ep.w_x   = reshape([1/sqrt(nx), sqrt(2/nx)*ones(1,nx-1)]/2, 1, nx, 1);  % forward (absorbs ×2 from FFT even-ext)
    ep.iw_x  = 2 * ep.w_x;                                                  % inverse (full synthesis weight)
    ep.itw_x = conj(ep.tw_x);

    ky = reshape(0:ny-1, 1, 1, ny);
    ep.tw_y  = exp(-1i * pi * ky / (2*ny));                     % (1 x 1 x ny)
    ep.w_y   = reshape([1/sqrt(ny), sqrt(2/ny)*ones(1,ny-1)]/2, 1, 1, ny);
    ep.iw_y  = 2 * ep.w_y;
    ep.itw_y = conj(ep.tw_y);
end
