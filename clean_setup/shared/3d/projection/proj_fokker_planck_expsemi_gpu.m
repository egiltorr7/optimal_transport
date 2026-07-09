function x_out = proj_fokker_planck_expsemi_gpu(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_GPU  3D FP projection via ETD.  GPU-compatible.
%
%   x_out = proj_fokker_planck_expsemi_gpu(x_in, problem, cfg)
%
%   Solves the L2 projection onto the 3D Fokker–Planck constraint using:
%     - 3D DCT along spatial dims 2/3/4 (direct fft along each dim, no permutes)
%     - Batched Thomas TDMA with (M x nt) layout (M = nx*ny*nz) for coalesced
%       GPU access; the permute to get this layout is done once per call
%     - DC spatial mode (1,1,1) inverted via DCT in time
%
%   Requires:
%     problem.expsemi_proj   precomputed by precomp_expsemi_proj (3D version)
%
%   Grid:
%     x_in.rho  (ntm x nx  x ny  x nz )  staggered density
%     x_in.mx   (nt  x nxm x ny  x nz )  staggered x-momentum
%     x_in.my   (nt  x nx  x nym x nz )  staggered y-momentum
%     x_in.mz   (nt  x nx  x ny  x nzm)  staggered z-momentum

    ops  = problem.ops;
    rho0 = problem.rho0;   % (nx x ny x nz)
    rho1 = problem.rho1;   % (nx x ny x nz)
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;
    ny   = problem.ny;
    nz   = problem.nz;
    ep   = problem.expsemi_proj;
    c_vals   = ep.c_vals;    % (1 x nx x ny x nz)
    phi_vals = ep.phi_vals;  % (1 x nx x ny x nz)

    mu    = x_in.rho;   % (ntm x nx x ny x nz)
    psi_x = x_in.mx;   % (nt  x nxm x ny x nz)
    psi_y = x_in.my;   % (nt  x nx  x nym x nz)
    psi_z = x_in.mz;   % (nt  x nx  x ny  x nzm)

    rho0_4d = reshape(rho0, 1, nx, ny, nz);
    rho1_4d = reshape(rho1, 1, nx, ny, nz);
    zeros_x = zeros(nt, ny, nz, 'like', mu);   % x-wall BCs
    zeros_y = zeros(nt, nx, nz, 'like', mu);   % y-wall BCs
    zeros_z = zeros(nt, nx, ny, 'like', mu);   % z-wall BCs

    %% --- ETD FP residual in DCT-xyz space ---
    mu_prev = cat(1, rho0_4d, mu);   % (nt x nx x ny x nz)
    mu_curr = cat(1, mu, rho1_4d);

    mu_prev_hat = dct3_xyz(mu_prev, ep);
    mu_curr_hat = dct3_xyz(mu_curr, ep);
    f_rho_hat   = (mu_curr_hat - c_vals .* mu_prev_hat) / problem.dt;

    div_psi = ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
            + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y) ...
            + ops.deriv_z_at_phi(psi_z, zeros_z, zeros_z);   % (nt x nx x ny x nz)
    div_psi_hat = dct3_xyz(div_psi, ep);

    f_hat = f_rho_hat + phi_vals .* div_psi_hat;

    if norm(f_hat(:)) * sqrt(problem.dt * problem.dx * problem.dy * problem.dz) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- Batched Thomas solve for all (kx,ky,kz) modes ---
    rhs             = f_hat;
    rhs(:, 1, 1, 1) = 0;   % DC mode handled separately

    phi_hat = thomas_solve(ep, rhs, nt, nx, ny, nz);

    %% --- DC spatial mode (1,1,1): singular T, solve via DCT in time ---
    f1_col         = f_hat(:, 1, 1, 1);                          % (nt x 1)
    f1_t           = dct_rows_gpu(f1_col');                       % (1 x nt)
    phi1_t         = zeros(1, nt, 'like', f_hat);
    phi1_t(2:end)  = f1_t(2:end) ./ reshape(problem.lambda_t(2:end), 1, []);
    phi_hat(:,1,1,1) = idct_rows_gpu(phi1_t)';

    %% --- rho update ---
    phi_hat_curr = phi_hat(1:ntm, :, :, :);
    phi_hat_next = phi_hat(2:nt,  :, :, :);
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / problem.dt;
    adj_rho      = idct3_xyz(adj_rho_hat, ep);

    x_out.rho = mu + adj_rho;

    %% --- mx / my / mz update ---
    phi_weighted = idct3_xyz(phi_vals .* phi_hat, ep);
    x_out.mx = psi_x + ops.deriv_x_at_m(phi_weighted);
    x_out.my = psi_y + ops.deriv_y_at_m(phi_weighted);
    x_out.mz = psi_z + ops.deriv_z_at_m(phi_weighted);
end

% ---------------------------------------------------------------------------
% 3-D DCT-II along spatial dims 2, 3, 4 (x, y, z) — no permutes.
% ---------------------------------------------------------------------------

function f_hat = dct3_xyz(f, ep)
    % DCT along x (dim 2)
    Nx = size(f, 2);
    xe = cat(2, f, f(:, Nx:-1:1, :, :));
    V  = fft(xe, [], 2);
    f_hat = real(V(:, 1:Nx, :, :) .* ep.tw_x) .* ep.w_x;

    % DCT along y (dim 3)
    Ny = size(f_hat, 3);
    ye = cat(3, f_hat, f_hat(:, :, Ny:-1:1, :));
    V  = fft(ye, [], 3);
    f_hat = real(V(:, :, 1:Ny, :) .* ep.tw_y) .* ep.w_y;

    % DCT along z (dim 4)
    Nz = size(f_hat, 4);
    ze = cat(4, f_hat, f_hat(:, :, :, Nz:-1:1));
    V  = fft(ze, [], 4);
    f_hat = real(V(:, :, :, 1:Nz) .* ep.tw_z) .* ep.w_z;
end

function f = idct3_xyz(f_hat, ep)
    % IDCT along z (dim 4) first
    [m, nx_sz, ny_sz, Nz] = size(f_hat);
    Z  = f_hat .* ep.iw_z;
    U  = Z .* ep.itw_z;
    ze = cat(4, U, zeros(m, nx_sz, ny_sz, Nz, 'like', U));
    f  = real(ifft(ze, [], 4)) * (2*Nz);
    f  = f(:, :, :, 1:Nz);

    % IDCT along y (dim 3)
    [m, nx_sz, Ny, nz_sz] = size(f);
    Z  = f .* ep.iw_y;
    U  = Z .* ep.itw_y;
    ye = cat(3, U, zeros(m, nx_sz, Ny, nz_sz, 'like', U));
    f  = real(ifft(ye, [], 3)) * (2*Ny);
    f  = f(:, :, 1:Ny, :);

    % IDCT along x (dim 2)
    [m, Nx, ny_sz, nz_sz] = size(f);
    Z  = f .* ep.iw_x;
    U  = Z .* ep.itw_x;
    xe = cat(2, U, zeros(m, Nx, ny_sz, nz_sz, 'like', U));
    f  = real(ifft(xe, [], 2)) * (2*Nx);
    f  = f(:, 1:Nx, :, :);
end

% ---------------------------------------------------------------------------
% Thomas (TDMA) — permute f_hat to (M x nt) for coalesced GPU column access.
% M = nx*ny*nz.  Diagonals stored in (nt x nx x ny x nz) layout in ep;
% the reshape to (M x nt) is done per-call (avoids double memory for 3D).
% ---------------------------------------------------------------------------

function phi_hat = thomas_solve(ep, f_hat, nt, nx, ny, nz)
    M = nx * ny * nz;

    % Permute (nt x nx x ny x nz) -> (nx x ny x nz x nt) -> (M x nt)
    d = reshape(permute(f_hat,       [2, 3, 4, 1]), M, nt);
    b = reshape(permute(ep.main_all, [2, 3, 4, 1]), M, nt);
    lower_r = reshape(permute(ep.lower_all, [2, 3, 4, 1]), M, nt-1);
    upper_r = reshape(permute(ep.upper_all, [2, 3, 4, 1]), M, nt-1);

    % Forward sweep
    for i = 2:nt
        w       = lower_r(:, i-1) ./ b(:, i-1);
        b(:, i) = b(:, i) - w .* upper_r(:, i-1);
        d(:, i) = d(:, i) - w .* d(:, i-1);
    end

    % Back substitution
    phi_r        = zeros(M, nt, 'like', f_hat);
    phi_r(:, nt) = d(:, nt) ./ b(:, nt);
    for i = nt-1:-1:1
        phi_r(:, i) = (d(:, i) - upper_r(:, i) .* phi_r(:, i+1)) ./ b(:, i);
    end

    % Permute back (M x nt) -> (nx x ny x nz x nt) -> (nt x nx x ny x nz)
    phi_hat = permute(reshape(phi_r, nx, ny, nz, nt), [4, 1, 2, 3]);
end

% ---------------------------------------------------------------------------
% FFT-based DCT-II/III along rows — for 1-D time DCT of DC spatial mode.
% ---------------------------------------------------------------------------

function X = dct_rows_gpu(x)
    [~, N] = size(x);
    xe  = [x, x(:, N:-1:1)];
    V   = fft(xe, [], 2);
    k   = 0:N-1;
    tw  = exp(-1i * pi * k / (2*N));
    raw = real(V(:, 1:N) .* tw);
    w   = [1/sqrt(N), sqrt(2/N) * ones(1, N-1)];
    X   = raw .* (w / 2);
end

function x = idct_rows_gpu(X)
    [m, N] = size(X);
    w  = [1/sqrt(N), sqrt(2/N) * ones(1, N-1)];
    Z  = X .* w;
    k  = 0:N-1;
    U  = complex(Z) .* exp(1i * pi * k / (2*N));
    x  = real(ifft([U, zeros(m, N, 'like', U)], [], 2)) * (2*N);
    x  = x(:, 1:N);
end
