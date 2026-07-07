function x_out = proj_fokker_planck_expsemi_gpu(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_GPU  2D FP projection via ETD.  GPU-compatible.
%
%   Identical to proj_fokker_planck_expsemi but replaces MATLAB's dct/idct
%   with FFT-based dct_rows/idct_rows (local) that work on gpuArray.
%
%   Requires:
%     problem.expsemi_proj   precomputed by precomp_expsemi_proj(problem, vareps)
%
%   Grid (same as proj_fokker_planck_expsemi):
%     x_in.rho  (ntm x nx  x ny)   staggered density
%     x_in.mx   (nt  x nxm x ny)   staggered x-momentum
%     x_in.my   (nt  x nx  x nym)  staggered y-momentum

    ops  = problem.ops;
    rho0 = problem.rho0;   % (nx x ny)
    rho1 = problem.rho1;   % (nx x ny)
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;
    ny   = problem.ny;
    ep   = problem.expsemi_proj;
    c_vals   = ep.c_vals;    % (1 x nx x ny)
    phi_vals = ep.phi_vals;  % (1 x nx x ny)

    mu    = x_in.rho;   % (ntm x nx x ny)
    psi_x = x_in.mx;   % (nt  x nxm x ny)
    psi_y = x_in.my;   % (nt  x nx  x nym)

    rho0_3d = reshape(rho0, 1, nx, ny);
    rho1_3d = reshape(rho1, 1, nx, ny);
    zeros_x = zeros(nt, ny, 'like', mu);
    zeros_y = zeros(nt, nx, 'like', mu);

    %% --- ETD FP residual in DCT-xy space ---
    mu_prev = cat(1, rho0_3d, mu);   % (nt x nx x ny)
    mu_curr = cat(1, mu, rho1_3d);   % (nt x nx x ny)

    mu_prev_hat = dct2_xy_gpu(mu_prev, nt, nx, ny);
    mu_curr_hat = dct2_xy_gpu(mu_curr, nt, nx, ny);
    f_rho_hat   = (mu_curr_hat - c_vals .* mu_prev_hat) / problem.dt;

    div_psi = ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
            + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y);   % (nt x nx x ny)
    div_psi_hat = dct2_xy_gpu(div_psi, nt, nx, ny);

    f_hat = f_rho_hat + phi_vals .* div_psi_hat;

    if norm(f_hat(:)) * sqrt(problem.dt * problem.dx * problem.dy) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- Batched Thomas solve for all (kx, ky) modes ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;   % DC mode handled separately below

    phi_hat = thomas_solve(ep.lower_all, ep.main_all, ep.upper_all, rhs, nt, nx, ny);

    %% --- DC spatial mode (kx=1, ky=1): singular T, handled via DCT in time ---
    f1_col         = f_hat(:, 1, 1);                                % (nt x 1)
    f1_t           = dct_rows_gpu(f1_col');                         % (1 x nt)
    phi1_t         = zeros(1, nt, 'like', f_hat);
    phi1_t(2:end)  = f1_t(2:end) ./ reshape(problem.lambda_t(2:end), 1, []);
    phi_hat(:,1,1) = idct_rows_gpu(phi1_t)';                        % (nt x 1)

    %% --- rho update: adj_rho_hat = (c * phi_next - phi_curr) / dt ---
    phi_hat_curr = phi_hat(1:ntm, :, :);     % (ntm x nx x ny)
    phi_hat_next = phi_hat(2:nt,  :, :);     % (ntm x nx x ny)
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / problem.dt;
    adj_rho      = idct2_xy_gpu(adj_rho_hat, ntm, nx, ny);

    x_out.rho = mu + adj_rho;

    %% --- mx/my update: delta_m = D_{x,y}^T [ IDCT(phi_vals .* phi_hat) ] ---
    phi_weighted = idct2_xy_gpu(phi_vals .* phi_hat, nt, nx, ny);   % (nt x nx x ny)
    x_out.mx = psi_x + ops.deriv_x_at_m(phi_weighted);
    x_out.my = psi_y + ops.deriv_y_at_m(phi_weighted);
end

% ---------------------------------------------------------------------------
% Batched Thomas (TDMA) — identical to proj_fokker_planck_expsemi.
% GPU-compatible: sequential over nt, vectorised element-wise over nx*ny.
% ---------------------------------------------------------------------------

function phi_hat = thomas_solve(lower_all, main_all, upper_all, f_hat, nt, nx, ny)
    b = main_all;
    d = f_hat;

    for i = 2:nt
        w        = lower_all(i-1,:,:) ./ b(i-1,:,:);
        b(i,:,:) = b(i,:,:) - w .* upper_all(i-1,:,:);
        d(i,:,:) = d(i,:,:) - w .* d(i-1,:,:);
    end

    phi_hat        = zeros(nt, nx, ny, 'like', f_hat);
    phi_hat(nt,:,:) = d(nt,:,:) ./ b(nt,:,:);
    for i = nt-1:-1:1
        phi_hat(i,:,:) = (d(i,:,:) - upper_all(i,:,:) .* phi_hat(i+1,:,:)) ./ b(i,:,:);
    end
end

% ---------------------------------------------------------------------------
% 2D DCT/IDCT using FFT-based row operations — GPU-compatible.
% DCT-II along spatial dims 2 (x) and 3 (y) of an (m x nx x ny) array.
% ---------------------------------------------------------------------------

function f_hat = dct2_xy_gpu(f, m, nx, ny)
    % DCT along x (dim 2): reshape so each row is a length-nx signal
    f_perm = permute(f, [1, 3, 2]);                          % (m x ny x nx)
    f_mat  = reshape(f_perm, m*ny, nx);                      % (m*ny x nx)
    f_mat  = dct_rows_gpu(f_mat);
    f_hat  = permute(reshape(f_mat, m, ny, nx), [1, 3, 2]);  % (m x nx x ny)

    % DCT along y (dim 3): reshape so each row is a length-ny signal
    f_mat  = reshape(f_hat, m*nx, ny);                       % (m*nx x ny)
    f_mat  = dct_rows_gpu(f_mat);
    f_hat  = reshape(f_mat, m, nx, ny);                      % (m x nx x ny)
end

function f = idct2_xy_gpu(f_hat, m, nx, ny)
    % IDCT along y (dim 3) first
    f_mat = reshape(f_hat, m*nx, ny);                        % (m*nx x ny)
    f_mat = idct_rows_gpu(f_mat);
    f     = reshape(f_mat, m, nx, ny);                       % (m x nx x ny)

    % IDCT along x (dim 2)
    f_perm = permute(f, [1, 3, 2]);                          % (m x ny x nx)
    f_mat  = reshape(f_perm, m*ny, nx);                      % (m*ny x nx)
    f_mat  = idct_rows_gpu(f_mat);
    f      = permute(reshape(f_mat, m, ny, nx), [1, 3, 2]);  % (m x nx x ny)
end

% ---------------------------------------------------------------------------
% FFT-based orthonormal DCT-II/III along rows.  Equivalent to dct(x')'/idct(x')'
% but works on gpuArray (no Signal Processing Toolbox dependency).
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
