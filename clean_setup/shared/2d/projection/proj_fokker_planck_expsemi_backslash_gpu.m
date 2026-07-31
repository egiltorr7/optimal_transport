function x_out = proj_fokker_planck_expsemi_backslash_gpu(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_BACKSLASH_GPU  ETD FP projection via mode-by-mode
%   sparse gpuArray backslash.
%
%   x_out = proj_fokker_planck_expsemi_backslash_gpu(x_in, problem, cfg)
%
%   Uses the SAME twiddle-based, permute-free DCT infrastructure and
%   residual/update steps as proj_fokker_planck_expsemi_gpu. The ONLY
%   difference is the linear solve: this variant solves each (kx,ky)
%   tridiagonal system independently via sparse gpuArray backslash
%   (T = sparse(...) with gpuArray inputs; T \ b) in a sequential double
%   loop over modes, instead of the batched Thomas sweep in thomas_solve.
%
%   Purpose: isolate BATCHING from DEVICE. This is one cell of a 2x2:
%
%                    CPU                              GPU
%     sparse    proj_fokker_planck_          proj_fokker_planck_
%     backslash expsemi_backslash            expsemi_backslash_gpu  (this file)
%
%     batched   proj_fokker_planck_          proj_fokker_planck_
%     Thomas    expsemi                      expsemi_gpu
%
%   Holding the DCT/residual/update code identical across all four isolates
%   what "GPU" buys you (rows) from what "batching" buys you (columns).
%
%   Expected result: slow, quite possibly slower than the CPU sparse version.
%   MathWorks documents that GPU sparse mldivide does not get the memory-
%   coalescing benefit dense GPU algorithms do, and this loop calls it
%   M = nx*ny times on tiny (nt x nt) systems -- per-call kernel-launch
%   overhead dominates for problems this small. Start with a small grid
%   (e.g. nx=ny=16) before scaling up.
%
%   Requires:
%     problem.expsemi_proj   precomputed by precomp_expsemi_proj(problem,vareps)
%     Fields used: c_vals, phi_vals, lower_all, main_all, upper_all,
%                  tw_x, w_x, iw_x, itw_x, tw_y, w_y, iw_y, itw_y
%
%   Grid:
%     x_in.rho  (ntm x nx  x ny)
%     x_in.mx   (nt  x nxm x ny)
%     x_in.my   (nt  x nx  x nym)

    ops = problem.ops;
    nt  = problem.nt;   ntm = nt - 1;
    nx  = problem.nx;
    ny  = problem.ny;
    ep  = problem.expsemi_proj;

    % Cast to GPU if not already. Self-contained regardless of caller state,
    % but for TIMING this should be a no-op -- pre-cast problem/x_in to GPU
    % ONCE outside the timed region (see bench_projection_gpu.m), otherwise
    % every timed call repeats the host->device transfer, which a real ADMM
    % loop never pays after the first iteration.
    mu       = to_gpu(x_in.rho);
    psi_x    = to_gpu(x_in.mx);
    psi_y    = to_gpu(x_in.my);
    rho0     = to_gpu(problem.rho0);
    rho1     = to_gpu(problem.rho1);
    lambda_t = to_gpu(problem.lambda_t);
    c_vals    = to_gpu(ep.c_vals);
    phi_vals  = to_gpu(ep.phi_vals);
    lower_all = to_gpu(ep.lower_all);
    main_all  = to_gpu(ep.main_all);
    upper_all = to_gpu(ep.upper_all);

    tw.tw_x = to_gpu(ep.tw_x);  tw.w_x = to_gpu(ep.w_x);
    tw.iw_x = to_gpu(ep.iw_x);  tw.itw_x = to_gpu(ep.itw_x);
    tw.tw_y = to_gpu(ep.tw_y);  tw.w_y = to_gpu(ep.w_y);
    tw.iw_y = to_gpu(ep.iw_y);  tw.itw_y = to_gpu(ep.itw_y);

    rho0_3d = reshape(rho0, 1, nx, ny);
    rho1_3d = reshape(rho1, 1, nx, ny);
    zeros_x = zeros(nt, ny, 'like', mu);
    zeros_y = zeros(nt, nx, 'like', mu);

    %% --- ETD FP residual in DCT-xy space ---
    mu_prev = cat(1, rho0_3d, mu);
    mu_curr = cat(1, mu, rho1_3d);

    mu_prev_hat = dct2_xy(mu_prev, tw);
    mu_curr_hat = dct2_xy(mu_curr, tw);
    f_rho_hat   = (mu_curr_hat - c_vals .* mu_prev_hat) / problem.dt;

    div_psi = ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
            + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y);
    div_psi_hat = dct2_xy(div_psi, tw);

    f_hat = f_rho_hat + phi_vals .* div_psi_hat;

    if norm(f_hat(:)) * sqrt(problem.dt * problem.dx * problem.dy) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- Mode-by-mode sparse gpuArray backslash (kx,ky) != (1,1) ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;   % DC mode handled separately below

    phi_hat = zeros(nt, nx, ny, 'like', f_hat);

    i_diag  = gpuArray((1:nt)');
    i_lower = gpuArray((2:nt)');    j_lower = gpuArray((1:nt-1)');
    i_upper = gpuArray((1:nt-1)');  j_upper = gpuArray((2:nt)');

    for ky = 1:ny
        for kx = 1:nx
            if kx == 1 && ky == 1
                continue;   % DC mode: singular T, handled separately below
            end

            lo = lower_all(:, kx, ky);   % (nt-1 x 1) gpuArray
            ma = main_all(:,  kx, ky);   % (nt   x 1) gpuArray
            up = upper_all(:, kx, ky);   % (nt-1 x 1) gpuArray

            T = sparse([i_diag;  i_lower; i_upper], ...
                       [i_diag;  j_lower; j_upper], ...
                       [ma;      lo;      up],  nt, nt);   % sparse gpuArray

            phi_hat(:, kx, ky) = T \ rhs(:, kx, ky);
        end
    end

    %% --- DC mode (kx=1,ky=1): singular T, solve via 1-D DCT in time ---
    f1_col         = f_hat(:, 1, 1);
    f1_t           = dct_rows_gpu(f1_col');
    phi1_t         = zeros(1, nt, 'like', f_hat);
    phi1_t(2:end)  = f1_t(2:end) ./ reshape(lambda_t(2:end), 1, []);
    phi_hat(:,1,1) = idct_rows_gpu(phi1_t)';

    %% --- rho update ---
    phi_hat_curr = phi_hat(1:ntm, :, :);
    phi_hat_next = phi_hat(2:nt,  :, :);
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / problem.dt;
    adj_rho      = idct2_xy(adj_rho_hat, tw);

    x_out.rho = mu + adj_rho;

    %% --- mx / my update ---
    phi_weighted = idct2_xy(phi_vals .* phi_hat, tw);
    x_out.mx = psi_x + ops.deriv_x_at_m(phi_weighted);
    x_out.my = psi_y + ops.deriv_y_at_m(phi_weighted);
end

% ---------------------------------------------------------------------------

function x = to_gpu(x)
    if ~isa(x, 'gpuArray')
        x = gpuArray(x);
    end
end

% ---------------------------------------------------------------------------
% 2-D DCT-II along x (dim 2) then y (dim 3) — no permutes. Identical to
% proj_fokker_planck_expsemi_gpu's dct2_xy/idct2_xy, parameterized by a
% twiddle struct instead of reading problem.expsemi_proj directly.
% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f, tw)
    Nx = size(f, 2);
    xe = cat(2, f, f(:, Nx:-1:1, :));
    V  = fft(xe, [], 2);
    f_hat = real(V(:, 1:Nx, :) .* tw.tw_x) .* tw.w_x;

    Ny = size(f_hat, 3);
    ye = cat(3, f_hat, f_hat(:, :, Ny:-1:1));
    V  = fft(ye, [], 3);
    f_hat = real(V(:, :, 1:Ny) .* tw.tw_y) .* tw.w_y;
end

function f = idct2_xy(f_hat, tw)
    [m, nx_sz, Ny] = size(f_hat);
    Z  = f_hat .* tw.iw_y;
    U  = Z .* tw.itw_y;
    ye = cat(3, U, zeros(m, nx_sz, Ny, 'like', U));
    f  = real(ifft(ye, [], 3)) * (2*Ny);
    f  = f(:, :, 1:Ny);

    [m, Nx, ny_sz] = size(f);
    Z  = f .* tw.iw_x;
    U  = Z .* tw.itw_x;
    xe = cat(2, U, zeros(m, Nx, ny_sz, 'like', U));
    f  = real(ifft(xe, [], 2)) * (2*Nx);
    f  = f(:, 1:Nx, :);
end

% ---------------------------------------------------------------------------
% FFT-based orthonormal DCT-II/III along rows — GPU-compatible. Identical to
% proj_fokker_planck_expsemi_gpu. Used only for the 1-D time DCT of the DC
% spatial mode.
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
