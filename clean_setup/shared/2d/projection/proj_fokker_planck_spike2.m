function x_out = proj_fokker_planck_spike2(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_SPIKE2  FP projection via Spike p=2 tridiagonal solver.
%
%   x_out = proj_fokker_planck_spike2(x_in, problem, cfg)
%
%   Drop-in replacement for proj_fokker_planck_banded.  Identical interface
%   and identical output when Thomas is stable (eps/dt <= 1).  When Thomas
%   is unstable (eps/dt > 1, M1d(nt) < 0), this solver remains correct
%   because it isolates the problematic last row into a separate scalar block.
%
%   Algorithm — Spike p=2:
%     Partition the (nt x nt) tridiagonal T into two blocks:
%       Block 1: T1 = T[1:nt-1, 1:nt-1]
%         M1d entries 1..nt-1 are all >= 1, so T1 is well-conditioned.
%         Solved by Thomas using precomputed pivots (spike_pivots).
%       Block 2: T[nt,nt]   (scalar per mode)
%         T(nt,nt) = M0(nt,nt) + lxy*M1d(nt) + lxy^2*M2(nt) > 0 for all lxy.
%         (Discriminant of this quadratic in lxy is negative: proved analytically.)
%
%     Let z = T1^{-1} * f[1:nt-1]  (Thomas on Block 1)
%         v = T1^{-1} * (up(nt-1)*e_{nt-1})  (precomputed spike vector)
%
%     Schur complement for x(nt):
%       S   = T(nt,nt) - T(nt,nt-1) * v(nt-1)        (1 x nx x ny)
%       x2  = (f(nt) - T(nt,nt-1) * z(nt-1)) / S
%
%     Correction for Block 1:
%       x1  = z - x2 * v
%
%   Requires problem.banded_proj from precomp_banded_proj_spike2.
%   All operations vectorised over (nx x ny) modes.
%   Compatible with gpuArray inputs (spike_pivots and spike_v must be on GPU).

    ops    = problem.ops;
    rho0   = problem.rho0;
    rho1   = problem.rho1;
    nt     = problem.nt;
    nx     = problem.nx;
    ny     = problem.ny;
    vareps = cfg.vareps;
    bp     = problem.banded_proj;

    mu    = x_in.rho;
    psi_x = x_in.mx;
    psi_y = x_in.my;

    zeros_x = zeros(nt, ny, 'like', mu);
    zeros_y = zeros(nt, nx, 'like', mu);

    %% --- FP residual  f = d_t mu + d_x psi_x + d_y psi_y - eps*Delta mu ---
    mu_phi   = ops.interp_t_at_phi(mu, rho0, rho1);
    nabla_mu = ops.deriv_x_at_phi(ops.deriv_x_at_m(mu_phi), zeros_x, zeros_x) ...
             + ops.deriv_y_at_phi(ops.deriv_y_at_m(mu_phi), zeros_y, zeros_y);

    f = ops.deriv_t_at_phi(mu, rho0, rho1) ...
      + ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
      + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y) ...
      - vareps * nabla_mu;

    if norm(f(:)) * sqrt(problem.dt * problem.dx * problem.dy) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- 2D DCT in x and y ---
    f_hat = dct2_xy(f, nt, nx, ny);   % (nt x nx x ny)

    %% --- Spike p=2 solve for all non-DC modes ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;   % DC mode handled separately below

    phi_hat = spike2_solve(bp, rhs, nt, nx, ny);

    %% --- (kx=1,ky=1) DC mode: lxy=0 -> singular T -> invert via DCT-t ---
    f1_t           = dct(f_hat(:, 1, 1));
    phi1_t         = zeros(nt, 1, 'like', f_hat);
    phi1_t(2:end)  = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:,1,1) = idct(phi1_t);

    %% --- 2D IDCT back to physical space ---
    phi = idct2_xy(phi_hat, nt, nx, ny);   % (nt x nx x ny)

    %% --- Apply A* corrections ---
    dphi_dx   = ops.deriv_x_at_m(phi);
    dphi_dy   = ops.deriv_y_at_m(phi);

    nabla_phi = ops.interp_t_at_rho( ...
        ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x) + ...
        ops.deriv_y_at_phi(dphi_dy, zeros_y, zeros_y));

    x_out.rho = mu    + ops.deriv_t_at_rho(phi) + vareps .* nabla_phi;
    x_out.mx  = psi_x + dphi_dx;
    x_out.my  = psi_y + dphi_dy;
end

% ---------------------------------------------------------------------------

function phi_hat = spike2_solve(bp, f_hat, nt, nx, ny)
% SPIKE2_SOLVE  Spike p=2 batched tridiagonal solve for all (kx,ky) modes.
%
%   Block 1 (rows 1..nt-1): Thomas with precomputed pivots bp.spike_pivots.
%   Block 2 (row nt):       Schur complement scalar solve.
%   Correction:             x1 = z - x2 * bp.spike_v
%
%   All sweeps are element-wise over (nx x ny) — fully GPU-vectorised.
%   Uses bp.thomas_mults and bp.spike_pivots_inv (precomputed) to replace
%   per-step divisions with multiplies, reducing GPU temporaries per step.

    ntm = nt - 1;

    up1   = bp.upper_all(1:nt-2, :, :);   % upper diag of T1        (nt-2 x nx x ny)
    w     = bp.thomas_mults;               % lo1(i-1) / b(i-1)       (nt-2 x nx x ny)
    binv  = bp.spike_pivots_inv;           % 1 / spike_pivots         (nt-1 x nx x ny)

    % --- Step 1: Thomas on T1 — RHS forward sweep ---
    d = f_hat(1:ntm, :, :);
    for i = 2:ntm
        d(i,:,:) = d(i,:,:) - w(i-1,:,:) .* d(i-1,:,:);
    end

    % Back-substitution: z = T1^{-1} * f[1:nt-1]
    z = zeros(ntm, nx, ny, 'like', f_hat);
    z(ntm,:,:) = d(ntm,:,:) .* binv(ntm,:,:);
    for i = ntm-1:-1:1
        z(i,:,:) = (d(i,:,:) - up1(i,:,:) .* z(i+1,:,:)) .* binv(i,:,:);
    end

    % --- Step 2: Schur complement for x(nt) ---
    lo_c  = bp.lower_all(ntm, :, :);   % T(nt, nt-1)    (1 x nx x ny)
    ma_nt = bp.main_all(nt,   :, :);   % T(nt, nt)      (1 x nx x ny)
    v     = bp.spike_v;                 % spike vector   (nt-1 x nx x ny)

    S  = ma_nt - lo_c .* v(ntm,:,:);
    x2 = (f_hat(nt,:,:) - lo_c .* z(ntm,:,:)) ./ S;   % (1 x nx x ny)

    % --- Step 3: Correct Block 1 ---
    x1 = z - x2 .* v;   % (nt-1 x nx x ny)

    phi_hat = cat(1, x1, x2);   % (nt x nx x ny)
end

% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f, nt, nx, ny)
% DCT-II along dim 2 (x) and dim 3 (y) of a (nt x nx x ny) array.
% Uses a single accumulator variable so only one large temporary is live
% at a time, reducing GPU memory fragmentation across repeated calls.
    f_hat = permute(f, [2, 1, 3]);       % (nx x nt x ny)
    f_hat = reshape(f_hat, nx, nt*ny);   % (nx x nt*ny)
    f_hat = dct(f_hat);
    f_hat = reshape(f_hat, nx, nt, ny);  % (nx x nt x ny)
    f_hat = permute(f_hat, [2, 1, 3]);   % (nt x nx x ny)
    f_hat = permute(f_hat, [3, 1, 2]);   % (ny x nt x nx)
    f_hat = reshape(f_hat, ny, nt*nx);   % (ny x nt*nx)
    f_hat = dct(f_hat);
    f_hat = reshape(f_hat, ny, nt, nx);  % (ny x nt x nx)
    f_hat = permute(f_hat, [2, 3, 1]);   % (nt x nx x ny)
end

function f = idct2_xy(f_hat, nt, nx, ny)
% IDCT-II along dim 2 (x) and dim 3 (y) of a (nt x nx x ny) array.
    f = permute(f_hat, [3, 1, 2]);   % (ny x nt x nx)
    f = reshape(f, ny, nt*nx);
    f = idct(f);
    f = reshape(f, ny, nt, nx);      % (ny x nt x nx)
    f = permute(f, [2, 3, 1]);       % (nt x nx x ny)
    f = permute(f, [2, 1, 3]);       % (nx x nt x ny)
    f = reshape(f, nx, nt*ny);
    f = idct(f);
    f = reshape(f, nx, nt, ny);      % (nx x nt x ny)
    f = permute(f, [2, 1, 3]);       % (nt x nx x ny)
end
