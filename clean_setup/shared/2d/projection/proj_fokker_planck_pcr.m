function x_out = proj_fokker_planck_pcr(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_PCR  FP projection via Parallel Cyclic Reduction (PCR).
%
%   x_out = proj_fokker_planck_pcr(x_in, problem, cfg)
%
%   Drop-in replacement for proj_fokker_planck_banded / proj_fokker_planck_spike2.
%   Identical mathematical result; different tridiagonal solver.
%
%   Why PCR?
%     Thomas / Spike use nt sequential kernel launches, each operating on
%     (nx x ny) elements.  For nt=64, nx=ny=64: 64 launches of 4096 elements
%     → GPU is ~10% utilised (kernel launch overhead dominates).
%     PCR uses log2(nt) = 6 sequential passes, each operating on the full
%     (nt x nx x ny) array → 6 launches of 262144 elements → GPU fully busy.
%
%   Algorithm (Parallel Cyclic Reduction):
%     Level s = 1..log2(nt), stride = 2^(s-1):
%       For all rows i simultaneously (vectorised over nt x nx x ny):
%         alpha_i = -a_i / b_{i-stride}   (0 if i <= stride)
%         gamma_i = -c_i / b_{i+stride}   (0 if i > nt-stride)
%         a_new_i =  alpha_i * a_{i-stride}
%         b_new_i =  b_i + alpha_i * c_{i-stride} + gamma_i * a_{i+stride}
%         c_new_i =  gamma_i * c_{i+stride}
%         d_new_i =  d_i + alpha_i * d_{i-stride} + gamma_i * d_{i+stride}
%     After log2(nt) levels: each equation involves only x_i.
%     Solution: x_i = d_i / b_i.
%
%   Stable for diagonally dominant T (our case for all lxy > 0).
%   Requires nt to be a power of 2.
%   Requires problem.banded_proj from precomp_banded_proj_pcr.
%   Compatible with gpuArray inputs.

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

    %% --- PCR solve for all non-DC modes (DC mode RHS zeroed, result overwritten) ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;

    phi_hat = pcr_solve(bp, rhs, nt, nx, ny);

    %% --- (kx=1,ky=1) DC mode: lxy=0 -> singular T -> DCT-t ---
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

function phi_hat = pcr_solve(bp, f_hat, nt, nx, ny)
% PCR_SOLVE  Parallel Cyclic Reduction for all (kx,ky) modes simultaneously.
%
%   log2(nt) sequential passes, each a full (nt x nx x ny) parallel operation.
%   At each level s (stride = 2^(s-1)):
%     lo = 1:nt-stride,  hi = stride+1:nt
%
%     alpha(hi) = a(hi) / b(lo)          (positive: subtract below)
%     gam(lo)   = c(lo) / b(hi)          (positive: subtract below)
%
%     a_new(hi) = -alpha .* a(lo)
%     c_new(lo) = -gam   .* c(hi)
%     b_new(hi) -= alpha .* c(lo)        (b_new starts as copy of b)
%     b_new(lo) -= gam   .* a(hi)
%     d_new(hi) -= alpha .* d(lo)        (d_new starts as copy of d)
%     d_new(lo) -= gam   .* d(hi)        (reads original d, not d_new)
%
%   Note on overlap rows (stride+1:nt-stride): b_new and d_new receive
%   contributions from both the hi and lo updates — this is correct because
%   the lo-update reads from b (original) and d (original) respectively.

    a = bp.a_all;     % (nt x nx x ny), a(1,:,:) = 0
    b = bp.main_all;  % (nt x nx x ny)
    c = bp.c_all;     % (nt x nx x ny), c(nt,:,:) = 0
    d = f_hat;        % (nt x nx x ny)

    for s = 1:bp.n_levels
        stride = 2^(s-1);
        lo = 1:nt-stride;    % rows with an upper neighbour
        hi = stride+1:nt;    % rows with a lower neighbour

        % Elimination multipliers
        alpha = a(hi,:,:) ./ b(lo,:,:);   % (nt-stride x nx x ny)
        gam   = c(lo,:,:) ./ b(hi,:,:);   % (nt-stride x nx x ny)

        % New off-diagonals (zeros elsewhere, from initialisation)
        a_new = zeros(nt, nx, ny, 'like', d);
        a_new(hi,:,:) = -alpha .* a(lo,:,:);

        c_new = zeros(nt, nx, ny, 'like', d);
        c_new(lo,:,:) = -gam .* c(hi,:,:);

        % New main diagonal: b_new = b, then add both contributions.
        % Overlap rows (hi ∩ lo = stride+1:nt-stride) get both updates;
        % the second indexed assignment reads b_new which already has the
        % first update — giving b_i + alpha_i*c_{i-s} + gamma_i*a_{i+s}. ✓
        b_new = b;
        b_new(hi,:,:) = b_new(hi,:,:) - alpha .* c(lo,:,:);
        b_new(lo,:,:) = b_new(lo,:,:) - gam   .* a(hi,:,:);

        % New RHS: d_new = d, subtract contributions.
        % IMPORTANT: both subtractions read from d (original), not d_new,
        % so overlap rows correctly accumulate both alpha and gamma terms. ✓
        d_new = d;
        d_new(hi,:,:) = d_new(hi,:,:) - alpha .* d(lo,:,:);
        d_new(lo,:,:) = d_new(lo,:,:) - gam   .* d(hi,:,:);

        a = a_new;  b = b_new;  c = c_new;  d = d_new;
    end

    phi_hat = d ./ b;
end

% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f, nt, nx, ny)
% DCT-II along dim 2 (x) and dim 3 (y) — 3 permutes (minimum).
    f_hat = permute(f, [2, 1, 3]);
    f_hat = reshape(f_hat, nx, nt*ny);
    f_hat = dct(f_hat);
    f_hat = reshape(f_hat, nx, nt, ny);
    f_hat = permute(f_hat, [3, 2, 1]);   % merged permute
    f_hat = reshape(f_hat, ny, nt*nx);
    f_hat = dct(f_hat);
    f_hat = reshape(f_hat, ny, nt, nx);
    f_hat = permute(f_hat, [2, 3, 1]);
end

function f = idct2_xy(f_hat, nt, nx, ny)
% IDCT-II along dim 2 (x) and dim 3 (y) — 3 permutes (minimum).
    f = permute(f_hat, [3, 1, 2]);
    f = reshape(f, ny, nt*nx);
    f = idct(f);
    f = reshape(f, ny, nt, nx);
    f = permute(f, [3, 2, 1]);   % merged permute
    f = reshape(f, nx, nt*ny);
    f = idct(f);
    f = reshape(f, nx, nt, ny);
    f = permute(f, [2, 1, 3]);
end
