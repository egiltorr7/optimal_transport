function x_out = proj_fokker_planck_expsemi(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI  2D FP projection via ETD (exact heat semigroup).
%
%   x_out = proj_fokker_planck_expsemi(x_in, problem, cfg)
%
%   Drop-in replacement for proj_fokker_planck_banded in 2D.  Uses the ETD
%   (exponential time differencing) discretisation of the Fokker–Planck
%   constraint instead of the Crank–Nicolson Padé(1,1) approximation, giving
%   exact treatment of the heat semigroup for each spatial DCT mode:
%
%       (rho_{j+1} - c_{kxky} * rho_j) / dt
%           + phi_{kxky} * (D_x psi_x + D_y psi_y)_{j+1/2}  =  0
%
%   where c = exp(-eps*lxy*dt) and phi = (1-c)/(eps*lxy*dt), lxy = lx + ly.
%
%   Algorithm (identical structure to proj_fokker_planck_banded):
%     1. Form ETD FP residual  f_hat  in DCT-xy space.
%     2. Batched Thomas solve  T_{kxky} phi_hat = f_hat  (precomputed diagonals).
%     3. DC mode (kx=1,ky=1): singular T, handled via DCT in time.
%     4. rho update:  adj_rho_hat = (c * phi_next - phi_curr) / dt  ->  IDCT.
%     5. mx/my update: delta_m = D_{x,y}^T[ IDCT(phi_vals .* phi_hat) ].
%
%   Requires:
%     problem.expsemi_proj   precomputed by precomp_expsemi_proj(problem, vareps)
%
%   Grid (same as all other 2D projections):
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
    % Boundary-augmented density: (nt x nx x ny)
    mu_prev = cat(1, rho0_3d, mu);   % [rho0; rho_1; ...; rho_{ntm}]
    mu_curr = cat(1, mu, rho1_3d);   % [rho_1; ...; rho_{ntm}; rho1]

    % Apply heat semigroup in DCT-xy space: S * rho = IDCT2(c .* DCT2(rho))
    % Equivalently, compute the residual entirely in DCT space.
    mu_prev_hat = dct2_xy(mu_prev, nt, nx, ny);   % (nt x nx x ny)
    mu_curr_hat = dct2_xy(mu_curr, nt, nx, ny);
    f_rho_hat   = (mu_curr_hat - c_vals .* mu_prev_hat) / problem.dt;

    % Divergence of staggered momentum -> physical space -> DCT-xy
    div_psi = ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
            + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y);   % (nt x nx x ny)
    div_psi_hat = dct2_xy(div_psi, nt, nx, ny);

    f_hat = f_rho_hat + phi_vals .* div_psi_hat;

    if norm(f_hat(:)) * sqrt(problem.dt * problem.dx * problem.dy) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- Batched Thomas solve for all (kx, ky) modes ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;   % DC mode handled separately below

    phi_hat = thomas_solve(ep.lower_all, ep.main_all, ep.upper_all, rhs, nt, nx, ny);

    %% --- DC spatial mode (kx=1, ky=1): lxy=0, c=1, phi=1, T=M0 singular ---
    % Invert M0 via DCT in time (same as proj_fokker_planck_banded).
    f1_t           = dct(f_hat(:, 1, 1));
    phi1_t         = zeros(nt, 1, 'like', f_hat);
    phi1_t(2:end)  = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:,1,1) = idct(phi1_t);

    %% --- rho update: adj_rho_hat = (c * phi_next - phi_curr) / dt ---
    phi_hat_curr = phi_hat(1:ntm, :, :);     % (ntm x nx x ny)
    phi_hat_next = phi_hat(2:nt,  :, :);     % (ntm x nx x ny)
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / problem.dt;
    adj_rho      = idct2_xy(adj_rho_hat, ntm, nx, ny);

    x_out.rho = mu + adj_rho;

    %% --- mx/my update: delta_m = D_{x,y}^T [ IDCT(phi_vals .* phi_hat) ] ---
    phi_weighted = idct2_xy(phi_vals .* phi_hat, nt, nx, ny);   % (nt x nx x ny)
    x_out.mx = psi_x + ops.deriv_x_at_m(phi_weighted);
    x_out.my = psi_y + ops.deriv_y_at_m(phi_weighted);
end

% ---------------------------------------------------------------------------
% Batched Thomas (TDMA) algorithm — identical to proj_fokker_planck_banded.
% Solves T_{kx,ky} * phi(:,kx,ky) = f_hat(:,kx,ky) for every mode at once.
% Compatible with gpuArray inputs (sequential over nt, vectorised over nx*ny).
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
% 2D DCT utilities — identical to proj_fokker_planck_banded.
% DCT-II along spatial dims 2 (x) and 3 (y) of an (m x nx x ny) array.
% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f, m, nx, ny)
    % DCT along x (dim 2)
    f2 = permute(f, [2, 1, 3]);          % (nx x m x ny)
    f2 = reshape(f2, nx, m * ny);
    f2 = dct(f2);
    f2 = reshape(f2, nx, m, ny);
    f2 = permute(f2, [2, 1, 3]);         % (m x nx x ny)

    % DCT along y (dim 3)
    f3 = permute(f2, [3, 1, 2]);         % (ny x m x nx)
    f3 = reshape(f3, ny, m * nx);
    f3 = dct(f3);
    f3 = reshape(f3, ny, m, nx);
    f_hat = permute(f3, [2, 3, 1]);      % (m x nx x ny)
end

function f = idct2_xy(f_hat, m, nx, ny)
    % IDCT along y (dim 3)
    f3 = permute(f_hat, [3, 1, 2]);      % (ny x m x nx)
    f3 = reshape(f3, ny, m * nx);
    f3 = idct(f3);
    f3 = reshape(f3, ny, m, nx);
    f2 = permute(f3, [2, 3, 1]);         % (m x nx x ny)

    % IDCT along x (dim 2)
    f2 = permute(f2, [2, 1, 3]);         % (nx x m x ny)
    f2 = reshape(f2, nx, m * ny);
    f2 = idct(f2);
    f2 = reshape(f2, nx, m, ny);
    f  = permute(f2, [2, 1, 3]);         % (m x nx x ny)
end
