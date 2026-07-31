function x_out = proj_fokker_planck_expsemi_builtin_dct_gpu(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_BUILTIN_DCT_GPU  ETD FP projection, GPU, using
%   MATLAB's built-in dct/idct instead of the hand-rolled FFT+twiddle
%   dct2_xy/idct2_xy in proj_fokker_planck_expsemi_gpu.
%
%   Identical algorithm/structure to proj_fokker_planck_expsemi_gpu -- same
%   batched (M x nt) Thomas solve (thomas_solve, unchanged), same DC-mode
%   handling -- with ONLY the DCT implementation swapped, so a timing/
%   correctness comparison between the two isolates exactly what the custom
%   DCT buys you (if anything) over MATLAB's own gpuArray-enabled dct/idct.
%
%   dct/idct support gpuArray with a "type-2 DCT only" restriction (which is
%   exactly the type used here), and dct(x,n,dim) takes a DIM argument like
%   fft(X,[],dim) -- so it can operate along dims 2/3 of a 3D array directly,
%   with no permute() needed, same permute-free property the hand-rolled
%   version was specifically built for.
%
%   Does NOT need ep.tw_x/w_x/iw_x/itw_x/tw_y/w_y/iw_y/itw_y (those are
%   specific to dct2_xy's hand-rolled twiddle math) -- only needs the same
%   Thomas-solve fields proj_fokker_planck_expsemi_gpu needs: c_vals,
%   phi_vals, lower_T, main_T, upper_T, main_T_mod.
%
%   Requires Signal Processing Toolbox (dct/idct) with gpuArray support.
%
%   Grid:
%     x_in.rho  (ntm x nx  x ny)
%     x_in.mx   (nt  x nxm x ny)
%     x_in.my   (nt  x nx  x nym)

    ops  = problem.ops;
    rho0 = problem.rho0;
    rho1 = problem.rho1;
    nt   = problem.nt;   ntm = nt - 1;
    nx   = problem.nx;
    ny   = problem.ny;
    ep   = problem.expsemi_proj;
    c_vals   = ep.c_vals;
    phi_vals = ep.phi_vals;

    mu    = x_in.rho;
    psi_x = x_in.mx;
    psi_y = x_in.my;

    rho0_3d = reshape(rho0, 1, nx, ny);
    rho1_3d = reshape(rho1, 1, nx, ny);
    zeros_x = zeros(nt, ny, 'like', mu);
    zeros_y = zeros(nt, nx, 'like', mu);

    %% --- ETD FP residual in DCT-xy space ---
    mu_prev = cat(1, rho0_3d, mu);
    mu_curr = cat(1, mu, rho1_3d);

    mu_prev_hat = dct2_xy(mu_prev);
    mu_curr_hat = dct2_xy(mu_curr);
    f_rho_hat   = (mu_curr_hat - c_vals .* mu_prev_hat) / problem.dt;

    div_psi = ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
            + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y);
    div_psi_hat = dct2_xy(div_psi);

    f_hat = f_rho_hat + phi_vals .* div_psi_hat;

    if norm(f_hat(:)) * sqrt(problem.dt * problem.dx * problem.dy) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- Batched Thomas solve (kx,ky) != (1,1) -- unchanged from expsemi_gpu ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;

    phi_hat = thomas_solve(ep, rhs, nt, nx, ny);

    %% --- DC mode (kx=1,ky=1): singular T, solve via 1-D DCT in time ---
    f1_col         = f_hat(:, 1, 1);
    f1_t           = dct(f1_col, [], 1);
    phi1_t         = zeros(nt, 1, 'like', f_hat);
    phi1_t(2:end)  = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:,1,1) = idct(phi1_t, [], 1);

    %% --- rho update ---
    phi_hat_curr = phi_hat(1:ntm, :, :);
    phi_hat_next = phi_hat(2:nt,  :, :);
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / problem.dt;
    adj_rho      = idct2_xy(adj_rho_hat);

    x_out.rho = mu + adj_rho;

    %% --- mx / my update ---
    phi_weighted = idct2_xy(phi_vals .* phi_hat);
    x_out.mx = psi_x + ops.deriv_x_at_m(phi_weighted);
    x_out.my = psi_y + ops.deriv_y_at_m(phi_weighted);
end

% ---------------------------------------------------------------------------
% 2-D DCT-II via built-in dct/idct with an explicit DIM argument -- no
% permute() needed, same shape convention as dct2_xy in expsemi_gpu.
% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f)
    f_hat = dct(f, [], 2);   % DCT along x (dim 2)
    f_hat = dct(f_hat, [], 3);   % DCT along y (dim 3)
end

function f = idct2_xy(f_hat)
    f = idct(f_hat, [], 3);   % IDCT along y (dim 3)
    f = idct(f, [], 2);       % IDCT along x (dim 2)
end

% ---------------------------------------------------------------------------
% Batched Thomas (TDMA) -- byte-for-byte identical to proj_fokker_planck_expsemi_gpu.
% ---------------------------------------------------------------------------

function phi_hat = thomas_solve(ep, f_hat, nt, nx, ny)
    M = nx * ny;
    d = reshape(permute(f_hat, [2, 3, 1]), M, nt);

    for i = 2:nt
        w       = ep.lower_T(:, i-1) ./ ep.main_T_mod(:, i-1);
        d(:, i) = d(:, i) - w .* d(:, i-1);
    end

    phi_r        = zeros(M, nt, 'like', f_hat);
    phi_r(:, nt) = d(:, nt) ./ ep.main_T_mod(:, nt);
    for i = nt-1:-1:1
        phi_r(:, i) = (d(:, i) - ep.upper_T(:, i) .* phi_r(:, i+1)) ./ ep.main_T_mod(:, i);
    end

    phi_hat = permute(reshape(phi_r, nx, ny, nt), [3, 1, 2]);
end
