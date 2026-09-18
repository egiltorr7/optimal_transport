function x_out = proj_fokker_planck_expsemi_gpu(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_EXPSEMI_GPU  GPU version of proj_fokker_planck_expsemi.
%
%   Drop-in replacement that uses FFT-based dct_rows/idct_rows and the batched
%   Thomas solver instead of MATLAB's dct/idct and per-mode LU loops.
%   All inputs/outputs are gpuArrays when use_gpu=true.
%
%   Requires problem.expsemi_proj set by precomp_expsemi_proj_gpu.

    ops      = problem.ops;
    rho0     = problem.rho0;    % (1 x nx) gpuArray
    rho1     = problem.rho1;    % (1 x nx) gpuArray
    nt       = problem.nt;
    ntm      = nt - 1;
    dt       = problem.dt;
    ep       = problem.expsemi_proj;
    c_vals   = ep.c_vals;    % (1 x nx) gpuArray: exp(-alpha_k)
    phi_vals = ep.phi_vals;  % (1 x nx) gpuArray: (1-c_k)/alpha_k

    mu  = x_in.rho;   % (ntm x nx) gpuArray
    psi = x_in.mx;    % (nt  x nxm) gpuArray

    zeros_x = zeros(nt, 1, 'like', mu);

    %% ETD FP residual in DCT space
    mu_prev   = [rho0; mu];
    mu_curr   = [mu;   rho1];
    S_prev    = apply_semigroup(mu_prev, c_vals);
    f_rho_hat = dct_rows((mu_curr - S_prev) / dt);          % (nt x nx)

    Dxm_hat   = dct_rows(ops.deriv_x_at_phi(psi, zeros_x, zeros_x));  % (nt x nx)

    f_hat = f_rho_hat + phi_vals .* Dxm_hat;

    % No early-exit check: gather() would force GPU->CPU sync every iteration.

    %% Solve T_k phi_k = f_hat(:,k)
    phi_hat = zeros(nt, problem.nx, 'like', mu);

    % k=1 (DC mode, lambda_x=0, c=1, phi=1): singular T_1, invert via DCT-in-t
    f1_row    = dct_rows(f_hat(:,1)');
    phi1_row  = zeros(1, nt, 'like', mu);
    phi1_row(2:end) = f1_row(2:end) ./ ep.lambda_t(2:end)';
    phi_hat(:,1) = idct_rows(phi1_row)';

    % k=2..nx: batched Thomas solve
    phi_hat(:, 2:end) = thomas_batch_solve(ep.D_mod, ep.e_vals, f_hat(:, 2:end));

    %% rho update
    phi_hat_curr = phi_hat(1:ntm, :);
    phi_hat_next = phi_hat(2:nt,  :);
    adj_rho_hat  = (c_vals .* phi_hat_next - phi_hat_curr) / dt;
    adj_rho      = idct_rows(adj_rho_hat);

    x_out.rho = mu + adj_rho;

    %% m update: D_x^T (Phi_op * phi)
    phi_weighted = idct_rows(phi_vals .* phi_hat);   % (nt x nx)
    x_out.mx = psi + ops.deriv_x_at_m(phi_weighted);
end

function S_rho = apply_semigroup(rho, c_vals)
    rho_hat = dct_rows(rho);
    S_rho   = idct_rows(rho_hat .* c_vals);
end
