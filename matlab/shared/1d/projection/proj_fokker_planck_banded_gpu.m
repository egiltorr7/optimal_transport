function x_out = proj_fokker_planck_banded_gpu(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_BANDED_GPU  GPU version of proj_fokker_planck_banded.
%
%   Drop-in replacement using FFT-based dct_rows/idct_rows and the batched
%   Thomas solver instead of MATLAB's dct/idct and per-mode LU loops.
%   All inputs/outputs are gpuArrays when use_gpu=true.
%
%   Requires problem.banded_proj set by precomp_banded_proj_gpu.

    ops    = problem.ops;
    rho0   = problem.rho0;    % (1 x nx) gpuArray
    rho1   = problem.rho1;    % (1 x nx) gpuArray
    nt     = problem.nt;
    ntm    = nt - 1;
    vareps = cfg.vareps;
    bp     = problem.banded_proj;

    mu  = x_in.rho;   % (ntm x nx) gpuArray
    psi = x_in.mx;    % (nt  x nxm) gpuArray

    zeros_x = zeros(nt, 1, 'like', mu);

    %% FP residual  f = d_t mu + d_x psi - eps * d_xx mu  (nt x nx)
    laplacian_mu = ops.deriv_x_at_phi( ...
                       ops.deriv_x_at_m( ...
                           ops.interp_t_at_phi(mu, rho0, rho1)), ...
                       zeros_x, zeros_x);

    f = ops.deriv_t_at_phi(mu, rho0, rho1) ...
      + ops.deriv_x_at_phi(psi, zeros_x, zeros_x) ...
      - vareps * laplacian_mu;

    % No early-exit check: gather() would force GPU->CPU sync every iteration.

    %% DCT in x
    f_hat   = dct_rows(f);                            % (nt x nx)
    phi_hat = zeros(nt, problem.nx, 'like', mu);

    % k=1 (DC mode, lambda_x=0): T_1 = M0 is singular, invert via DCT-in-t
    f1_row    = dct_rows(f_hat(:,1)');                % (1 x nt)
    phi1_row  = zeros(1, nt, 'like', mu);
    phi1_row(2:end) = f1_row(2:end) ./ bp.lambda_t(2:end)';
    phi_hat(:,1) = idct_rows(phi1_row)';              % (nt x 1)

    % k=2..nx: batched Thomas solve
    phi_hat(:, 2:end) = thomas_batch_solve(bp.D_mod, bp.e_vals, f_hat(:, 2:end));

    %% IDCT in x -> phi in physical space
    phi = idct_rows(phi_hat);                         % (nt x nx)

    %% Apply A* to get corrections
    dphi_dx    = ops.deriv_x_at_m(phi);
    nablax_phi = ops.interp_t_at_rho( ...
                     ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x));

    x_out.rho = mu  + ops.deriv_t_at_rho(phi) + vareps * nablax_phi;
    x_out.mx  = psi + dphi_dx;
end
