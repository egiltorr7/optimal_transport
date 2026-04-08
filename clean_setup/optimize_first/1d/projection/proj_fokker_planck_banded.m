function x_out = proj_fokker_planck_banded(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_BANDED  Project onto the FP constraint via DCT-x + tridiagonal-t.
%
%   x_out = proj_fokker_planck_banded(x_in, problem, cfg)
%
%   Same interface as proj_fokker_planck but solves AA*phi = f exactly by:
%     1. Computing the FP residual  f = d_t mu + d_x psi - eps * d_xx mu
%     2. DCT in x -> nx decoupled nt-vectors f_hat(:,k)
%     3. k=1 (lambda_x=0, M0 singular): invert via DCT in t using lambda_t
%     4. k=2..nx: tridiagonal solve using precomputed LU (from problem.banded_proj)
%     5. IDCT in x -> phi in physical space
%     6. Apply A* to correct: rho += d_t^T phi + eps*d_xx^T phi,  mx += d_x^T phi
%
%   Requires:
%     problem.banded_proj   precomputed by  precomp_banded_proj(problem, vareps)
%     problem.lambda_t      precomputed by  setup_problem  (nt x 1)
%
%   Inputs / outputs match proj_fokker_planck (drop-in replacement).

    ops    = problem.ops;
    rho0   = problem.rho0;
    rho1   = problem.rho1;
    nt     = problem.nt;
    vareps = cfg.vareps;
    bp     = problem.banded_proj;

    zeros_x = zeros(nt, 1);

    mu  = x_in.rho;   % (ntm x nx)
    psi = x_in.mx;    % (nt  x nxm)

    %% --- FP residual  f = d_t mu + d_x psi - eps * d_xx mu  (nt x nx) ---
    laplacian_mu = ops.deriv_x_at_phi( ...
                       ops.deriv_x_at_m( ...
                           ops.interp_t_at_phi(mu, rho0, rho1)), ...
                       zeros_x, zeros_x);

    f = ops.deriv_t_at_phi(mu, rho0, rho1) ...
      + ops.deriv_x_at_phi(psi, zeros_x, zeros_x) ...
      - vareps * laplacian_mu;

    if norm(f(:)) * sqrt(problem.dt * problem.dx) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- DCT in x (each row of f independently) ---
    f_hat = dct(f')';    % (nt x nx)

    phi_hat = zeros(nt, problem.nx);

    % k=1: DC mode, lambda_x=0 => T_1 = M0 is singular.
    % Invert via DCT in t; (k=1, l=1) is the gauge freedom -> zero.
    f1_t              = dct(f_hat(:, 1));
    phi1_t            = zeros(nt, 1);
    phi1_t(2:end)     = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:, 1)     = idct(phi1_t);

    % k=2..nx: SPD tridiagonal solve with precomputed LU
    for k = 2:problem.nx
        phi_hat(:, k) = bp.Tk_U{k} \ (bp.Tk_L{k} \ (bp.Tk_P{k} * f_hat(:, k)));
    end

    %% --- IDCT in x -> phi in physical space (nt x nx) ---
    phi = idct(phi_hat')';

    %% --- Apply A* to get corrections ---
    dphi_dx    = ops.deriv_x_at_m(phi);
    nablax_phi = ops.interp_t_at_rho( ...
                     ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x));

    x_out.rho = mu  + ops.deriv_t_at_rho(phi) + vareps * nablax_phi;
    x_out.mx  = psi + dphi_dx;
end
