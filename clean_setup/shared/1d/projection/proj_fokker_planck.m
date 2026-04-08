function x_out = proj_fokker_planck(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK  Project onto the Fokker-Planck constraint set.
%
%   x_out = proj_fokker_planck(x_in, problem, cfg)
%
%   Enforces:  d_t rho + d_x m = eps * d_xx rho
%   with BCs:  rho(0,.) = rho0,  rho(1,.) = rho1,  m = 0 at x=0,1.
%
%   Uses a spectral solve (DCT) to invert the biharmonic-modified Laplacian.
%
%   Inputs:
%     x_in.rho  (ntm x nx)  point to project, density   (staggered)
%     x_in.mx   (nt  x nxm) point to project, momentum  (staggered)
%     problem               problem struct
%     cfg                   config struct (uses cfg.vareps)
%
%   Output:
%     x_out.rho  (ntm x nx)
%     x_out.mx   (nt  x nxm)

    ops    = problem.ops;
    rho0   = problem.rho0;
    rho1   = problem.rho1;
    nt     = problem.nt;
    vareps = cfg.vareps;

    zeros_x      = zeros(nt, 1);
    dirichlet_bc = zeros(nt, 1);

    mu  = x_in.rho;
    psi = x_in.mx;

    % Time derivative of mu at phi-locations: (nt x nx)
    dmu_dt = ops.deriv_t_at_phi(mu, rho0, rho1);

    % Spatial Laplacian of mu at phi-locations: (nt x nx)
    nablax_mu = ops.interp_t_at_phi(mu, rho0, rho1);
    nablax_mu = ops.deriv_x_at_phi(ops.deriv_x_at_m(nablax_mu), zeros_x, zeros_x);

    % Divergence of psi at phi-locations: (nt x nx)
    dpsi_dx = ops.deriv_x_at_phi(psi, dirichlet_bc, dirichlet_bc);

    % Spectral solve:  (-Delta + eps^2 * Delta^2) phi = rhs
    rhs      = dmu_dt + dpsi_dx - vareps .* nablax_mu;
    phi_temp = spectral_solve(rhs, problem, cfg);  % (nt x nx)

    % Apply corrections
    dphi_dx    = ops.deriv_x_at_m(phi_temp);                                         % (nt x nxm)
    nablax_phi = ops.interp_t_at_rho(ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x)); % (ntm x nx)

    x_out.rho = mu  + ops.deriv_t_at_rho(phi_temp) + vareps .* nablax_phi;
    x_out.mx  = psi + dphi_dx;
end

% ---------------------------------------------------------------------------

function phi = spectral_solve(f, problem, cfg)
% SPECTRAL_SOLVE  Invert  (-Delta + eps^2 * Delta^2) phi = f  via DCT.

    lambda_x = problem.lambda_x;
    lambda_t = problem.lambda_t;
    vareps   = cfg.vareps;

    lambda       = lambda_x + lambda_t + (vareps^2) .* lambda_x.^2;
    phi_hat      = mirt_dctn(f);
    phi_hat      = phi_hat ./ lambda;
    phi_hat(1,1) = 0.0;
    phi          = mirt_idctn(phi_hat);
end
