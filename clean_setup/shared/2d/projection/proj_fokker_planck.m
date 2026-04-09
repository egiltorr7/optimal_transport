function x_out = proj_fokker_planck(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK  Project onto the Fokker-Planck constraint set.
%
%   x_out = proj_fokker_planck(x_in, problem, cfg)
%
%   Enforces:  d_t rho + d_x mx + d_y my = eps * (d_xx + d_yy) rho
%   with BCs:  rho(0,.) = rho0,  rho(1,.) = rho1,  mx=my=0 at domain walls.
%
%   Uses a spectral solve (DCT) to invert the biharmonic-modified Laplacian.
%
%   Inputs:
%     x_in.rho  (ntm x nx  x ny)   point to project, density    (staggered)
%     x_in.mx   (nt  x nxm x ny)   point to project, x-momentum (staggered)
%     x_in.my   (nt  x nx  x nym)  point to project, y-momentum (staggered)
%     problem                       problem struct
%     cfg                           config struct (uses cfg.vareps)
%
%   Output:
%     x_out.rho  (ntm x nx  x ny)
%     x_out.mx   (nt  x nxm x ny)
%     x_out.my   (nt  x nx  x nym)

    ops    = problem.ops;
    rho0   = problem.rho0;
    rho1   = problem.rho1;
    nt     = problem.nt;
    nx     = problem.nx;
    ny     = problem.ny;
    vareps = cfg.vareps;

    % Zero BCs for momentum (Dirichlet no-flux walls)
    zeros_x = zeros(nt, ny);    % x-wall BCs: shape matches squeeze(out(:,1,:))
    zeros_y = zeros(nt, nx);    % y-wall BCs: shape matches squeeze(out(:,:,1))

    mu    = x_in.rho;
    psi_x = x_in.mx;
    psi_y = x_in.my;

    % Time derivative of mu at phi-locations: (nt x nx x ny)
    dmu_dt = ops.deriv_t_at_phi(mu, rho0, rho1);

    % Spatial Laplacian of mu at phi-locations: (nt x nx x ny)
    mu_phi   = ops.interp_t_at_phi(mu, rho0, rho1);
    nabla_mu = ops.deriv_x_at_phi(ops.deriv_x_at_m(mu_phi), zeros_x, zeros_x) ...
             + ops.deriv_y_at_phi(ops.deriv_y_at_m(mu_phi), zeros_y, zeros_y);

    % Divergence of (psi_x, psi_y) at phi-locations: (nt x nx x ny)
    dpsi_dx = ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x);
    dpsi_dy = ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y);

    % Spectral solve:  (-Delta + eps^2 * Delta_xy^2) phi = rhs
    rhs      = dmu_dt + dpsi_dx + dpsi_dy - vareps .* nabla_mu;
    phi_temp = spectral_solve(rhs, problem, cfg);   % (nt x nx x ny)

    % Apply corrections
    dphi_dx   = ops.deriv_x_at_m(phi_temp);   % (nt x nxm x ny)
    dphi_dy   = ops.deriv_y_at_m(phi_temp);   % (nt x nx  x nym)

    nabla_phi = ops.interp_t_at_rho( ...
        ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x) + ...
        ops.deriv_y_at_phi(dphi_dy, zeros_y, zeros_y));   % (ntm x nx x ny)

    x_out.rho = mu    + ops.deriv_t_at_rho(phi_temp) + vareps .* nabla_phi;
    x_out.mx  = psi_x + dphi_dx;
    x_out.my  = psi_y + dphi_dy;
end

% ---------------------------------------------------------------------------

function phi = spectral_solve(f, problem, cfg)
% SPECTRAL_SOLVE  Invert  (-Delta_{t,x,y} + eps^2 * Delta_{x,y}^2) phi = f  via DCT.

    vareps = cfg.vareps;

    % Reshape eigenvalues for broadcasting with f of size (nt x nx x ny)
    lt = reshape(problem.lambda_t, [], 1,  1 );   % (nt x 1  x 1 )
    lx = reshape(problem.lambda_x, 1,  [], 1 );   % (1  x nx x 1 )
    ly = reshape(problem.lambda_y, 1,  1,  []);   % (1  x 1  x ny)

    lambda_xy       = lx + ly;                               % (1 x nx x ny)
    lambda          = lt + lambda_xy + vareps^2 .* lambda_xy.^2;  % (nt x nx x ny)

    phi_hat         = mirt_dctn(f);
    phi_hat         = phi_hat ./ lambda;
    phi_hat(1,1,1)  = 0.0;
    phi             = mirt_idctn(phi_hat);
end
