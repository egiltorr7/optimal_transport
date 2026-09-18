function ep = precomp_expsemi_proj(problem, vareps)
% PRECOMP_EXPSEMI_PROJ  Precompute LU factors for proj_fokker_planck_expsemi.
%
%   ep = precomp_expsemi_proj(problem, vareps)
%
%   Uses the exact-semigroup / ETD (exponential time differencing) scheme.
%   For each DCT-x mode j, the FP constraint is integrated exactly over one
%   time step using the integrating factor exp(eps*lambda_x(j)*dt):
%
%       (rho_{j+1} - c_j * rho_j) / dt  +  phi_j * D_x m_{j+1/2}  =  0
%
%   where:
%       alpha_j = eps * lambda_x(j) * dt
%       c_j     = exp(-alpha_j)
%       phi_j   = (1 - c_j) / alpha_j     (phi(0) = 1 by L'Hopital)
%
%   phi_j weights the momentum forcing by the exact integrated propagator
%   (vs. phi_j=1 in the simple expsemi scheme).  For small alpha, phi->1
%   and the scheme reduces to the original expsemi.  For large alpha,
%   phi->0 and the constraint decouples in time (diffusion-dominated).
%
%   The normal equations decouple per mode j into T_j phi_j = f_j with:
%
%     diagonal:    [ 1/dt^2 + phi_j^2 * lx,
%                    (1+c_j^2)/dt^2 + phi_j^2 * lx,  (k=2..nt-1)
%                    c_j^2/dt^2 + phi_j^2 * lx ]
%     off-diagonal: -c_j/dt^2   (constant)
%
%   j=1 (DC mode, lambda_x=0, c=1, phi=1): T_1 is the singular time
%   Laplacian M0, handled at projection time via DCT-in-t (unchanged).
%
%   Output fields:
%     ep.c_vals      (1 x nx)   semigroup coefficients c_j = exp(-alpha_j)
%     ep.phi_vals    (1 x nx)   ETD weights phi_j = (1-c_j)/alpha_j
%     ep.Tj_L{j}    cell(nx,1) LU factors (L)  for j=2..nx
%     ep.Tj_U{j}    cell(nx,1) LU factors (U)  for j=2..nx
%     ep.Tj_P{j}    cell(nx,1) LU factors (P)  for j=2..nx

    nt  = problem.nt;
    nx  = problem.nx;
    dt  = problem.dt;

    alpha_vals  = vareps * problem.lambda_x * dt;   % (1 x nx)
    c_vals      = exp(-alpha_vals);                 % (1 x nx)

    % phi(alpha) = (1 - exp(-alpha))/alpha, limit phi(0) = 1
    phi_vals      = ones(1, nx);
    nz            = alpha_vals > 1e-14;
    phi_vals(nz)  = (1 - c_vals(nz)) ./ alpha_vals(nz);

    ep.c_vals   = c_vals;
    ep.phi_vals = phi_vals;

    ep.Tj_L = cell(nx, 1);
    ep.Tj_U = cell(nx, 1);
    ep.Tj_P = cell(nx, 1);

    for j = 2:nx
        lx  = problem.lambda_x(j);
        cj  = c_vals(j);
        phj = phi_vals(j);

        d      = ((1 + cj^2)/dt^2 + phj^2 * lx) * ones(nt, 1);
        d(1)   = 1/dt^2 + phj^2 * lx;
        d(nt)  = cj^2/dt^2 + phj^2 * lx;

        od = (-cj / dt^2) * ones(nt - 1, 1);

        Tj = sparse(1:nt,   1:nt,   d,  nt, nt) + ...
             sparse(2:nt,   1:nt-1, od, nt, nt) + ...
             sparse(1:nt-1, 2:nt,   od, nt, nt);

        [L, U, P] = lu(Tj);
        ep.Tj_L{j} = L;
        ep.Tj_U{j} = U;
        ep.Tj_P{j} = P;
    end
end
