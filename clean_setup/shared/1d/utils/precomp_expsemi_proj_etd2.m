function ep = precomp_expsemi_proj_etd2(problem, vareps)
% PRECOMP_EXPSEMI_PROJ_ETD2  Precompute LU factors for proj_fokker_planck_expsemi_etd2.
%
%   ep = precomp_expsemi_proj_etd2(problem, vareps)
%
%   ETD2: D_x m is approximated as linear in time over each interval
%   [t_j, t_{j+1}] and integrated exactly against exp(eps*lambda_x(k)*s):
%
%       (rho_{j+1} - c_k*rho_j)/dt  +  phi1_k*(D_x m)^j  +  phi2_k*(D_x m)^{j+1}  =  0
%
%   where beta_k = eps*lambda_x(k)*dt, c_k = exp(-beta_k), and:
%
%       phi1_k = (1 - (1+beta_k)*c_k) / beta_k^2   [weight on interval start]
%       phi2_k = (beta_k - 1 + c_k)  / beta_k^2   [weight on interval end  ]
%
%   Both limits to 1/2 as beta_k -> 0 (trapezoid rule); for large beta_k,
%   phi1 -> 0 and phi2 -> 1/beta_k (correct endpoint emphasis, fixing
%   ETD1's midpoint bias at large eps).
%
%   The normal-equation matrix T_k per DCT-x mode k is still symmetric
%   tridiagonal (same structure as ETD1) with modified entries:
%
%     diagonal:    [ 1/dt^2               + (phi1^2+phi2^2)*lx,    (j=1)
%                    (1+c^2)/dt^2         + (phi1^2+phi2^2)*lx,    (j=2..nt-1)
%                    c^2/dt^2             + (phi1^2+phi2^2)*lx ]   (j=nt)
%     off-diagonal: -c/dt^2 + phi1*phi2*lx   (constant)
%
%   k=1 (DC mode, lambda_x=0, c=1): T_1 is the singular time Laplacian,
%   handled at projection time via DCT-in-t (identical to ETD1).
%
%   Output fields:
%     ep.c_vals      (1 x nx)   semigroup coefficients c_k
%     ep.phi1_vals   (1 x nx)   ETD2 start-of-interval weights
%     ep.phi2_vals   (1 x nx)   ETD2 end-of-interval weights
%     ep.Tj_L{k}    cell(nx,1) LU factor L for k=2..nx
%     ep.Tj_U{k}    cell(nx,1) LU factor U for k=2..nx
%     ep.Tj_P{k}    cell(nx,1) LU factor P for k=2..nx

    nt = problem.nt;
    nx = problem.nx;
    dt = problem.dt;

    alpha_vals = vareps * problem.lambda_x * dt;   % (1 x nx)
    c_vals     = exp(-alpha_vals);                 % (1 x nx)

    % phi1(b) = (1 - (1+b)*exp(-b)) / b^2,  limit 1/2 as b->0
    % phi2(b) = (b - 1 + exp(-b))   / b^2,  limit 1/2 as b->0
    % Use limits for small b to avoid catastrophic cancellation.
    phi1_vals = 0.5 * ones(1, nx);
    phi2_vals = 0.5 * ones(1, nx);
    nz = alpha_vals > 1e-4;
    b  = alpha_vals(nz);
    ck = c_vals(nz);
    phi1_vals(nz) = (1 - (1 + b) .* ck) ./ b.^2;
    phi2_vals(nz) = (b - 1 + ck) ./ b.^2;

    ep.c_vals    = c_vals;
    ep.phi1_vals = phi1_vals;
    ep.phi2_vals = phi2_vals;

    ep.Tj_L = cell(nx, 1);
    ep.Tj_U = cell(nx, 1);
    ep.Tj_P = cell(nx, 1);

    for k = 2:nx
        lx  = problem.lambda_x(k);
        ck_k = c_vals(k);
        p1  = phi1_vals(k);
        p2  = phi2_vals(k);

        d     = ((1 + ck_k^2)/dt^2 + (p1^2 + p2^2)*lx) * ones(nt, 1);
        d(1)  =  1/dt^2            + (p1^2 + p2^2)*lx;
        d(nt) =  ck_k^2/dt^2      + (p1^2 + p2^2)*lx;

        od = (-ck_k/dt^2 + p1*p2*lx) * ones(nt-1, 1);

        Tk = sparse(1:nt,   1:nt,   d,  nt, nt) + ...
             sparse(2:nt,   1:nt-1, od, nt, nt) + ...
             sparse(1:nt-1, 2:nt,   od, nt, nt);

        [L, U, P] = lu(Tk);
        ep.Tj_L{k} = L;
        ep.Tj_U{k} = U;
        ep.Tj_P{k} = P;
    end
end
