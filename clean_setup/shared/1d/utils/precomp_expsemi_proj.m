function ep = precomp_expsemi_proj(problem, vareps)
% PRECOMP_EXPSEMI_PROJ  Precompute LU factors for proj_fokker_planck_expsemi.
%
%   ep = precomp_expsemi_proj(problem, vareps)
%
%   Uses the exact heat semigroup S_eps^dt = exp(eps*Delta*dt) instead of
%   the Crank-Nicolson (Padé 1/1) or backward-Euler approximations.  Under
%   DCT in x, the semigroup is diagonal with coefficients
%
%       c_j = exp(-eps * lambda_x(j) * dt)
%
%   and the FP normal equations decouple per mode j into an (nt x nt)
%   symmetric tridiagonal system T_j phi_j = f_j with
%
%     diagonal:    [ 1/dt^2 + lx,
%                    (1+c_j^2)/dt^2 + lx,  (k=2..nt-1)
%                    c_j^2/dt^2 + lx ]
%     off-diagonal: -c_j/dt^2   (constant)
%
%   Limits:
%     c_j->0 (large eps*lx*dt): diagonal -> 1/dt^2 + lx everywhere except
%             last row which -> lx; off-diagonal -> 0. Time steps decouple.
%     c_j->1 (small eps or lx->0): recovers the pure-time Laplacian M0
%             (same as the DC mode), as expected.
%
%   j=1 (DC mode, lambda_x=0, c=1): T_1 is the singular time Laplacian M0,
%   handled at projection time via DCT-in-t with gauge fix (same as banded).
%
%   Output fields:
%     ep.c_vals      (1 x nx)   semigroup coefficients per x-mode
%     ep.Tj_L{j}    cell(nx,1) LU factors (L)  for j=2..nx
%     ep.Tj_U{j}    cell(nx,1) LU factors (U)  for j=2..nx
%     ep.Tj_P{j}    cell(nx,1) LU factors (P)  for j=2..nx

    nt  = problem.nt;
    nx  = problem.nx;
    dt  = problem.dt;

    ep.c_vals = exp(-vareps * problem.lambda_x * dt);   % (1 x nx)

    ep.Tj_L = cell(nx, 1);
    ep.Tj_U = cell(nx, 1);
    ep.Tj_P = cell(nx, 1);

    for j = 2:nx
        lx = problem.lambda_x(j);
        cj = ep.c_vals(j);

        d      = ((1 + cj^2)/dt^2 + lx) * ones(nt, 1);
        d(1)   = 1/dt^2 + lx;
        d(nt)  = cj^2/dt^2 + lx;

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
