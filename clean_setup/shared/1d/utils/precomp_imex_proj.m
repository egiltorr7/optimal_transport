function ip = precomp_imex_proj(problem, vareps)
% PRECOMP_IMEX_PROJ  Precompute LU factors for the IMEX FP projection.
%
%   ip = precomp_imex_proj(problem, vareps)
%
%   The IMEX discretisation of the FP equation uses backward Euler for the
%   diffusion term (eps * d_xx rho at the LATER rho-grid time) instead of the
%   Crank-Nicolson average used in the standard banded projection.
%
%   After DCT in x, the projection normal equations decouple per x-mode j
%   into an (nt x nt) symmetric tridiagonal system T_j phi_j = r_j.
%
%   T_j structure (j = 2,...,nx):
%     H_j = 1/dt + eps * lambda_x(j)            (IMEX Helmholtz eigenvalue)
%
%     diagonal:     [ H_j^2 + lx,  H_j^2 + (1/dt)^2 + lx,  ...,  (1/dt)^2 + lx ]
%                     k=1             k=2,...,ntm                    k=nt
%     off-diagonal: -H_j / dt   (constant, all adjacent pairs)
%
%   j=1 (DC mode, lambda_x=0): T_1 is singular; handled by gauge fix in
%   proj_fokker_planck_imex (phi_hat(:,1) = 0).
%
%   Inputs:
%     problem    problem struct (needs nt, nx, dt, lambda_x)
%     vareps     regularisation parameter eps
%
%   Output:
%     ip         struct with fields:
%                  ip.Tj_L{j}, ip.Tj_U{j}, ip.Tj_P{j}  -- LU factors, j=2,...,nx

    nt  = problem.nt;
    nx  = problem.nx;
    dt  = problem.dt;

    ip.Tj_L = cell(nx, 1);
    ip.Tj_U = cell(nx, 1);
    ip.Tj_P = cell(nx, 1);

    for j = 2:nx
        lx = problem.lambda_x(j);
        Hj = 1/dt + vareps * lx;

        % Main diagonal
        d      = (Hj^2 + (1/dt)^2 + lx) * ones(nt, 1);
        d(1)   = Hj^2 + lx;           % first row: no rho_{k-1} term (BC)
        d(nt)  = (1/dt)^2 + lx;       % last row:  no H rho_k term (rho_nt = rho1, BC)

        % Off-diagonal (constant)
        od = (-Hj / dt) * ones(nt - 1, 1);

        % Build sparse symmetric tridiagonal and LU-factor
        Tj = sparse(1:nt,   1:nt,   d,  nt, nt) + ...
             sparse(2:nt,   1:nt-1, od, nt, nt) + ...
             sparse(1:nt-1, 2:nt,   od, nt, nt);

        [L, U, P] = lu(Tj);
        ip.Tj_L{j} = L;
        ip.Tj_U{j} = U;
        ip.Tj_P{j} = P;
    end
end
