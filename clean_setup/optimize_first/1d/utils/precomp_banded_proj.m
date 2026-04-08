function bp = precomp_banded_proj(problem, vareps)
% PRECOMP_BANDED_PROJ  Precompute LU factors for proj_fokker_planck_banded.
%
%   bp = precomp_banded_proj(problem, vareps)
%
%   Call once per (grid, vareps) pair and store the result in
%   problem.banded_proj before running the ADMM loop.
%
%   The FP operator AA* decouples under DCT in x into nx independent
%   nt x nt tridiagonal systems:
%
%     T_k = M0 + lambda_x(k)*diag(M1d) + lambda_x(k)^2 * M2,  k=1..nx
%
%   where:
%     M0  = -Dt_phi * Dt_rho          (tridiagonal, = time neg-Laplacian)
%     M1d = diag(1, ..., 1) with      (diagonal, boundary correction from eps)
%           M1d(1)  = 1 + eps/dt
%           M1d(nt) = 1 - eps/dt
%     M2  = eps^2 * It_phi * It_rho   (tridiagonal)
%
%   k=1 (lambda_x=0, T_1=M0) is singular and handled at projection time
%   via a 1D DCT in t using problem.lambda_t.
%   k=2..nx are SPD; their LU factors are stored here.
%
%   Output fields:
%     bp.Tk_L, bp.Tk_U, bp.Tk_P   cell(1,nx)  LU factors for k=2..nx

    nt  = problem.nt;   ntm = nt - 1;
    nx  = problem.nx;
    dt  = problem.dt;
    dx  = problem.dx;

    % Time operators (nt x ntm)
    It_phi = 0.5 * toeplitz([1 1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]);
    Dt_phi =       toeplitz([1 -1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]) / dt;

    % Time back-operators (ntm x nt)
    It_rho = 0.5 * toeplitz([1 zeros(1,ntm-1)], [1 1 zeros(1,ntm-1)]);
    Dt_rho =       toeplitz([-1 zeros(1,ntm-1)], [-1 1 zeros(1,ntm-1)]) / dt;

    % T_k building blocks (all nt x nt)
    M0_bnd = -Dt_phi * Dt_rho;                  % positive semi-definite
    M1d    = ones(nt, 1);
    M1d(1) = 1 + vareps / dt;
    M1d(nt)= 1 - vareps / dt;
    M2_bnd = vareps^2 * (It_phi * It_rho);

    % Spatial DCT-II (Neumann) eigenvalues
    lambda_x = (2 - 2*cos(pi * dx * (0:nx-1))) / dx^2;   % 1 x nx

    % LU-factorize T_k for k=2..nx
    Tk_L = cell(1, nx);
    Tk_U = cell(1, nx);
    Tk_P = cell(1, nx);
    for k = 2:nx
        lxk = lambda_x(k);
        Tk  = sparse(M0_bnd + diag(lxk * M1d) + lxk^2 * M2_bnd);
        [Tk_L{k}, Tk_U{k}, Tk_P{k}] = lu(Tk);
    end

    bp.Tk_L = Tk_L;
    bp.Tk_U = Tk_U;
    bp.Tk_P = Tk_P;
end
