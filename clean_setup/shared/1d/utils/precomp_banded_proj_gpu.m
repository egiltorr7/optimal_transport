function bp = precomp_banded_proj_gpu(problem, vareps)
% PRECOMP_BANDED_PROJ_GPU  GPU-optimised precomputation for proj_fokker_planck_banded_gpu.
%
%   bp = precomp_banded_proj_gpu(problem, vareps)
%
%   The per-mode system T_k = M0 + lx_k*diag(M1d) + lx_k^2*M2 has:
%
%     Diagonal:
%       d(1)       = 1/dt^2 + lx_k*(1+eps/dt) + lx_k^2*eps^2/4
%       d(2..nt-1) = 2/dt^2 + lx_k           + lx_k^2*eps^2/2
%       d(nt)      = 1/dt^2 + lx_k*(1-eps/dt) + lx_k^2*eps^2/4
%
%     Off-diagonal (constant for all nt-1 positions per mode k):
%       e_k = -1/dt^2 + lx_k^2*eps^2/4
%
%   This constant-off-diagonal structure allows a batched Thomas precomputation
%   vectorised over all nx-1 spatial modes simultaneously.
%
%   Output fields (all gpuArray):
%     bp.D_mod    (nt x nx-1)  Thomas-modified diagonal for modes k=2..nx
%     bp.e_vals   (1  x nx-1)  constant off-diagonal per mode
%     bp.lambda_t (nt x 1)     time-DCT eigenvalues for DC mode

    nt  = problem.nt;
    nx  = problem.nx;
    dt  = problem.dt;
    lx  = problem.lambda_x(2:end);   % (1 x nx-1), modes k=2..nx

    % Build diagonal matrix D (nt x nx-1)
    D       = repmat(2/dt^2 + lx + lx.^2 * vareps^2/2, nt, 1);
    D(1,:)  = 1/dt^2 + lx*(1 + vareps/dt) + lx.^2 * vareps^2/4;
    D(nt,:) = 1/dt^2 + lx*(1 - vareps/dt) + lx.^2 * vareps^2/4;

    % Constant off-diagonal per mode
    e = -1/dt^2 + lx.^2 * vareps^2/4;   % (1 x nx-1)

    bp.D_mod    = thomas_batch_precomp(gpuArray(D), gpuArray(e));
    bp.e_vals   = gpuArray(e);
    bp.lambda_t = gpuArray(problem.lambda_t);
end
