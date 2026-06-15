function ep = precomp_expsemi_proj_gpu(problem, vareps)
% PRECOMP_EXPSEMI_PROJ_GPU  GPU-optimised precomputation for proj_fokker_planck_expsemi_gpu.
%
%   ep = precomp_expsemi_proj_gpu(problem, vareps)
%
%   Instead of per-mode LU factorisation (as in precomp_expsemi_proj), this
%   assembles all nt x nt tridiagonal systems (one per x-mode j=2..nx) into a
%   batched (nt x nx-1) representation and runs the Thomas forward sweep in a
%   single vectorised pass over modes.
%
%   Output fields (all gpuArray):
%     ep.c_vals   (1 x nx)    semigroup coefficients exp(-vareps*lambda_x*dt)
%     ep.D_mod    (nt x nx-1) Thomas-modified diagonal for modes j=2..nx
%     ep.e_vals   (1 x nx-1)  constant off-diagonal for modes j=2..nx
%     ep.lambda_t (nt x 1)    time-DCT eigenvalues (for DC mode gauge fix)

    nt  = problem.nt;
    nx  = problem.nx;
    dt  = problem.dt;

    c_all  = exp(-vareps * problem.lambda_x * dt);   % (1 x nx) CPU
    ep.c_vals   = gpuArray(c_all);
    ep.lambda_t = gpuArray(problem.lambda_t);

    % Modes j=2..nx only (DC mode j=1 is handled separately)
    c  = c_all(2:end);                     % (1 x nx-1)
    lx = problem.lambda_x(2:end);          % (1 x nx-1)

    % Diagonal of T_j: d(1)=1/dt^2+lx, d(2..nt-1)=(1+c^2)/dt^2+lx, d(nt)=c^2/dt^2+lx
    D      = repmat((1 + c.^2)/dt^2 + lx, nt, 1);   % (nt x nx-1)
    D(1,:) = 1/dt^2 + lx;
    D(nt,:)= c.^2/dt^2 + lx;

    % Constant off-diagonal per mode
    e = -c / dt^2;   % (1 x nx-1)

    ep.D_mod  = thomas_batch_precomp(gpuArray(D), gpuArray(e));
    ep.e_vals = gpuArray(e);
end
