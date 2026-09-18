function ep = precomp_expsemi_proj_gpu(problem, vareps)
% PRECOMP_EXPSEMI_PROJ_GPU  GPU-optimised precomputation for proj_fokker_planck_expsemi_gpu.
%
%   ep = precomp_expsemi_proj_gpu(problem, vareps)
%
%   ETD (exponential time differencing) version: see precomp_expsemi_proj for
%   the mathematical details.  The T_j diagonal uses phi_j^2 * lx instead of
%   lx, where phi_j = (1-c_j)/alpha_j is the ETD weight.
%
%   Output fields (all gpuArray):
%     ep.c_vals   (1 x nx)    semigroup coefficients c_j = exp(-alpha_j)
%     ep.phi_vals (1 x nx)    ETD weights phi_j = (1-c_j)/alpha_j
%     ep.D_mod    (nt x nx-1) Thomas-modified diagonal for modes j=2..nx
%     ep.e_vals   (1 x nx-1)  constant off-diagonal for modes j=2..nx
%     ep.lambda_t (nt x 1)    time-DCT eigenvalues (for DC mode gauge fix)

    nt  = problem.nt;
    nx  = problem.nx;
    dt  = problem.dt;

    alpha_all = vareps * problem.lambda_x * dt;   % (1 x nx) CPU
    c_all     = exp(-alpha_all);

    phi_all      = ones(1, nx);
    nz           = alpha_all > 1e-14;
    phi_all(nz)  = (1 - c_all(nz)) ./ alpha_all(nz);

    ep.c_vals   = gpuArray(c_all);
    ep.phi_vals = gpuArray(phi_all);
    ep.lambda_t = gpuArray(problem.lambda_t);

    % Modes j=2..nx only (DC mode j=1 is handled separately)
    c   = c_all(2:end);                    % (1 x nx-1)
    phi = phi_all(2:end);                  % (1 x nx-1)
    lx  = problem.lambda_x(2:end);        % (1 x nx-1)

    % Diagonal of T_j with ETD phi^2 * lx weight
    D      = repmat((1 + c.^2)/dt^2 + phi.^2 .* lx, nt, 1);   % (nt x nx-1)
    D(1,:) = 1/dt^2 + phi.^2 .* lx;
    D(nt,:)= c.^2/dt^2 + phi.^2 .* lx;

    % Constant off-diagonal per mode
    e = -c / dt^2;   % (1 x nx-1)

    ep.D_mod  = thomas_batch_precomp(gpuArray(D), gpuArray(e));
    ep.e_vals = gpuArray(e);
end
