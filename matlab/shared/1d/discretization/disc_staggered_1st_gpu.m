function ops = disc_staggered_1st_gpu(problem)
% DISC_STAGGERED_1ST_GPU  GPU version of disc_staggered_1st.
%
%   Identical interface and behaviour; all precomputed matrices are wrapped
%   in gpuArray so closures execute matrix multiplications on the GPU via
%   cuBLAS without any per-call data transfer.

    nt  = problem.nt;  ntm = nt - 1;
    nx  = problem.nx;  nxm = nx - 1;
    dt  = problem.dt;
    dx  = problem.dx;

    It_fwd = gpuArray(0.5 * toeplitz([1 1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]));
    It_bwd = gpuArray(0.5 * toeplitz([1 zeros(1,ntm-1)],   [1 1 zeros(1,ntm-1)]));
    Dt_fwd = gpuArray(toeplitz([1 -1 zeros(1,ntm-1)],  [1 zeros(1,ntm-1)])  / dt);
    Dt_bwd = gpuArray(toeplitz([-1 zeros(1,ntm-1)], [-1 1 zeros(1,ntm-1)]) / dt);

    Ix_fwd = gpuArray(0.5 * toeplitz([1 1 zeros(1,nxm-1)], [1 zeros(1,nxm-1)]));
    Ix_bwd = gpuArray(0.5 * toeplitz([1 zeros(1,nxm-1)],   [1 1 zeros(1,nxm-1)]));
    Dx_fwd = gpuArray(toeplitz([1 -1 zeros(1,nxm-1)],  [1 zeros(1,nxm-1)])  / dx);
    Dx_bwd = gpuArray(toeplitz([-1 zeros(1,nxm-1)], [-1 1 zeros(1,nxm-1)]) / dx);

    ops.interp_t_at_phi = @(in, bc0, bc1) t_fwd_interp(in, bc0, bc1, It_fwd);
    ops.interp_t_at_rho = @(in)           It_bwd * in;
    ops.interp_x_at_phi = @(in, bc0, bc1) x_fwd_interp(in, bc0, bc1, Ix_fwd);
    ops.interp_x_at_m   = @(in)           in * Ix_bwd';
    ops.deriv_t_at_phi  = @(in, bc0, bc1) t_fwd_deriv(in, bc0, bc1, Dt_fwd, dt);
    ops.deriv_t_at_rho  = @(in)           Dt_bwd * in;
    ops.deriv_x_at_phi  = @(in, bc0, bc1) x_fwd_deriv(in, bc0, bc1, Dx_fwd, dx);
    ops.deriv_x_at_m    = @(in)           in * Dx_bwd';

    ops.interp_t_at_rho_adj = @(in) It_fwd * in;
    ops.interp_x_at_m_adj   = @(in) in * Ix_bwd;
end

function out = t_fwd_interp(in, bc0, bc1, M)
    out = M * in;
    out(1,:)   = out(1,:)   + 0.5 * bc0;
    out(end,:) = out(end,:) + 0.5 * bc1;
end

function out = x_fwd_interp(in, bc0, bc1, M)
    out = in * M';
    out(:,1)   = out(:,1)   + 0.5 * bc0;
    out(:,end) = out(:,end) + 0.5 * bc1;
end

function out = t_fwd_deriv(in, bc0, bc1, M, dt)
    out = M * in;
    out(1,:)   = out(1,:)   - bc0 / dt;
    out(end,:) = out(end,:) + bc1 / dt;
end

function out = x_fwd_deriv(in, bc0, bc1, M, dx)
    out = in * M';
    out(:,1)   = out(:,1)   - bc0 / dx;
    out(:,end) = out(:,end) + bc1 / dx;
end
