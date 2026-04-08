function ops = disc_staggered_1st(problem)
% DISC_STAGGERED_1ST  1st-order staggered finite-difference operators for 1D.
%
%   ops = disc_staggered_1st(problem)
%
%   All operators are returned as function handles in the ops struct.
%   Grid dimensions:
%     rho lives at  (ntm x nx)  -- time-interior, space cell-center
%     mx  lives at  (nt  x nxm) -- all time, space-staggered
%     phi lives at  (nt  x nx)  -- auxiliary potential, all time & space
%
%   Operators:
%     interp_t_at_phi(in, bc0, bc1)  (ntm x nx)  -> (nt  x nx)
%     interp_t_at_rho(in)            (nt  x nx)  -> (ntm x nx)
%     interp_x_at_phi(in, bc0, bc1)  (nt  x nxm) -> (nt  x nx)
%     interp_x_at_m(in)              (nt  x nx)  -> (nt  x nxm)
%     deriv_t_at_phi(in, bc0, bc1)   (ntm x nx)  -> (nt  x nx)
%     deriv_t_at_rho(in)             (nt  x nx)  -> (ntm x nx)
%     deriv_x_at_phi(in, bc0, bc1)   (nt  x nxm) -> (nt  x nx)
%     deriv_x_at_m(in)               (nt  x nx)  -> (nt  x nxm)
%
%   Adjoint operators (L2 adjoints of the purely linear parts, no BC terms):
%     interp_t_at_rho_adj(in)        (ntm x nx)  -> (nt  x nx)   adjoint of interp_t_at_rho
%     interp_x_at_m_adj(in)          (nt  x nxm) -> (nt  x nx)   adjoint of interp_x_at_m

    nt  = problem.nt;  ntm = nt - 1;
    nx  = problem.nx;  nxm = nx - 1;
    dt  = problem.dt;
    dx  = problem.dx;

    % --- Time operators (act on rows via left-multiply) ---
    % interp: (nt x ntm) maps ntm-row input to nt-row output
    It_fwd = 0.5 * toeplitz([1 1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]);
    % interp: (ntm x nt) maps nt-row input to ntm-row output
    It_bwd = 0.5 * toeplitz([1 zeros(1,ntm-1)],   [1 1 zeros(1,ntm-1)]);
    % deriv: (nt x ntm)
    Dt_fwd = toeplitz([1 -1 zeros(1,ntm-1)],  [1 zeros(1,ntm-1)])  / dt;
    % deriv: (ntm x nt)
    Dt_bwd = toeplitz([-1 zeros(1,ntm-1)], [-1 1 zeros(1,ntm-1)]) / dt;

    % --- Space operators (act on cols via right-multiply with transpose) ---
    % interp: (nx x nxm) maps nxm-col input to nx-col output
    Ix_fwd = 0.5 * toeplitz([1 1 zeros(1,nxm-1)], [1 zeros(1,nxm-1)]);
    % interp: (nxm x nx) maps nx-col input to nxm-col output
    Ix_bwd = 0.5 * toeplitz([1 zeros(1,nxm-1)],   [1 1 zeros(1,nxm-1)]);
    % deriv: (nx x nxm)
    Dx_fwd = toeplitz([1 -1 zeros(1,nxm-1)],  [1 zeros(1,nxm-1)])  / dx;
    % deriv: (nxm x nx)
    Dx_bwd = toeplitz([-1 zeros(1,nxm-1)], [-1 1 zeros(1,nxm-1)]) / dx;

    % --- Function handles (closures capture precomputed matrices) ---
    ops.interp_t_at_phi = @(in, bc0, bc1) t_fwd_interp(in, bc0, bc1, It_fwd);
    ops.interp_t_at_rho = @(in)           It_bwd * in;
    ops.interp_x_at_phi = @(in, bc0, bc1) x_fwd_interp(in, bc0, bc1, Ix_fwd);
    ops.interp_x_at_m   = @(in)           in * Ix_bwd';
    ops.deriv_t_at_phi  = @(in, bc0, bc1) t_fwd_deriv(in, bc0, bc1, Dt_fwd, dt);
    ops.deriv_t_at_rho  = @(in)           Dt_bwd * in;
    ops.deriv_x_at_phi  = @(in, bc0, bc1) x_fwd_deriv(in, bc0, bc1, Dx_fwd, dx);
    ops.deriv_x_at_m    = @(in)           in * Dx_bwd';

    % --- Adjoint operators (purely linear, no BC correction terms) ---
    % adjoint of (It_bwd *) is (* It_bwd') = (* It_fwd)  [since It_bwd' = It_fwd]
    ops.interp_t_at_rho_adj = @(in) It_fwd * in;
    % adjoint of (* Ix_bwd') is (* Ix_bwd)  [right-multiply adjoint]
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
