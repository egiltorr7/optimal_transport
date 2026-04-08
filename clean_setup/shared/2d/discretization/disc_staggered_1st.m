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
    ny  = problem.ny;  nym = ny - 1;
    dt  = problem.dt;
    dx  = problem.dx;
    dy  = problem.dy;

    % % --- Time operators (act on rows via left-multiply) ---
    % % interp: (nt x ntm) maps ntm-row input to nt-row output
    % It_fwd = 0.5 * toeplitz([1 1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]);
    % % interp: (ntm x nt) maps nt-row input to ntm-row output
    % It_bwd = 0.5 * toeplitz([1 zeros(1,ntm-1)],   [1 1 zeros(1,ntm-1)]);
    % % deriv: (nt x ntm)
    % Dt_fwd = toeplitz([1 -1 zeros(1,ntm-1)],  [1 zeros(1,ntm-1)])  / dt;
    % % deriv: (ntm x nt)
    % Dt_bwd = toeplitz([-1 zeros(1,ntm-1)], [-1 1 zeros(1,ntm-1)]) / dt;
    % 
    % % --- Space operators (act on cols via right-multiply with transpose) ---
    % % interp: (nx x nxm) maps nxm-col input to nx-col output
    % Ix_fwd = 0.5 * toeplitz([1 1 zeros(1,nxm-1)], [1 zeros(1,nxm-1)]);
    % % interp: (nxm x nx) maps nx-col input to nxm-col output
    % Ix_bwd = 0.5 * toeplitz([1 zeros(1,nxm-1)],   [1 1 zeros(1,nxm-1)]);
    % % deriv: (nx x nxm)
    % Dx_fwd = toeplitz([1 -1 zeros(1,nxm-1)],  [1 zeros(1,nxm-1)])  / dx;
    % % deriv: (nxm x nx)
    % Dx_bwd = toeplitz([-1 zeros(1,nxm-1)], [-1 1 zeros(1,nxm-1)]) / dx;

    % --- Function handles (closures capture precomputed matrices) ---
    ops.interp_t_at_phi = @(in, bc0, bc1) t_fwd_interp(in, bc0, bc1);
    ops.interp_t_at_rho = @(in)           t_bwd_interp(in);
    ops.interp_x_at_phi = @(in, bc0, bc1) x_fwd_interp(in, bc0, bc1);
    ops.interp_x_at_m   = @(in)           x_bwd_interp(in);
    ops.interp_y_at_phi = @(in, bc0, bc1) y_fwd_interp(in, bc0, bc1);
    ops.interp_y_at_m   = @(in)           y_bwd_interp(in);
    ops.deriv_t_at_phi  = @(in, bc0, bc1) t_fwd_deriv(in, bc0, bc1, dt);
    ops.deriv_t_at_rho  = @(in)           t_bwd_deriv(in);
    ops.deriv_x_at_phi  = @(in, bc0, bc1) x_fwd_deriv(in, bc0, bc1, dx);
    ops.deriv_x_at_m    = @(in)           x_bwd_deriv(in);
    ops.deriv_y_at_phi  = @(in, bc0, bc1) y_fwd_deriv(in, bc0, bc1, dy);
    ops.deriv_y_at_m    = @(in)           y_bwd_deriv(in);

    % --- Adjoint operators (purely linear, no BC correction terms) ---
    % adjoint of (It_bwd *) is (* It_bwd') = (* It_fwd)  [since It_bwd' = It_fwd]
    ops.interp_t_at_rho_adj = @(in) 0.5*(in(1:end-1,:,:) + in(2:end,:,:));
    % adjoint of (* Ix_bwd') is (* Ix_bwd)  [right-multiply adjoint]
    ops.interp_x_at_m_adj   = @(in) 0.5*(in(:,1:end-1,:) + in(:,2:end,:));
    ops.interp_y_at_m_adj   = @(in) 0.5*(in(:,:,1:end-1) + in(:,:,2:end));
end

%% Helper functions 
function out = t_fwd_interp(in, bc0, bc1)

    n = size(in,1) + 1;
    out = zeros(n, size(in,2), size(in,3), 'like', in);

    % Interior
    out(2:end-1,:,:) = 0.5 * in(1:end-1,:,:);
    out(2:end-1,:,:) = out(2:end-1,:,:) + 0.5 * in(2:end,:,:);

    % Boundaries
    out(1,:,:)   = 0.5 * bc0;
    out(end,:,:) = 0.5 * bc1;

end

function out = t_bwd_interp(in)

    out = 0.5 * (in(1:end-1,:,:) + in(2:end,:,:));
end


function out = x_fwd_interp(in, bc0, bc1)
    n = size(in,2) + 1;
    out = zeros(size(in,1), n, size(in,3), 'like', in);

    % Interior
    out(:,2:end-1,:) = 0.5 * in(:,1:end-1,:);
    out(:,2:end-1,:) = out(:,2:end-1,:) + 0.5 * in(:,2:end,:);

    % Boundaries
    out(:,1,:)   = 0.5 * bc0;
    out(:,end,:) = 0.5 * bc1;
end

function out = x_bwd_interp(in)

    out = 0.5 * (in(:,1:end-1,:) + in(:,2:end,:));
end

function out = y_fwd_interp(in, bc0, bc1)
    n = size(in,3) + 1;
    out = zeros(size(in,1), size(in,2), n, 'like', in);

    % Interior
    out(:,:,2:end-1) = 0.5 * in(:,:,1:end-1);
    out(:,:,2:end-1) = out(:,:,2:end-1) + 0.5 * in(:,:,2:end);

    % Boundaries
    out(:,:,1)   = 0.5 * bc0;
    out(:,:,end) = 0.5 * bc1;
end

function out = y_bwd_interp(in)

    out = 0.5 * (in(:,:,1:end-1) + in(:,:,2:end));
end

function out = t_fwd_deriv(in, bc0, bc1, dt)

    n = size(in,1) + 1;
    out = zeros(n, size(in,2), size(in,3), 'like', in);

    % Interior
    out(2:end-1,:,:) = - in(1:end-1,:,:)/dt;
    out(2:end-1,:,:) = out(2:end-1,:,:) + in(2:end,:,:)/dt;

    % Boundaries
    out(1,:,:)   =  -bc0/dt;
    out(end,:,:) = bc1/dt;
    
end

function out = t_bwd_deriv(in)
    out = (in(1:end-1,:,:)- in(2:end,:,:))/dt;
end

function out = x_fwd_deriv(in, bc0, bc1, dx)

    n = size(in,2) + 1;
    out = zeros(size(in,1), n, size(in,3), 'like', in);

    % Interior
    out(:,2:end-1,:) = - in(:,1:end-1,:)/dx;
    out(:,2:end-1,:) = out(:,2:end-1,:) +  in(:,2:end,:)/dx;

    % Boundaries
    out(:,1,:)   = -bc0/dx;
    out(:,end,:) = bc1/dx;
end

function out = x_bwd_deriv(in)
    out = (in(:,1:end-1,:)- in(:,2:end,:))/dx;
end

function out = y_fwd_deriv(in, bc0, bc1, dy)

    n = size(in,2) + 1;
    out = zeros(size(in,1), n, size(in,3), 'like', in);

    % Interior
    out(:,:,2:end-1) = - in(:,:,1:end-1)/dy;
    out(:,:,2:end-1) = out(:,:,2:end-1) +  in(:,:,2:end)/dy;

    % Boundaries
    out(:,:,1)   = -bc0/dy;
    out(:,:,end) = bc1/dy;
end

function out = y_bwd_deriv(in)
    out = (in(:,:,1:end-1)- in(:,:,2:end))/dy;
end
