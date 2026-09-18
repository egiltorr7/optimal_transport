function ops = disc_staggered_1st_3d(problem)
% DISC_STAGGERED_1ST_3D  1st-order staggered finite-difference operators for 3D.
%
%   ops = disc_staggered_1st_3d(problem)
%
%   Variable grids (4-D arrays):
%     rho  (ntm x nx  x ny  x nz )  density at interior times, cell centres
%     mx   (nt  x nxm x ny  x nz )  x-momentum, x-face staggered
%     my   (nt  x nx  x nym x nz )  y-momentum, y-face staggered
%     mz   (nt  x nx  x ny  x nzm)  z-momentum, z-face staggered
%     phi  (nt  x nx  x ny  x nz )  Lagrange multiplier, cell centres
%
%   Operators returned as function handles in the ops struct.
%
%     interp_t_at_phi(in, bc0, bc1)  (ntm x nx x ny x nz)  -> (nt x nx x ny x nz)
%     interp_t_at_rho(in)            (nt  x nx x ny x nz)  -> (ntm x nx x ny x nz)
%     interp_x_at_phi(in, bc0, bc1)  (nt x nxm x ny x nz)  -> (nt x nx x ny x nz)
%     interp_x_at_m(in)              (nt x nx x ny x nz)   -> (nt x nxm x ny x nz)
%     interp_y_at_phi(in, bc0, bc1)  (nt x nx x nym x nz)  -> (nt x nx x ny x nz)
%     interp_y_at_m(in)              (nt x nx x ny x nz)   -> (nt x nx x nym x nz)
%     interp_z_at_phi(in, bc0, bc1)  (nt x nx x ny x nzm)  -> (nt x nx x ny x nz)
%     interp_z_at_m(in)              (nt x nx x ny x nz)   -> (nt x nx x ny x nzm)
%     deriv_t_at_phi(in, bc0, bc1)   (ntm x nx x ny x nz)  -> (nt x nx x ny x nz)
%     deriv_t_at_rho(in)             (nt  x nx x ny x nz)  -> (ntm x nx x ny x nz)
%     deriv_x_at_phi(in, bc0, bc1)   (nt x nxm x ny x nz)  -> (nt x nx x ny x nz)
%     deriv_x_at_m(in)               (nt x nx x ny x nz)   -> (nt x nxm x ny x nz)
%     deriv_y_at_phi(in, bc0, bc1)   (nt x nx x nym x nz)  -> (nt x nx x ny x nz)
%     deriv_y_at_m(in)               (nt x nx x ny x nz)   -> (nt x nx x nym x nz)
%     deriv_z_at_phi(in, bc0, bc1)   (nt x nx x ny x nzm)  -> (nt x nx x ny x nz)
%     deriv_z_at_m(in)               (nt x nx x ny x nz)   -> (nt x nx x ny x nzm)

    dt = problem.dt;
    dx = problem.dx;
    dy = problem.dy;
    dz = problem.dz;

    ops.interp_t_at_phi = @(in, bc0, bc1) t_fwd_interp(in, bc0, bc1);
    ops.interp_t_at_rho = @(in)           t_bwd_interp(in);
    ops.interp_x_at_phi = @(in, bc0, bc1) x_fwd_interp(in, bc0, bc1);
    ops.interp_x_at_m   = @(in)           x_bwd_interp(in);
    ops.interp_y_at_phi = @(in, bc0, bc1) y_fwd_interp(in, bc0, bc1);
    ops.interp_y_at_m   = @(in)           y_bwd_interp(in);
    ops.interp_z_at_phi = @(in, bc0, bc1) z_fwd_interp(in, bc0, bc1);
    ops.interp_z_at_m   = @(in)           z_bwd_interp(in);

    ops.deriv_t_at_phi  = @(in, bc0, bc1) t_fwd_deriv(in, bc0, bc1, dt);
    ops.deriv_t_at_rho  = @(in)           t_bwd_deriv(in, dt);
    ops.deriv_x_at_phi  = @(in, bc0, bc1) x_fwd_deriv(in, bc0, bc1, dx);
    ops.deriv_x_at_m    = @(in)           x_bwd_deriv(in, dx);
    ops.deriv_y_at_phi  = @(in, bc0, bc1) y_fwd_deriv(in, bc0, bc1, dy);
    ops.deriv_y_at_m    = @(in)           y_bwd_deriv(in, dy);
    ops.deriv_z_at_phi  = @(in, bc0, bc1) z_fwd_deriv(in, bc0, bc1, dz);
    ops.deriv_z_at_m    = @(in)           z_bwd_deriv(in, dz);

    ops.interp_t_at_rho_adj = @(in) 0.5*(in(1:end-1,:,:,:) + in(2:end,:,:,:));
    ops.interp_x_at_m_adj   = @(in) 0.5*(in(:,1:end-1,:,:) + in(:,2:end,:,:));
    ops.interp_y_at_m_adj   = @(in) 0.5*(in(:,:,1:end-1,:) + in(:,:,2:end,:));
    ops.interp_z_at_m_adj   = @(in) 0.5*(in(:,:,:,1:end-1) + in(:,:,:,2:end));
end

%% --- Time operators (dim 1) ---

function out = t_fwd_interp(in, bc0, bc1)
    [ntm, nx, ny, nz] = size(in);
    nt  = ntm + 1;
    out = zeros(nt, nx, ny, nz, 'like', in);
    out(1:end-1,:,:,:) = 0.5 * in;
    out(2:end,:,:,:)   = out(2:end,:,:,:) + 0.5 * in;
    out(1,:,:,:)   = out(1,:,:,:)   + 0.5 * reshape(bc0, 1, nx, ny, nz);
    out(end,:,:,:) = out(end,:,:,:) + 0.5 * reshape(bc1, 1, nx, ny, nz);
end

function out = t_bwd_interp(in)
    out = 0.5 * (in(1:end-1,:,:,:) + in(2:end,:,:,:));
end

function out = t_fwd_deriv(in, bc0, bc1, dt)
    [ntm, nx, ny, nz] = size(in);
    nt  = ntm + 1;
    out = zeros(nt, nx, ny, nz, 'like', in);
    out(1:end-1,:,:,:) =  in / dt;
    out(2:end,:,:,:)   = out(2:end,:,:,:) - in / dt;
    out(1,:,:,:)   = out(1,:,:,:)   - reshape(bc0, 1, nx, ny, nz) / dt;
    out(end,:,:,:) = out(end,:,:,:) + reshape(bc1, 1, nx, ny, nz) / dt;
end

function out = t_bwd_deriv(in, dt)
    out = (in(2:end,:,:,:) - in(1:end-1,:,:,:)) / dt;
end

%% --- x operators (dim 2) ---

function out = x_fwd_interp(in, bc0, bc1)
    [nt, nxm, ny, nz] = size(in);
    nx  = nxm + 1;
    out = zeros(nt, nx, ny, nz, 'like', in);
    out(:,1:end-1,:,:) = 0.5 * in;
    out(:,2:end,:,:)   = out(:,2:end,:,:) + 0.5 * in;
    out(:,1,:,:)   = out(:,1,:,:)   + 0.5 * reshape(bc0, nt, 1, ny, nz);
    out(:,end,:,:) = out(:,end,:,:) + 0.5 * reshape(bc1, nt, 1, ny, nz);
end

function out = x_bwd_interp(in)
    out = 0.5 * (in(:,1:end-1,:,:) + in(:,2:end,:,:));
end

function out = x_fwd_deriv(in, bc0, bc1, dx)
    [nt, nxm, ny, nz] = size(in);
    nx  = nxm + 1;
    out = zeros(nt, nx, ny, nz, 'like', in);
    out(:,1:end-1,:,:) =  in / dx;
    out(:,2:end,:,:)   = out(:,2:end,:,:) - in / dx;
    out(:,1,:,:)   = out(:,1,:,:)   - reshape(bc0, nt, 1, ny, nz) / dx;
    out(:,end,:,:) = out(:,end,:,:) + reshape(bc1, nt, 1, ny, nz) / dx;
end

function out = x_bwd_deriv(in, dx)
    out = (in(:,2:end,:,:) - in(:,1:end-1,:,:)) / dx;
end

%% --- y operators (dim 3) ---

function out = y_fwd_interp(in, bc0, bc1)
    [nt, nx, nym, nz] = size(in);
    ny  = nym + 1;
    out = zeros(nt, nx, ny, nz, 'like', in);
    out(:,:,1:end-1,:) = 0.5 * in;
    out(:,:,2:end,:)   = out(:,:,2:end,:) + 0.5 * in;
    out(:,:,1,:)   = out(:,:,1,:)   + 0.5 * reshape(bc0, nt, nx, 1, nz);
    out(:,:,end,:) = out(:,:,end,:) + 0.5 * reshape(bc1, nt, nx, 1, nz);
end

function out = y_bwd_interp(in)
    out = 0.5 * (in(:,:,1:end-1,:) + in(:,:,2:end,:));
end

function out = y_fwd_deriv(in, bc0, bc1, dy)
    [nt, nx, nym, nz] = size(in);
    ny  = nym + 1;
    out = zeros(nt, nx, ny, nz, 'like', in);
    out(:,:,1:end-1,:) =  in / dy;
    out(:,:,2:end,:)   = out(:,:,2:end,:) - in / dy;
    out(:,:,1,:)   = out(:,:,1,:)   - reshape(bc0, nt, nx, 1, nz) / dy;
    out(:,:,end,:) = out(:,:,end,:) + reshape(bc1, nt, nx, 1, nz) / dy;
end

function out = y_bwd_deriv(in, dy)
    out = (in(:,:,2:end,:) - in(:,:,1:end-1,:)) / dy;
end

%% --- z operators (dim 4) ---

function out = z_fwd_interp(in, bc0, bc1)
    [nt, nx, ny, nzm] = size(in);
    nz  = nzm + 1;
    out = zeros(nt, nx, ny, nz, 'like', in);
    out(:,:,:,1:end-1) = 0.5 * in;
    out(:,:,:,2:end)   = out(:,:,:,2:end) + 0.5 * in;
    out(:,:,:,1)   = out(:,:,:,1)   + 0.5 * reshape(bc0, nt, nx, ny, 1);
    out(:,:,:,end) = out(:,:,:,end) + 0.5 * reshape(bc1, nt, nx, ny, 1);
end

function out = z_bwd_interp(in)
    out = 0.5 * (in(:,:,:,1:end-1) + in(:,:,:,2:end));
end

function out = z_fwd_deriv(in, bc0, bc1, dz)
    [nt, nx, ny, nzm] = size(in);
    nz  = nzm + 1;
    out = zeros(nt, nx, ny, nz, 'like', in);
    out(:,:,:,1:end-1) =  in / dz;
    out(:,:,:,2:end)   = out(:,:,:,2:end) - in / dz;
    out(:,:,:,1)   = out(:,:,:,1)   - reshape(bc0, nt, nx, ny, 1) / dz;
    out(:,:,:,end) = out(:,:,:,end) + reshape(bc1, nt, nx, ny, 1) / dz;
end

function out = z_bwd_deriv(in, dz)
    out = (in(:,:,:,2:end) - in(:,:,:,1:end-1)) / dz;
end
