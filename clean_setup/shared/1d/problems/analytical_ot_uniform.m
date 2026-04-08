function [rho_ana, mx_ana] = analytical_ot_uniform(problem)
% ANALYTICAL_OT_UNIFORM  Exact OT (eps=0) solution for uniform[0.2,0.4] -> uniform[0.6,0.8].
%
%   [rho_ana, mx_ana] = analytical_ot_uniform(problem)
%
%   Both marginals have the same support width w = 0.2, so the Brenier map
%   is a pure translation:
%
%     T(x) = x + 0.4   (shift = c - a = 0.6 - 0.2)
%
%   The displacement interpolation gives:
%
%     rho(t,x) = Uniform[a + shift*t, b + shift*t]
%              = (1/w) * 1_{x in [0.2 + 0.4*t, 0.4 + 0.4*t]}
%
%     m(t,x)   = rho(t,x) * shift   (constant velocity = 0.4 in the support)
%
%   The kinetic energy equals the squared Wasserstein-2 distance:
%
%     KE = W_2^2(rho0, rho1) = shift^2 = 0.16
%
%   This is the exact OT solution; for eps>0 (SB) use numerical reference.
%
%   Outputs (same grid convention as analytical_gaussian):
%     rho_ana  (ntm x nx)   density   at times k*dt,        positions (i-0.5)*dx
%     mx_ana   (nt  x nxm)  momentum  at times (k-0.5)*dt,  positions j*dx

    a = 0.2;   b = 0.4;
    c = 0.6;   % d = 0.8;
    w     = b - a;       % support width = 0.2
    shift = c - a;       % translation  = 0.4
    v     = shift;       % constant velocity

    nt  = problem.nt;   ntm = nt - 1;   dt = problem.dt;
    nx  = problem.nx;   nxm = nx - 1;   dx = problem.dx;

    % --- rho: times k*dt (k=1..ntm), positions (i-0.5)*dx ---
    t_rho = (1:ntm)' * dt;
    x_rho = ((1:nx) - 0.5) * dx;

    rho_ana = zeros(ntm, nx);
    for k = 1:ntm
        lo = a + shift * t_rho(k);
        hi = lo + w;
        row = double(x_rho >= lo & x_rho <= hi) / w;
        s   = sum(row);
        if s > 0
            rho_ana(k, :) = row / s;
        end
    end

    % --- mx: times (k-0.5)*dt (k=1..nt), positions j*dx ---
    t_mx = ((1:nt)' - 0.5) * dt;
    x_mx = (1:nxm) * dx;

    mx_ana = zeros(nt, nxm);
    for k = 1:nt
        lo  = a + shift * t_mx(k);
        hi  = lo + w;
        row = double(x_mx >= lo & x_mx <= hi) / w;
        s   = sum(row);
        if s > 0
            row = row / s;
        end
        mx_ana(k, :) = row * v;
    end
end
