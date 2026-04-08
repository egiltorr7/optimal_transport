function [rho_ana, mx_ana] = analytical_ot_bimodal_to_bimodal(problem)
% ANALYTICAL_OT_BIMODAL_TO_BIMODAL  Exact OT solution for inward bimodal transport.
%
%   [rho_ana, mx_ana] = analytical_ot_bimodal_to_bimodal(problem)
%
%   Problem:
%     rho0 = N(0.20, 0.04^2) + N(0.80, 0.04^2)
%     rho1 = N(0.35, 0.04^2) + N(0.65, 0.04^2)
%
%   Optimal transport plan (eps=0, non-crossing):
%     left  peak translates 0.20 -> 0.35  (v_L = +0.15)
%     right peak translates 0.80 -> 0.65  (v_R = -0.15)
%
%   McCann displacement interpolation:
%     mu_L(t) = 0.20 + 0.15*t
%     mu_R(t) = 0.80 - 0.15*t
%     rho(t,x) = N(x; mu_L(t), 0.04^2) + N(x; mu_R(t), 0.04^2)  (normalised)
%     m(t,x)   = v_L*N(x; mu_L(t), 0.04^2) + v_R*N(x; mu_R(t), 0.04^2)  (normalised)
%
%   The velocity field satisfies the continuity equation exactly:
%     d_t rho + d_x m = 0    (eps=0, no diffusion)
%   because each Gaussian component independently satisfies it with
%   constant velocity v_k.
%
%   KE = W_2^2 = (1/2)*0.15^2 + (1/2)*0.15^2 = 0.0225
%
%   Outputs (same grid convention as analytical_gaussian):
%     rho_ana  (ntm x nx)   density   at times k*dt,        positions (i-0.5)*dx
%     mx_ana   (nt  x nxm)  momentum  at times (k-0.5)*dt,  positions j*dx

    mu_L0 = 0.20;   mu_L1 = 0.35;   v_L = mu_L1 - mu_L0;   % +0.15
    mu_R0 = 0.80;   mu_R1 = 0.65;   v_R = mu_R1 - mu_R0;   % -0.15
    sigma = 0.04;

    nt  = problem.nt;   ntm = nt - 1;   dt = problem.dt;
    nx  = problem.nx;   nxm = nx - 1;   dx = problem.dx;

    Normal = @(x, mu) exp(-0.5*((x-mu)/sigma).^2) / (sqrt(2*pi)*sigma);

    % --- rho: times k*dt (k=1..ntm), positions (i-0.5)*dx ---
    t_rho = (1:ntm)' * dt;
    x_rho = ((1:nx) - 0.5) * dx;

    rho_ana = zeros(ntm, nx);
    for k = 1:ntm
        muL = mu_L0 + v_L * t_rho(k);
        muR = mu_R0 + v_R * t_rho(k);
        row = Normal(x_rho, muL) + Normal(x_rho, muR);
        rho_ana(k, :) = row / sum(row);
    end

    % --- mx: times (k-0.5)*dt (k=1..nt), positions j*dx ---
    t_mx = ((1:nt)' - 0.5) * dt;
    x_mx = (1:nxm) * dx;

    mx_ana = zeros(nt, nxm);
    for k = 1:nt
        muL = mu_L0 + v_L * t_mx(k);
        muR = mu_R0 + v_R * t_mx(k);
        rho_raw = Normal(x_mx, muL) + Normal(x_mx, muR);
        m_raw   = v_L * Normal(x_mx, muL) + v_R * Normal(x_mx, muR);
        S = sum(rho_raw);
        if S > 0
            mx_ana(k, :) = m_raw / S;   % same normalisation factor as rho
        end
    end
end
