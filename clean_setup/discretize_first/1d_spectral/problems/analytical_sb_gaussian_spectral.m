function [rho_ana, mx_ana] = analytical_sb_gaussian_spectral(problem, vareps)
% ANALYTICAL_SB_GAUSSIAN_SPECTRAL  Exact SB solution on the spectral edge grid.
%
%   [rho_ana, mx_ana] = analytical_sb_gaussian_spectral(problem, vareps)
%
%   Same analytical formula as analytical_sb_gaussian but evaluated at the
%   (nt+1) edge time points t_n = n/nt, n = 0,...,nt, and the collocated
%   spatial grid x_j = (j-0.5)/nx used in discretize_first/1d_spectral/.
%
%   Outputs:
%     rho_ana  (nt+1 x nx)   density   at t = 0, dt, ..., 1
%     mx_ana   (nt+1 x nx)   momentum  at t = 0, dt, ..., 1  (collocated)

    mu0 = 1/3;   mu1 = 2/3;   sigma = 0.05;

    nt = problem.nt;   dt = problem.dt;
    nx = problem.nx;   dx = problem.dx;

    Normal = @(x, mu, sig) exp(-0.5*((x - mu)/sig).^2) / (sqrt(2*pi)*sig);

    alpha  = sqrt(sigma^4 + vareps^2) - sigma^2;

    % Edge time points: t = 0, dt, 2*dt, ..., 1
    t_edges = (0:nt)' * dt;            % (nt+1 x 1)
    x_cc    = ((1:nx) - 0.5) * dx;    % (1 x nx)  collocated spatial grid

    mu_t    = (1 - t_edges)*mu0 + t_edges*mu1;                  % (nt+1 x 1)
    sig2_t  = sigma^2 + 2*alpha * t_edges .* (1 - t_edges);    % (nt+1 x 1)
    sig_t   = sqrt(sig2_t);

    rho_ana = zeros(nt+1, nx);
    mx_ana  = zeros(nt+1, nx);

    for k = 1 : nt+1
        row = Normal(x_cc, mu_t(k), sig_t(k));
        rho_ana(k, :) = row / sum(row);

        v_k = (mu1 - mu0) + (alpha*(1 - 2*t_edges(k)) - vareps) / sig2_t(k) ...
              * (x_cc - mu_t(k));
        mx_ana(k, :) = rho_ana(k, :) .* v_k;
    end
end
