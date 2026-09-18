function x_out = prox_ke_spectral(x_in, sigma, problem)
% PROX_KE_SPECTRAL  Proximal operator for KE on the staggered interior grid.
%
%   x_out = prox_ke_spectral(x_in, sigma, problem)
%
%   Variable layout:
%     x_in.rho  (nt-1 x nx)   interior density at t_1,...,t_{nt-1}
%     x_in.mx   ( nt  x nx)   momentum at:
%                                FE/IMEX: LEFT  edges t_0,...,t_{nt-1}
%                                BE:      RIGHT edges t_1,...,t_nt
%
%   KE pairing (mx row j in MATLAB = time slot j):
%
%     FE/IMEX (mx at LEFT edges):
%       j=1     : mx at t=0 paired with rho0   (fixed BC)
%       j=2..nt : mx at t_{j-1} paired with interior rho row j-1
%
%     BE (mx at RIGHT edges):
%       j=1..nt-1: mx at t_j paired with interior rho row j
%       j=nt    : mx at t=1 paired with rho1   (fixed BC)
%
%   Interior pair prox (exact, via cubic solve):
%     min_{r,m}  m^2/(2r)  +  (1/2sigma)*((r-rc)^2 + (m-mc)^2)
%
%   Boundary prox (rho fixed at rho_bc):
%     min_m  m^2/(2*rho_bc)  +  (1/2sigma)*(m-mc)^2
%     => mx_new = rho_bc .* mc ./ (rho_bc + sigma)

    nt   = problem.nt;
    rho0 = problem.rho0;   % (1 x nx)
    rho1 = problem.rho1;
    rho_c = x_in.rho;      % (nt-1 x nx)
    mx_c  = x_in.mx;       % (nt   x nx)

    rho_out = zeros(nt-1, size(rho_c, 2));
    mx_out  = zeros(nt,   size(mx_c,  2));

    use_be = strcmp(problem.time_disc, 'be');

    if ~use_be
        % FE / IMEX: mx at LEFT edges t_0,...,t_{nt-1}

        % Boundary: mx row 1 (t=0) paired with rho0
        mx_out(1, :) = rho0 .* mx_c(1,:) ./ (rho0 + sigma);

        % Interior: rho rows 1..nt-1 paired with mx rows 2..nt
        [rho_out, mx_int_out] = prox_pair(rho_c, mx_c(2:end, :), sigma);
        mx_out(2:end, :) = mx_int_out;

    else
        % BE: mx at RIGHT edges t_1,...,t_nt

        % Interior: rho rows 1..nt-1 paired with mx rows 1..nt-1
        [rho_out, mx_int_out] = prox_pair(rho_c, mx_c(1:end-1, :), sigma);
        mx_out(1:end-1, :) = mx_int_out;

        % Boundary: mx row nt (t=1) paired with rho1
        mx_out(nt, :) = rho1 .* mx_c(nt,:) ./ (rho1 + sigma);
    end

    x_out.rho = rho_out;
    x_out.mx  = mx_out;
end

% -------------------------------------------------------------------------
function [rho_new, mx_new] = prox_pair(rho_c, mx_c, sigma)
% Exact pointwise prox of  m^2/(2r)  for a batch of (rho,mx) pairs.
% Works on arrays of any matching size.
    rho_new = solve_cubic(1, 2*sigma - rho_c, sigma^2 - 2*sigma*rho_c, ...
                          -sigma*(sigma*rho_c + 0.5*mx_c.^2));
    mx_new  = rho_new .* mx_c ./ (rho_new + sigma);

    % Clip rare numerical negatives from cubic solver; do NOT use a
    % positive threshold — clipping small positive Gaussian tails creates
    % step discontinuities that cause Gibbs oscillations in the spectral step.
    neg_ind = (rho_new < 0);
    rho_new(neg_ind) = 0.0;
    mx_new(neg_ind)  = 0.0;
end
