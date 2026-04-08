%% test_nablax_sign.m
%
% Quick test to confirm that the two-step operator in proj_fokker_planck:
%
%   nablax_mu = ops.deriv_x_at_m(mu)
%   nablax_mu = ops.deriv_x_at_phi(nablax_mu, zeros, zeros)
%
% computes the POSITIVE spatial Laplacian (+d_xx), not the negative one.
%
% Test function: f(x) = cos(pi*x)
%   f''(x) = -pi^2 * cos(pi*x)   <-- negative, so result should be negative
%
% If the operator is +d_xx we get -pi^2*cos(pi*x).
% If the operator is -d_xx we get +pi^2*cos(pi*x).

clear; clc;

base = fileparts(mfilename('fullpath'));
run(fullfile(base, '..', 'setup_paths.m'));

%% Setup
problem.nt = 32;  problem.dt = 1/32;
problem.nx = 128; problem.dx = 1/128;
ops = disc_staggered_1st(problem);

nt  = problem.nt;
nx  = problem.nx;
ntm = nt - 1;
dx  = problem.dx;

% Cell-center x-coordinates (phi-locations)
x_phi = ((1:nx) - 0.5) * dx;   % (1 x nx)

% Test field: f(x) = cos(pi*x), constant in time  (ntm x nx)
f     = repmat(cos(pi * x_phi), ntm, 1);
f_xx  = repmat(-pi^2 * cos(pi * x_phi), ntm, 1);   % exact d_xx f

%% Apply the two-step operator
zeros_ntm = zeros(ntm, 1);
nablax_f  = ops.deriv_x_at_m(f);                               % (ntm x nxm)
nablax_f  = ops.deriv_x_at_phi(nablax_f, zeros_ntm, zeros_ntm); % (ntm x nx)

%% Compare with exact second derivative (interior only — boundary is 1st order)
interior = 2:nx-1;
err_positive = max(abs(nablax_f(:,interior) - f_xx(:,interior)));
err_negative = max(abs(nablax_f(:,interior) + f_xx(:,interior)));

fprintf('Error vs +d_xx (positive Laplacian): %.2e\n', err_positive);
fprintf('Error vs -d_xx (negative Laplacian): %.2e\n', err_negative);

if err_positive < err_negative
    fprintf('\nResult: operator computes POSITIVE Laplacian (+d_xx)\n');
else
    fprintf('\nResult: operator computes NEGATIVE Laplacian (-d_xx)\n');
end
