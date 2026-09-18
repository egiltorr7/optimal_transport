function heat = precomp_heat_neumann_2d(problem, cfg)
% PRECOMP_HEAT_NEUMANN_2D  Precompute 2D heat kernels with Neumann BCs (DCT-II).
%
%   heat = precomp_heat_neumann_2d(problem, cfg)
%
%   Reference process: reflected Brownian motion on [0,1]^2.
%   The 2D heat semigroup H_t is separable and diagonalised by the 2D DCT-II:
%
%     H_t[phi]_hat(k,l) = exp(-eps * (lambda_x(k) + lambda_y(l)) * t) * phi_hat(k,l)
%
%   Inputs:
%     problem   struct with fields: lambda_x (1 x nx), lambda_y (1 x ny)
%     cfg       struct with fields: vareps
%
%   Output:
%     heat.apply_full    phi_out = heat.apply_full(phi_in)
%                        Apply H_T (full time T=1) — used in Sinkhorn loop.
%                        phi_in / phi_out are (nx x ny).
%     heat.apply_time    phi_out = heat.apply_time(phi_in, t)
%                        Apply H_t for arbitrary t — used for trajectory recovery.
%     heat.name          'neumann_2d'

    lambda_x = problem.lambda_x(:);     % (nx x 1)
    lambda_y = problem.lambda_y(:)';    % (1  x ny)
    vareps   = cfg.vareps;

    % 2D decay: lambda_2d(i,j) = lambda_x(i) + lambda_y(j)
    decay_T  = exp(-vareps * (lambda_x + lambda_y) * 1);   % (nx x ny), T=1

    heat.apply_full = @(phi) apply_neumann_2d(phi, decay_T);
    heat.apply_time = @(phi, t) apply_neumann_2d(phi, exp(-vareps * (lambda_x + lambda_y) * t));
    heat.name       = 'neumann_2d';
end

% -------------------------------------------------------------------------

function phi_out = apply_neumann_2d(phi_in, decay)
    phi_out = idct2(decay .* dct2(phi_in));
    phi_out = max(phi_out, 0);
end
