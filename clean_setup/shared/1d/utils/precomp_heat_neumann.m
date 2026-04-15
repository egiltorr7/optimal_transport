function heat = precomp_heat_neumann(problem, cfg)
% PRECOMP_HEAT_NEUMANN  Precompute heat kernels with Neumann BCs (DCT-II).
%
%   heat = precomp_heat_neumann(problem, cfg)
%
%   Reference process: reflected Brownian motion on [0,1].
%   The heat semigroup H_t is diagonalised by the DCT-II:
%
%     H_t[phi]_hat(k) = exp(-eps * lambda_k * t) * phi_hat(k)
%
%   Inputs:
%     problem   struct with fields: lambda_x (1 x nx), dt
%     cfg       struct with fields: vareps
%
%   Output:
%     heat.apply_full    phi_out = heat.apply_full(phi_in)
%                        Apply H_T (full time T=1) — used in Sinkhorn loop.
%     heat.apply_time    phi_out = heat.apply_time(phi_in, t)
%                        Apply H_t for arbitrary t — used for trajectory recovery.
%     heat.name          'neumann'

    lambda_x = problem.lambda_x(:);
    vareps   = cfg.vareps;

    decay_T  = exp(-vareps * lambda_x * 1);   % H_T, T=1

    heat.apply_full = @(phi) apply_neumann(phi(:), decay_T);
    heat.apply_time = @(phi, t) apply_neumann(phi(:), exp(-vareps * lambda_x * t));
    heat.name       = 'neumann';
end

% -------------------------------------------------------------------------

function phi_out = apply_neumann(phi_in, decay)
    phi_out = idct(decay .* dct(phi_in));
    phi_out = max(phi_out, 0);
end
