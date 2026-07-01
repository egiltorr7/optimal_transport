function heat = precomp_heat_neumann_log(problem, cfg)
% PRECOMP_HEAT_NEUMANN_LOG  Log-domain heat kernel with Neumann BCs (DCT-II).
%
%   heat = precomp_heat_neumann_log(problem, cfg)
%
%   Drop-in replacement for precomp_heat_neumann for use with
%   sinkhorn_hopf_cole_logdomain.  All apply functions operate in
%   log-space: they accept log(phi) and return log(H_t[phi]).
%
%   Stabilization: subtract max before exponentiation (log-sum-exp trick),
%   so exp() is applied to values in (-inf, 0] — no overflow.
%
%   Inputs:
%     problem   struct with fields: lambda_x (nx x 1)
%     cfg       struct with fields: vareps
%
%   Output:
%     heat.apply_full   log_phi_out = heat.apply_full(log_phi_in)
%                       Apply H_T (full time T=1) in log-space.
%     heat.apply_time   log_phi_out = heat.apply_time(log_phi_in, t)
%                       Apply H_t for arbitrary t in log-space.
%     heat.name         'neumann_log'

    lambda_x = problem.lambda_x(:);
    vareps   = cfg.vareps;
    decay_T  = exp(-vareps * lambda_x);   % H_T, T=1

    heat.apply_full = @(lp) log_apply_neumann(lp(:), decay_T);
    heat.apply_time = @(lp, t) log_apply_neumann(lp(:), exp(-vareps * lambda_x * t));
    heat.name       = 'neumann_log';
end

% -------------------------------------------------------------------------

function lp_out = log_apply_neumann(lp_in, decay)
    % Use max over finite entries only; cap +Inf to c_max+500 so exp stays finite
    fin     = isfinite(lp_in);
    if ~any(fin)
        lp_out = -inf(size(lp_in));
        return;
    end
    c_max        = max(lp_in(fin));
    lp_safe      = lp_in;
    lp_safe(~fin & lp_in > 0) = c_max + 500;   % +Inf → large but finite
    % -Inf entries → exp(-Inf - c_max) = 0, contribute nothing
    phi_hat = dct(exp(lp_safe - c_max));
    phi_out = idct(decay .* phi_hat);
    phi_out = max(phi_out, 0);
    lp_out  = log(phi_out) + c_max;
end
