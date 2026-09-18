function [x, y, delta, info] = ladmm_solve_monitored(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts, iter_fn)
% LADMM_SOLVE_MONITORED  Identical to ladmm_solve but calls iter_fn(x, k)
%   unconditionally after each x-update.  Use only in post-processing.
%
%   [x, y, delta, info] = ladmm_solve_monitored(..., iter_fn)
%
%   iter_fn  @(x, k) -> scalar   called after every x-step; return value
%                                 stored in info.iter_vals(k)
%
%   All other arguments are identical to ladmm_solve.
%   See ladmm_solve.m for full documentation.

    gamma    = opts.gamma;
    tau      = opts.tau;
    alpha    = get_opt(opts, 'alpha', 1.0);
    max_iter = opts.max_iter;
    tol      = opts.tol;

    if isfield(opts, 'norm_fn')
        norm_fn = opts.norm_fn;
    else
        norm_fn = @s_norm;
    end

    x     = x0;
    y     = y0;
    delta = s_zeros(b);
    By    = B_fn(y0);

    residual  = zeros(max_iter, 1);
    iter_vals = zeros(max_iter, 1);

    tic;
    for t = 1:max_iter

        y_prev  = y;
        By_prev = By;

        % --- x-subproblem (linearized) ---
        r = s_sub(s_add(A_fn(x), By_prev), b);
        g = At_fn(s_sub(s_scale(gamma, r), delta));
        x = prox_f1(s_sub(x, s_scale(1/tau, g)), 1/tau);

        % --- monitor ---
        iter_vals(t) = iter_fn(x, t);

        % --- over-relaxation ---
        Ax    = A_fn(x);
        z_hat = s_add(s_scale(alpha,     Ax), ...
                      s_scale(1 - alpha, s_sub(b, By_prev)));

        % --- y-subproblem (exact) ---
        y  = solve_y(delta, z_hat);
        By = B_fn(y);

        % --- dual update ---
        delta = s_sub(delta, s_scale(gamma, s_sub(s_add(z_hat, By), b)));

        % --- convergence on ||y^{t+1} - y^t|| ---
        residual(t) = norm_fn(s_sub(y, y_prev));

        if residual(t) < tol
            residual  = residual(1:t);
            iter_vals = iter_vals(1:t);
            break;
        end
    end

    info.residual  = residual;
    info.iter_vals = iter_vals;
    info.iters     = length(residual);
    info.converged = residual(end) < tol;
    info.walltime  = toc;
end

%% -----------------------------------------------------------------------
%% Local helpers (duplicated from ladmm_solve to keep files independent)
%% -----------------------------------------------------------------------

function v = get_opt(opts, field, default)
    if isfield(opts, field)
        v = opts.(field);
    else
        v = default;
    end
end

function c = s_add(a, b)
    if isstruct(a)
        c = a;
        for f = fieldnames(a)', c.(f{1}) = a.(f{1}) + b.(f{1}); end
    else
        c = a + b;
    end
end

function c = s_sub(a, b)
    if isstruct(a)
        c = a;
        for f = fieldnames(a)', c.(f{1}) = a.(f{1}) - b.(f{1}); end
    else
        c = a - b;
    end
end

function c = s_scale(s, a)
    if isstruct(a)
        c = a;
        for f = fieldnames(a)', c.(f{1}) = s * a.(f{1}); end
    else
        c = s * a;
    end
end

function c = s_zeros(a)
    if isstruct(a)
        c = a;
        for f = fieldnames(a)', c.(f{1}) = zeros(size(a.(f{1}))); end
    else
        c = zeros(size(a));
    end
end

function n = s_norm(a)
    if isstruct(a)
        n = 0;
        for f = fieldnames(a)'
            n = n + sum(a.(f{1})(:).^2);
        end
        n = sqrt(n);
    else
        n = norm(a(:));
    end
end
