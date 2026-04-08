function [x, y, delta, info] = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts)
% LADMM_SOLVE  Over-relaxed linearized GADMM following Fang, He, Liu, Yuan (2015).
%
%   [x, y, delta, info] = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts)
%
%   Solves the structured convex problem
%
%       min   f1(x) + f2(y)
%       s.t.  A*x + B*y = b,   x in X, y in Y
%
%   The x-subproblem from standard GADMM eq. (3) is linearized around x^t:
%   the quadratic penalty (γ/2)||Ax + By^t - b||² is replaced by its
%   first-order Taylor expansion plus a proximal term (τ/2)||x - x^t||²,
%   giving a simple gradient step followed by a proximal operator:
%
%     r       = A*x^t + B*y^t - b                    % primal residual
%     g       = A^T * (γ*r - δ^t)                    % gradient w.r.t. x
%     x^{t+1} = prox_{f1/τ}( x^t - (1/τ)*g )        % proximal gradient step
%
%   The y-subproblem and dual update follow eq. (3) exactly (same as GADMM):
%
%     z_hat   = α * A*x^{t+1} + (1-α) * (b - B*y^t) % over-relaxation
%     y^{t+1} = argmin_{y in Y}  f2(y) - y^T B^T δ^t
%                                 + (γ/2) || z_hat + B*y - b ||²
%     δ^{t+1} = δ^t - γ * (z_hat + B*y^{t+1} - b)
%
%   Convergence requires τ > γ * ||A||².
%
%   Inputs:
%     prox_f1   @(v, step) -> x_new   proximal operator of f1 with step-size step
%                                      i.e. argmin f1(x) + (step/2)||x - v||²
%     solve_y   @(delta, z_hat) -> y_new   solves y-subproblem exactly
%     A_fn      @(x)  -> A*x      linear operator A
%     At_fn     @(v)  -> A^T*v    adjoint of A
%     B_fn      @(y)  -> B*y      linear operator B
%     b                           right-hand side (same type as A*x and B*y)
%     x0                          initial x
%     y0                          initial y
%     opts.gamma      penalty parameter γ (> 0)
%     opts.tau        proximal penalty τ for x-update (> 0, τ > γ||A||²)
%     opts.alpha      over-relaxation parameter (default 1.0)
%     opts.max_iter   maximum iterations
%     opts.tol        convergence tolerance on ||y^{t+1} - y^t|| (norm_fn)
%     opts.norm_fn    @(v) -> scalar   norm for convergence (default: struct norm)
%
%   Outputs:
%     x, y      primal variables at termination
%     delta     dual variable at termination (unscaled, same type as b)
%     info.residual   (n_iters x 1)  ||y^{t+1} - y^t|| at each iteration
%     info.iters      number of iterations taken
%     info.converged  true if stopped due to tolerance
%     info.walltime   elapsed wall time (seconds)

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
    By    = B_fn(y0);     % cache B*y to avoid calling B_fn twice per iteration

    residual = zeros(max_iter, 1);

    tic;
    for t = 1:max_iter

        y_prev  = y;
        By_prev = By;

        % --- x-subproblem (linearized) ---
        % r = A*x^t + B*y^t - b
        r = s_sub(s_add(A_fn(x), By_prev), b);
        % g = A^T * (γ*r - δ^t)
        g = At_fn(s_sub(s_scale(gamma, r), delta));
        % x^{t+1} = prox_{f1/τ}(x^t - (1/τ)*g)
        x = prox_f1(s_sub(x, s_scale(1/tau, g)), 1/tau);

        % --- over-relaxation ---
        Ax    = A_fn(x);
        z_hat = s_add(s_scale(alpha,       Ax), ...
                      s_scale(1 - alpha,   s_sub(b, By_prev)));

        % --- y-subproblem (exact) ---
        y  = solve_y(delta, z_hat);
        By = B_fn(y);

        % --- dual update ---
        % δ^{t+1} = δ^t - γ * (z_hat + B*y^{t+1} - b)
        delta = s_sub(delta, s_scale(gamma, s_sub(s_add(z_hat, By), b)));

        % --- convergence on ||y^{t+1} - y^t|| ---
        residual(t) = norm_fn(s_sub(y, y_prev));

        if residual(t) < tol
            residual = residual(1:t);
            break;
        end
    end

    info.residual  = residual;
    info.iters     = length(residual);
    info.converged = residual(end) < tol;
    info.walltime  = toc;
end

%% -----------------------------------------------------------------------
%% Local helpers
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
