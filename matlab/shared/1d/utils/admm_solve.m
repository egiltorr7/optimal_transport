function [x, y, delta, info] = admm_solve(solve_x, solve_y, A_fn, B_fn, b, x0, y0, opts)
% ADMM_SOLVE  Over-relaxed GADMM following Fang, He, Liu, Yuan (2015) eq. (3).
%
%   [x, y, delta, info] = admm_solve(solve_x, solve_y, A_fn, B_fn, b, x0, y0, opts)
%
%   Solves the structured convex problem
%
%       min   f1(x) + f2(y)
%       s.t.  A*x + B*y = b,   x in X, y in Y
%
%   Algorithm (eq. 3, unscaled dual δ, over-relaxation α):
%
%     x^{t+1} = argmin_{x in X}  f1(x) - x^T A^T δ^t
%                                 + (γ/2) || A*x + B*y^t - b ||²
%
%     z_hat   = α * A*x^{t+1} + (1-α) * (b - B*y^t)   % over-relaxation
%
%     y^{t+1} = argmin_{y in Y}  f2(y) - y^T B^T δ^t
%                                 + (γ/2) || z_hat + B*y - b ||²
%
%     δ^{t+1} = δ^t - γ * (z_hat + B*y^{t+1} - b)
%
%   α = 1 recovers standard ADMM; α ∈ (1,2) gives over-relaxation.
%
%   Inputs:
%     solve_x   @(delta, y)       -> x_new   solves x-subproblem
%     solve_y   @(delta, z_hat)   -> y_new   solves y-subproblem
%                                             (z_hat = α*Ax + (1-α)*(b-By))
%     A_fn      @(x)  -> A*x      linear operator A
%     B_fn      @(y)  -> B*y      linear operator B
%     b                           right-hand side (same type as A*x and B*y)
%     x0                          initial x
%     y0                          initial y
%     opts.gamma      penalty parameter γ (> 0)
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
    delta = s_zeros(b);   % unscaled dual variable δ, same shape as b
    By    = B_fn(y0);     % cache B*y to avoid calling B_fn twice per iteration

    residual = zeros(max_iter, 1);

    tic;
    for t = 1:max_iter

        y_prev = y;
        By_prev = By;   % B*y^t already computed from previous iteration

        % --- x-subproblem ---
        x = solve_x(delta, y);

        % --- over-relaxation ---
        Ax = A_fn(x);
        % z_hat = α * A*x^{t+1} + (1-α) * (b - B*y^t)
        z_hat = s_add(s_scale(alpha,       Ax), ...
                      s_scale(1 - alpha,   s_sub(b, By_prev)));

        % --- y-subproblem ---
        y  = solve_y(delta, z_hat);
        By = B_fn(y);   % compute B*y^{t+1} once, reuse below and next iter

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
%% Local helpers (local subfunctions are resolved at parse time by MATLAB,
%% avoiding the path-lookup overhead of external .m files)
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
