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
%   Optional opts fields:
%     opts.print_every  print progress every N iterations (default 100, 0 = silent)
%     opts.stall_window number of iters to look back for stall detection (default 50)
%     opts.stall_tol    residual must decrease by this factor or it's a stall (default 0.999)
%
%   Outputs:
%     x, y      primal variables at termination
%     delta     dual variable at termination (unscaled, same type as b)
%     info.residual   (n_iters x 1)  ||y^{t+1} - y^t|| at each iteration
%     info.iters      number of iterations taken
%     info.converged  true if stopped due to tolerance
%     info.stalled    true if residual stopped decreasing before convergence
%     info.diverged   true if NaN/Inf appeared in residual
%     info.walltime   elapsed wall time (seconds)

    gamma        = opts.gamma;
    tau          = opts.tau;
    alpha        = get_opt(opts, 'alpha',        1.0);
    max_iter     = opts.max_iter;
    tol          = opts.tol;
    print_every  = get_opt(opts, 'print_every',  100);
    stall_window = get_opt(opts, 'stall_window', 50);
    stall_tol    = get_opt(opts, 'stall_tol',    0.999);

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
    stalled  = false;
    diverged = false;

    t_start = tic;
    iter_times = zeros(min(max_iter, max(print_every, 1)), 1);  % ring buffer for iter timing

    if print_every > 0
        fprintf('  [ADMM]  iter       residual      best res   sec/iter    ETA\n');
        fprintf('  ---------------------------------------------------------------\n');
    end

    t_last_print = tic;

    for t = 1:max_iter

        t_iter_start = tic;
        y_prev  = y;
        By_prev = By;

        % --- x-subproblem (linearized) ---
        r = s_sub(s_add(A_fn(x), By_prev), b);
        g = At_fn(s_sub(s_scale(gamma, r), delta));
        x = prox_f1(s_sub(x, s_scale(1/tau, g)), 1/tau);

        % --- over-relaxation ---
        Ax    = A_fn(x);
        z_hat = s_add(s_scale(alpha,       Ax), ...
                      s_scale(1 - alpha,   s_sub(b, By_prev)));

        % --- y-subproblem (exact) ---
        y  = solve_y(delta, z_hat);
        By = B_fn(y);

        % --- dual update ---
        delta = s_sub(delta, s_scale(gamma, s_sub(s_add(z_hat, By), b)));

        % --- residual ---
        residual(t) = norm_fn(s_sub(y, y_prev));
        iter_times(mod(t-1, numel(iter_times))+1) = toc(t_iter_start);

        % --- NaN / Inf detection ---
        if ~isfinite(residual(t))
            diverged = true;
            residual = residual(1:t);
            if print_every > 0
                fprintf('  [ADMM]  *** DIVERGED at iter %d: residual = %g ***\n', t, residual(t));
                fprintf('  [ADMM]  Possible cause: Thomas instability (eps/dt > 1). Try proj_fokker_planck_spike2.\n');
            end
            break;
        end

        % --- stall detection: residual not decreasing over stall_window iters ---
        if t > stall_window
            ratio = residual(t) / residual(t - stall_window);
            stalled = ratio > stall_tol;
        end

        % --- convergence ---
        if residual(t) < tol
            residual = residual(1:t);
            if print_every > 0
                best_res = min(residual);
                sec_iter = mean(iter_times(iter_times > 0));
                fprintf('  [ADMM]  %6d/%d   %10.3e   %10.3e   %7.3fs  CONVERGED\n', ...
                    t, max_iter, residual(end), best_res, sec_iter);
            end
            break;
        end

        % --- periodic progress print ---
        if print_every > 0 && mod(t, print_every) == 0
            best_res = min(residual(1:t));
            sec_iter = mean(iter_times(iter_times > 0));
            remaining = (max_iter - t) * sec_iter;
            % Estimate ETA from convergence rate (geometric decay over stall_window)
            if t > stall_window && residual(t) > 0 && residual(t - stall_window) > 0
                decay_rate = (residual(t) / residual(t - stall_window)) ^ (1/stall_window);
                if decay_rate < 1
                    iters_to_tol = log(tol / residual(t)) / log(decay_rate);
                    remaining = min(remaining, iters_to_tol * sec_iter);
                end
            end
            stall_str = '';
            if stalled, stall_str = '  *** STALLED ***'; end
            fprintf('  [ADMM]  %6d/%d   %10.3e   %10.3e   %7.3fs  ETA %s%s\n', ...
                t, max_iter, residual(t), best_res, sec_iter, ...
                format_eta(remaining), stall_str);
        end
    end

    info.residual  = residual;
    info.iters     = length(residual);
    info.converged = ~diverged && residual(end) < tol;
    info.stalled   = stalled && ~info.converged;
    info.diverged  = diverged;
    info.walltime  = toc(t_start);
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

function s = format_eta(seconds)
    if ~isfinite(seconds) || seconds > 86400
        s = '  ???';
    elseif seconds >= 3600
        s = sprintf('%2.0fh%02.0fm', floor(seconds/3600), mod(floor(seconds/60),60));
    elseif seconds >= 60
        s = sprintf('%2.0fm%02.0fs', floor(seconds/60), mod(seconds,60));
    else
        s = sprintf('%5.0fs',  seconds);
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
        for f = fieldnames(a)', c.(f{1}) = zeros(size(a.(f{1})), 'like', a.(f{1})); end
    else
        c = zeros(size(a), 'like', a);
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
