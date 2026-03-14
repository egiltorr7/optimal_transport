function [x, z, info] = admm_solve(prox_f, prox_g, x0, opts)
% ADMM_SOLVE  Abstract over-relaxed ADMM following Fang et al. (GADMM).
%
%   [x, z, info] = admm_solve(prox_f, prox_g, x0, opts)
%
%   Solves the consensus problem:
%
%       min   f(x) + g(z)   s.t.   x = z
%
%   which is the special case A1=I, A2=-I, b=0 of the general GADMM problem
%
%       min   theta1(x) + theta2(z)   s.t.   A1*x + A2*z = b
%
%   from: Fang et al., "Generalized alternating direction method of
%   multipliers: new theoretical insights and applications", 2015.
%
%   Algorithm (scaled dual form, over-relaxation parameter alpha):
%
%     x^{k+1}  = prox_f( z^k + u^k,  sigma )          ... x-step
%     z_hat    = alpha * x^{k+1} + (1-alpha) * z^k     ... over-relaxation
%     z^{k+1}  = prox_g( z_hat  - u^k,  sigma )        ... z-step
%     u^{k+1}  = u^k - ( z_hat - z^{k+1} )             ... dual update
%
%   where sigma = 1/rho and u = lambda/rho is the scaled dual variable.
%   alpha = 1 recovers standard ADMM; alpha in (1,2) is over-relaxation.
%
%   The state variables x, z, u can be structs, arrays, or any type that
%   supports the arithmetic operations needed -- these are provided via
%   opts.vec_ops (see below). If opts.vec_ops is omitted, the state is
%   assumed to be a struct with numeric array fields (fieldwise ops).
%
%   Inputs:
%     prox_f    @(v, sigma) -> v_new   proximal operator of f
%     prox_g    @(v, sigma) -> v_new   proximal operator of g
%                                      (indicator function -> orthogonal projection)
%     x0                               initial primal variable (same type as x, z, u)
%     opts.rho        penalty parameter rho (> 0)
%     opts.alpha      over-relaxation parameter (default 1.0, range (0,2))
%     opts.max_iter   maximum number of iterations
%     opts.tol        convergence tolerance on ||z^{k+1} - z^k||
%     opts.norm_fn    @(v) -> scalar   norm used for convergence check
%                                      (default: Euclidean norm of all fields)
%
%   Outputs:
%     x, z      primal variables at termination
%     info.residual   (n_iters x 1)  ||z^{k+1} - z^k|| at each iteration
%     info.iters      number of iterations taken
%     info.converged  true if stopped due to tolerance
%     info.walltime   elapsed wall time (seconds)

    %% Parse options
    rho      = opts.rho;
    alpha    = get_opt(opts, 'alpha', 1.0);
    max_iter = opts.max_iter;
    tol      = opts.tol;
    sigma    = 1 / rho;

    %% Vector operations (fieldwise on structs, or standard if numeric array)
    if isfield(opts, 'norm_fn')
        norm_fn = opts.norm_fn;
    else
        norm_fn = @(v) struct_norm(v);
    end

    %% Initialise primal and dual variables
    x = x0;
    z = x0;
    u = struct_zeros(x0);   % scaled dual variable u = lambda / rho

    residual = zeros(max_iter, 1);

    tic;
    for k = 1:max_iter

        z_prev = z;

        % --- x-step: prox of f ---
        x = prox_f(struct_add(z, u), sigma);

        % --- over-relaxation ---
        z_hat = struct_add(struct_scale(alpha,     x), ...
                           struct_scale(1 - alpha, z));

        % --- z-step: prox of g ---
        z = prox_g(struct_sub(z_hat, u), sigma);

        % --- dual update ---
        u = struct_sub(u, struct_sub(z_hat, z));

        % --- convergence check: change in z ---
        residual(k) = norm_fn(struct_sub(z, z_prev));

        if residual(k) < tol
            residual = residual(1:k);
            break;
        end
    end

    info.residual  = residual;
    info.iters     = length(residual);
    info.converged = residual(end) < tol;
    info.walltime  = toc;
end

%% -----------------------------------------------------------------------
%% Struct arithmetic helpers (fieldwise operations, work for any struct
%% with numeric array fields or for plain numeric arrays)
%% -----------------------------------------------------------------------

function c = struct_add(a, b)
    if isstruct(a)
        c = a;
        for f = fieldnames(a)'
            c.(f{1}) = a.(f{1}) + b.(f{1});
        end
    else
        c = a + b;
    end
end

function c = struct_sub(a, b)
    if isstruct(a)
        c = a;
        for f = fieldnames(a)'
            c.(f{1}) = a.(f{1}) - b.(f{1});
        end
    else
        c = a - b;
    end
end

function c = struct_scale(s, a)
    if isstruct(a)
        c = a;
        for f = fieldnames(a)'
            c.(f{1}) = s * a.(f{1});
        end
    else
        c = s * a;
    end
end

function c = struct_zeros(a)
    if isstruct(a)
        c = a;
        for f = fieldnames(a)'
            c.(f{1}) = zeros(size(a.(f{1})));
        end
    else
        c = zeros(size(a));
    end
end

function n = struct_norm(a)
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

function v = get_opt(opts, field, default)
    if isfield(opts, field)
        v = opts.(field);
    else
        v = default;
    end
end
