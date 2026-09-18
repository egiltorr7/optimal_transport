function [p, m, info] = newton_fisher_sb(rho0, rho1, dx, dt, L, vareps, opts)
% NEWTON_FISHER_SB  Proximal-Newton solver for the Fisher-information-
%   regularized dynamic Schrodinger bridge problem.
%
%   [p, m, info] = newton_fisher_sb(rho0, rho1, dx, dt, L, vareps, opts)
%
%   Reference: Li, Yin & Osher, "Computations of optimal transport
%   distance with Fisher information regularization", J. Sci. Comput. 75
%   (2018), Sect. 3 ("Algorithm").
%
%   Solves, over free interior time levels l = 1..L:
%
%     min_{p,m}  sum_{l=1}^{L+1} [ K_l(m_l,p_l) + vareps^2 * I_l(p_l) ]
%     s.t.       (p_l - p_{l-1})/dt + div(m_l) = 0,  l = 1..L+1
%                p_0 = rho0,  p_{L+1} = rho1,  m_l(0)=m_l(nx)=0
%
%   K_l, I_l are the discrete kinetic energy and Fisher information
%   per LEVEL_OBJECTIVE. This is the graph/discrete form of paper Eq.(7),
%   obtained from the ORIGINAL Fokker-Planck-constrained Schrodinger
%   bridge by the m -> m - vareps*grad(p) change of variables (paper
%   Sect. 2 "Derivation of (5)"): the constraint here is the ORDINARY
%   continuity equation, with the diffusion strength vareps entering only
%   through the Fisher-information penalty on p, not through the
%   constraint. This is the same vareps as this codebase's cfg.vareps.
%
%   IMPORTANT: this file was written without access to a MATLAB/Octave
%   interpreter to execute or debug it against. It runs two self-checks
%   automatically before the main loop (see opts.run_selftest, default
%   true): (1) the analytic gradient vs. a directional finite difference
%   of the full objective, and (2) the assembled sparse constraint matrix
%   vs. an independently-coded direct residual computation. Both print a
%   PASS/FAIL report; a FAIL is a `warning`, not an error, so you can see
%   what happens downstream, but you should not trust the solve results
%   if either check fails. Report the printed relative errors back if you
%   hit a failure -- that pinpoints whether the bug is in the objective/
%   gradient or in the constraint assembly.
%
%   Inputs:
%     rho0, rho1  (1 x nx)  boundary probability masses (each summing to 1)
%     dx, dt                grid spacing
%     L                     number of free interior time levels
%     vareps                Fisher-info / diffusion strength (beta in the paper)
%     opts.max_iter  (50)    paper uses ~50 Newton steps
%     opts.tol       (1e-5)  relative objective-change stopping tolerance (paper's criterion)
%     opts.alpha     (0.3)   base Newton step size (paper's fixed choice)
%     opts.fd_h      (1e-6)  RELATIVE finite-difference step for the per-level Hessian
%     opts.hess_reg_rel (1e-6)  Tikhonov regularization added to H before
%                            the KKT solve, as hess_reg_rel * max(diag(H))
%                            * I. K(m,p)+vareps^2*I(p) is exactly
%                            positively homogeneous of degree 1 in (p_l,m_l)
%                            jointly (Euler's theorem: perspective functions
%                            never have a strongly convex Hessian), so each
%                            per-level Hessian block has an EXACT zero
%                            eigenvalue along the (p_l,m_l) direction at
%                            every iterate, regardless of flooring. The full
%                            KKT system is generically still nonsingular
%                            (the constraint Jacobian A does not annihilate
%                            that same direction), but can be numerically
%                            very ill-conditioned; this regularization is a
%                            standard Levenberg-Marquardt-style safeguard.
%     opts.max_backtrack (20)
%     opts.verbose   (true)
%     opts.run_selftest (true)
%
%   Outputs:
%     p  (L x nx)       interior densities (rows are levels 1..L)
%     m  (L+1 x nx-1)   fluxes (rows are levels 1..L+1)
%     info.obj_hist, info.iters, info.walltime, info.converged
%     info.selftest     struct with the self-check results

    if nargin < 7, opts = struct(); end
    max_iter     = get_opt(opts, 'max_iter', 50);
    tol          = get_opt(opts, 'tol', 1e-5);
    alpha0       = get_opt(opts, 'alpha', 0.3);
    fd_h         = get_opt(opts, 'fd_h', 1e-6);
    hess_reg_rel = get_opt(opts, 'hess_reg_rel', 1e-6);
    max_backtrack = get_opt(opts, 'max_backtrack', 20);
    verbose      = get_opt(opts, 'verbose', true);
    run_selftest = get_opt(opts, 'run_selftest', true);

    nx = numel(rho0);
    rho0 = rho0(:)';
    rho1 = rho1(:)';

    [A, c_bc, idx] = build_constraint_operator(nx, L, dt, dx, rho0, rho1);

    info.selftest = struct();
    if run_selftest
        info.selftest.gradient   = check_gradient(idx, rho0, rho1, vareps, dx, verbose);
        info.selftest.constraint = check_constraint(A, c_bc, idx, nx, L, dt, dx, rho0, rho1, verbose);
    end

    % --- feasible initialization ---
    t_frac = (1:L)' / (L + 1);
    p0mat = (1 - t_frac) * rho0 + t_frac * rho1;   % (L x nx)

    u = zeros(idx.n_free, 1);
    for l = 1:L
        u(idx.p_range(l)) = p0mat(l, :)';
    end
    % Solve A_m * m = -(A_p*p + c_bc) for a feasible m^0 (least-squares;
    % exact if the RHS is in the range of A_m, which mass conservation of
    % the linear-interpolation p^0 guarantees up to floating point).
    Am = A(:, idx.n_p + 1 : end);
    rhs = -(A(:, 1:idx.n_p) * u(1:idx.n_p) + c_bc);
    u(idx.n_p + 1:end) = Am \ rhs;

    resid0 = norm(A * u + c_bc);
    if verbose
        fprintf('[newton_fisher_sb] feasible init: ||A*u+c|| = %.3e (should be ~1e-10 or smaller)\n', resid0);
    end

    % --- Newton loop ---
    obj_hist = zeros(max_iter, 1);

    tic;
    converged = false;
    k = 0;
    for k = 1:max_iter
        [obj_cur, grad] = objective_and_grad(u, idx, rho0, rho1, vareps, dx);
        H = assemble_hessian(u, idx, rho0, rho1, vareps, dx, fd_h);

        % Tikhonov safeguard against the exact per-level null direction
        % (see opts.hess_reg_rel doc above). Scaled to H's own magnitude
        % so it stays a small perturbation regardless of problem scale.
        if hess_reg_rel > 0
            reg = hess_reg_rel * max(abs(diag(H)));
            H = H + reg * speye(idx.n_free);
        end

        n_rows = size(A, 1);
        KKT = [H, A'; A, sparse(n_rows, n_rows)];
        rhs_kkt = [-grad; zeros(n_rows, 1)];
        sol = KKT \ rhs_kkt;
        d = sol(1:idx.n_free);

        % --- backtracking safeguard: keep all p > 0 and objective non-increasing ---
        alpha = alpha0;
        accepted = false;
        for bt = 1:max_backtrack
            u_try = u + alpha * d;
            if all_p_positive(u_try, idx, L)
                [obj_try, ~] = objective_and_grad(u_try, idx, rho0, rho1, vareps, dx);
                if obj_try <= obj_cur + 1e-12 * abs(obj_cur)
                    accepted = true;
                    break;
                end
            end
            alpha = alpha / 2;
        end
        if ~accepted
            if verbose
                fprintf('[newton_fisher_sb] iter %d: line search failed after %d backtracks, stopping.\n', k, max_backtrack);
            end
            obj_hist(k) = obj_cur;
            break;
        end

        u = u_try;
        obj_hist(k) = obj_try;

        rel_change = abs(obj_try - obj_cur) / max(abs(obj_cur), eps);
        if verbose
            fprintf('[newton_fisher_sb] iter %3d:  obj = %.8e   rel_change = %.3e   alpha = %.3g\n', ...
                k, obj_try, rel_change, alpha);
        end

        if rel_change < tol
            converged = true;
            break;
        end
    end
    walltime = toc;

    obj_hist = obj_hist(1:k);

    p = zeros(L, nx);
    for l = 1:L
        p(l, :) = u(idx.p_range(l))';
    end
    m = zeros(L + 1, nx - 1);
    for l = 1:(L + 1)
        m(l, :) = u(idx.m_range(l))';
    end

    info.obj_hist  = obj_hist;
    info.iters     = k;
    info.converged = converged;
    info.walltime  = walltime;
    info.final_constraint_residual = norm(A * u + c_bc);
end

%% ------------------------------------------------------------------
function v = get_opt(opts, field, default)
    if isfield(opts, field), v = opts.(field); else, v = default; end
end

%% ------------------------------------------------------------------
function [obj, grad] = objective_and_grad(u, idx, rho0, rho1, vareps, dx)
    L = idx.L;
    grad = zeros(idx.n_free, 1);
    obj = 0;
    for l = 1:L
        p_l = u(idx.p_range(l))';
        m_l = u(idx.m_range(l))';
        [obj_l, gp, gm] = level_objective(p_l, m_l, vareps, dx);
        obj = obj + obj_l;
        grad(idx.p_range(l)) = grad(idx.p_range(l)) + gp';
        grad(idx.m_range(l)) = grad(idx.m_range(l)) + gm';
    end
    % level L+1: p is fixed (= rho1), only m is free
    m_last = u(idx.m_range(L + 1))';
    [obj_l, ~, gm] = level_objective(rho1, m_last, vareps, dx);
    obj = obj + obj_l;
    grad(idx.m_range(L + 1)) = grad(idx.m_range(L + 1)) + gm';
end

%% ------------------------------------------------------------------
function H = assemble_hessian(u, idx, rho0, rho1, vareps, dx, fd_h)
    L = idx.L;
    nx = idx.nx;
    block_sz = 2 * nx - 1;
    % Preallocate triplets: L full (2nx-1)^2 blocks + one (nx-1)^2 block.
    max_nnz = L * block_sz^2 + (nx - 1)^2;
    I = zeros(max_nnz, 1); J = zeros(max_nnz, 1); V = zeros(max_nnz, 1);
    ct = 0;

    for l = 1:L
        p_l = u(idx.p_range(l))';
        m_l = u(idx.m_range(l))';
        Hl = level_hessian_fd(p_l, m_l, vareps, dx, fd_h);
        rows_cols = [idx.p_range(l), idx.m_range(l)];
        [I, J, V, ct] = place_block(I, J, V, ct, Hl, rows_cols, rows_cols);
    end

    % level L+1: only the m-m block is free (p fixed at rho1)
    m_last = u(idx.m_range(L + 1))';
    Hl_full = level_hessian_fd(rho1, m_last, vareps, dx, fd_h);
    Hmm = Hl_full(nx+1:end, nx+1:end);
    rc = idx.m_range(L + 1);
    [I, J, V, ct] = place_block(I, J, V, ct, Hmm, rc, rc);

    I = I(1:ct); J = J(1:ct); V = V(1:ct);
    H = sparse(I, J, V, idx.n_free, idx.n_free);
end

function [I, J, V, ct] = place_block(I, J, V, ct, block, rows, cols)
    [rr, cc] = ndgrid(rows, cols);
    n = numel(block);
    r = ct + (1:n);
    I(r) = rr(:); J(r) = cc(:); V(r) = block(:);
    ct = ct + n;
end

%% ------------------------------------------------------------------
function tf = all_p_positive(u, idx, L)
    tf = true;
    for l = 1:L
        if any(u(idx.p_range(l)) <= 0)
            tf = false;
            return;
        end
    end
end

%% ------------------------------------------------------------------
function report = check_gradient(idx, rho0, rho1, vareps, dx, verbose)
% Directional finite-difference check of objective_and_grad at a random
% strictly-feasible-in-sign point (does not need to satisfy the linear
% constraint -- the objective and gradient don't depend on it).
    rng_state = rng;
    rng(0);

    L = idx.L; nx = idx.nx; %#ok<NASGU>
    t_frac = (1:L)' / (L + 1);
    p0mat = (1 - t_frac) * rho0 + t_frac * rho1;

    u = zeros(idx.n_free, 1);
    for l = 1:L
        u(idx.p_range(l)) = p0mat(l, :)';
    end
    u(idx.n_p + 1:end) = 0.01 * (rand(idx.n_m, 1) - 0.5);

    dirn = randn(idx.n_free, 1);
    dirn(1:idx.n_p) = dirn(1:idx.n_p) .* 0.1 .* u(1:idx.n_p);  % scale so p stays positive
    dirn = dirn / norm(dirn);

    h = 1e-6;
    [~, grad] = objective_and_grad(u, idx, rho0, rho1, vareps, dx);
    [obj_p, ~] = objective_and_grad(u + h * dirn, idx, rho0, rho1, vareps, dx);
    [obj_m, ~] = objective_and_grad(u - h * dirn, idx, rho0, rho1, vareps, dx);

    fd_deriv  = (obj_p - obj_m) / (2 * h);
    an_deriv  = grad' * dirn;
    rel_err = abs(fd_deriv - an_deriv) / max(abs(fd_deriv), 1e-12);

    report.fd_deriv = fd_deriv;
    report.an_deriv = an_deriv;
    report.rel_err  = rel_err;
    report.passed   = rel_err < 1e-4;

    if verbose
        fprintf('[selftest] gradient check: analytic=%.8e  fd=%.8e  rel_err=%.3e  -> %s\n', ...
            an_deriv, fd_deriv, rel_err, ternary(report.passed, 'PASS', 'FAIL'));
    end
    if ~report.passed
        warning('newton_fisher_sb:gradient_check_failed', ...
            'Analytic gradient does not match finite-difference directional derivative (rel_err=%.3e). Do not trust the solve.', rel_err);
    end

    rng(rng_state);
end

%% ------------------------------------------------------------------
function report = check_constraint(A, c_bc, idx, nx, L, dt, dx, rho0, rho1, verbose)
% Cross-checks the assembled sparse A against an independently-coded
% direct (plain-loop) computation of the same continuity residual, on a
% random test point. Catches row/column indexing bugs that a
% "solve then re-check with the same A" test would not.
    p_full = zeros(L + 2, nx);
    p_full(1, :)     = rho0;
    p_full(end, :)   = rho1;
    p_full(2:end-1,:) = rand(L, nx) * 0.1 + 0.05;   % arbitrary positive interior

    m = (rand(L + 1, nx - 1) - 0.5) * 0.1;

    u = zeros(idx.n_free, 1);
    for l = 1:L
        u(idx.p_range(l)) = p_full(l + 1, :)';
        u(idx.m_range(l)) = m(l, :)';
    end
    u(idx.m_range(L + 1)) = m(L + 1, :)';

    resid_A = A * u + c_bc;
    resid_A = reshape(resid_A, nx, L + 1)';   % (L+1) x nx

    resid_direct = zeros(L + 1, nx);
    for l = 1:(L + 1)
        m_l = m(l, :);
        m_ext = [0, m_l, 0];   % ghost zero-flux at both ends
        div_m = (m_ext(2:end) - m_ext(1:end-1)) / dx;
        resid_direct(l, :) = (p_full(l + 1, :) - p_full(l, :)) / dt + div_m;
    end

    err = max(abs(resid_A(:) - resid_direct(:)));
    report.max_abs_err = err;
    report.passed = err < 1e-10;

    if verbose
        fprintf('[selftest] constraint check: max|A*u+c - direct_residual| = %.3e  -> %s\n', ...
            err, ternary(report.passed, 'PASS', 'FAIL'));
    end
    if ~report.passed
        warning('newton_fisher_sb:constraint_check_failed', ...
            'Assembled constraint matrix does not match independent direct computation (max_err=%.3e). Do not trust the solve.', err);
    end
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
