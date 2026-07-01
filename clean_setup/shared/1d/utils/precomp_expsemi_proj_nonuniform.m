function ep = precomp_expsemi_proj_nonuniform(problem, vareps)
% PRECOMP_EXPSEMI_PROJ_NONUNIFORM  Precompute LU factors for non-uniform ETD projection.
%
%   ep = precomp_expsemi_proj_nonuniform(problem, vareps)
%
%   Like precomp_expsemi_proj but handles a non-uniform time grid stored in
%   problem.dt_vec (nt x 1).  The off-diagonal of T_k is no longer constant:
%
%     e_k(n) = -c_k^{n+1} / (dt_vec(n) * dt_vec(n+1))
%
%   where c_k^n = exp(-vareps * lambda_x(k) * dt_vec(n)).
%
%   Output fields:
%     ep.c_mat    (nt x nx)   semigroup coefficients per step and mode
%     ep.phi_mat  (nt x nx)   ETD weights per step and mode
%     ep.Tj_L     cell(nx,1)  LU factors L for k=1..nx
%     ep.Tj_U     cell(nx,1)  LU factors U for k=1..nx
%     ep.Tj_P     cell(nx,1)  LU factors P for k=1..nx

    nt     = problem.nt;
    nx     = problem.nx;
    dt_vec = problem.dt_vec;   % (nt x 1)
    lx     = problem.lambda_x; % (1 x nx)

    % Per-step, per-mode semigroup and ETD weights
    % c_mat(n,k) = exp(-vareps * lx(k) * dt_vec(n))
    alpha_mat = vareps * dt_vec * lx;   % (nt x nx) outer product
    c_mat     = exp(-alpha_mat);

    phi_mat        = ones(nt, nx);
    nz             = alpha_mat > 1e-14;
    phi_mat(nz)    = (1 - c_mat(nz)) ./ alpha_mat(nz);

    ep.c_mat   = c_mat;
    ep.phi_mat = phi_mat;

    ep.Tj_L = cell(nx, 1);
    ep.Tj_U = cell(nx, 1);
    ep.Tj_P = cell(nx, 1);

    for k = 1:nx
        ck  = c_mat(:, k);    % (nt x 1)
        phk = phi_mat(:, k);  % (nt x 1)
        lxk = lx(k);

        % Diagonal entries (nt x 1)
        d      = (1 + ck.^2) ./ dt_vec.^2 + phk.^2 * lxk;
        d(1)   =  1          / dt_vec(1)^2 + phk(1)^2 * lxk;
        d(nt)  =  ck(nt)^2  / dt_vec(nt)^2 + phk(nt)^2 * lxk;

        % Off-diagonal entries (nt-1 x 1): coupling between rows n and n+1
        % comes from c_{k}^{n+1} / (dt_n * dt_{n+1})
        e = -ck(2:nt) ./ (dt_vec(1:nt-1) .* dt_vec(2:nt));

        % Gauge regularisation for k=1 (DC mode, T_1 singular)
        if k == 1
            d(1) = d(1) + 1e-10 / dt_vec(1)^2;
        end

        Tk = sparse(1:nt,   1:nt,   d,  nt, nt) + ...
             sparse(2:nt,   1:nt-1, e,  nt, nt) + ...
             sparse(1:nt-1, 2:nt,   e,  nt, nt);

        [L, U, P] = lu(Tk);
        ep.Tj_L{k} = L;
        ep.Tj_U{k} = U;
        ep.Tj_P{k} = P;
    end
end
