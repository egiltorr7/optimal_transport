function [A, c_bc, idx] = build_constraint_operator(nx, L, dt, dx, rho0, rho1)
% BUILD_CONSTRAINT_OPERATOR  Sparse Jacobian of the discrete continuity
%   constraint for the Fisher-information-regularized (Li-Yin-Osher 2018)
%   formulation, i.e. the ORDINARY (inviscid) continuity equation:
%
%       (p_l(j) - p_{l-1}(j))/dt + (m_l(j) - m_l(j-1))/dx = 0
%
%   for l = 1,...,L+1, j = 1,...,nx, with boundary data
%       p_0     = rho0     (fixed)
%       p_{L+1} = rho1     (fixed)
%       m_l(0) = m_l(nx) = 0   (no-flux / Neumann BC)
%
%   Free variables are ordered as u = [p_vec; m_vec]:
%     p_vec: levels l=1..L,   level l occupies  (l-1)*nx       + (1:nx)
%     m_vec: levels l=1..L+1, level l occupies  n_p + (l-1)*(nx-1) + (1:nx-1)
%   where n_p = L*nx.
%
%   The constraint is written  A*u + c_bc = 0,  with c_bc absorbing the
%   fixed boundary terms (rho0 at l=1, rho1 at l=L+1).
%
%   Outputs:
%     A     sparse ((L+1)*nx) x (L*nx + (L+1)*(nx-1))
%     c_bc  ((L+1)*nx) x 1
%     idx   struct with fields nx, L, n_p, n_m, n_free, p_range(l), m_range(l)
%           (p_range/m_range are function handles returning index vectors
%           into u for level l)

    n_p = L * nx;
    n_m = (L + 1) * (nx - 1);
    n_free = n_p + n_m;
    n_rows = (L + 1) * nx;

    idx.nx     = nx;
    idx.L      = L;
    idx.n_p    = n_p;
    idx.n_m    = n_m;
    idx.n_free = n_free;
    idx.p_range = @(l) (l - 1) * nx + (1:nx);                       % l = 1..L
    idx.m_range = @(l) n_p + (l - 1) * (nx - 1) + (1:nx - 1);       % l = 1..L+1

    rho0 = rho0(:)';
    rho1 = rho1(:)';

    % Generous preallocation for triplet lists (4 nonzeros per row max,
    % from +p_l, -p_{l-1}, +m_l(j), -m_l(j-1); trimmed automatically by sparse()).
    I = zeros(4 * n_rows, 1);
    J = zeros(4 * n_rows, 1);
    V = zeros(4 * n_rows, 1);
    nnz_ct = 0;
    c_bc = zeros(n_rows, 1);

    for l = 1:(L + 1)
        rows = (l - 1) * nx + (1:nx);   % (1 x nx)

        % --- +p_l(j)/dt ---
        if l <= L
            cols = idx.p_range(l);
            [I, J, V, nnz_ct] = add_triplets(I, J, V, nnz_ct, rows, cols, 1/dt * ones(1, nx));
        else
            c_bc(rows) = c_bc(rows) + rho1(:) / dt;
        end

        % --- -p_{l-1}(j)/dt ---
        if l >= 2
            cols = idx.p_range(l - 1);
            [I, J, V, nnz_ct] = add_triplets(I, J, V, nnz_ct, rows, cols, -1/dt * ones(1, nx));
        else
            c_bc(rows) = c_bc(rows) - rho0(:) / dt;
        end

        % --- +m_l(j)/dx  for j = 1..nx-1 (edge j is the right-flux of node j) ---
        m_cols = idx.m_range(l);                 % edges 1..nx-1
        rows_j = rows(1:nx-1);                    % nodes j = 1..nx-1
        [I, J, V, nnz_ct] = add_triplets(I, J, V, nnz_ct, rows_j, m_cols, 1/dx * ones(1, nx-1));

        % --- -m_l(j-1)/dx  for j = 2..nx (edge j-1 is the left-flux of node j) ---
        rows_j = rows(2:nx);                      % nodes j = 2..nx
        [I, J, V, nnz_ct] = add_triplets(I, J, V, nnz_ct, rows_j, m_cols, -1/dx * ones(1, nx-1));
    end

    I = I(1:nnz_ct);
    J = J(1:nnz_ct);
    V = V(1:nnz_ct);
    A = sparse(I, J, V, n_rows, n_free);
end

function [I, J, V, nnz_ct] = add_triplets(I, J, V, nnz_ct, rows, cols, vals)
% Adds one nonzero (rows(k), cols(k), vals(k)) per k = 1..numel(rows).
% rows, cols, vals must all be the same length (paired, not outer-product).
    n = numel(rows);
    idx_range = nnz_ct + (1:n);
    I(idx_range) = rows(:);
    J(idx_range) = cols(:);
    V(idx_range) = vals(:);
    nnz_ct = nnz_ct + n;
end
