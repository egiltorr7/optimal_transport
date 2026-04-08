function sp = precomp_spectral_proj(problem, cfg)
% PRECOMP_SPECTRAL_PROJ  Precompute per-mode projection matrices (interior grid).
%
%   sp = precomp_spectral_proj(problem, cfg)
%
%   Time grid:
%     rho : (nt-1) x nx  at interior times  t_i = i*dt,  i = 1,...,nt-1
%     mx  :  nt    x nx  at:
%              'fe' / 'imex': LEFT  edges  t_j = j*dt,  j = 0,...,nt-1
%              'be':          RIGHT edges  t_j = j*dt,  j = 1,...,nt
%
%   Variable vector per mode k  (size 2*nt-1):
%     u_k = [rho_hat_k(1:nt-1); mx_hat_k(1:nt)]
%
%   Constraint matrix L_k is  nt x (2*nt-1).
%   BCs rho0 and rho1 are substituted into the RHS f_k (not extra rows).
%
%   Schemes  (alpha = 1 - dt*eps*w^2,  gamma = 1 + dt*eps*w^2,  beta = dt*i*w):
%
%   FE ('fe') -- fully explicit:
%     n=1:         rho(1)          + beta*mx(0)   = alpha*rho0_k
%     n=2..nt-1:   rho(n) - alpha*rho(n-1) + beta*mx(n-1) = 0
%     n=nt:       -alpha*rho(nt-1) + beta*mx(nt-1) = -rho1_k
%
%   IMEX ('imex') -- implicit diffusion, explicit divergence; mx at LEFT edges
%   BE   ('be')   -- fully implicit;                          mx at RIGHT edges
%     (IMEX and BE share the same L_k; only RHS scaling differs from FE)
%     n=1:         gamma*rho(1)    + beta*mx(?)   = rho0_k
%     n=2..nt-1:   gamma*rho(n) - rho(n-1) + beta*mx(?) = 0
%     n=nt:       -rho(nt-1)      + beta*mx(?)   = -gamma*rho1_k
%
%   The RHS f_k is assembled in proj_fokker_planck_spectral using:
%     f_k(1)  = f_scale_0(k) * rho0_hat(k)
%     f_k(nt) += f_scale_T(k) * rho1_hat(k)    ("+=" handles nt=1 correctly)
%
%   Projection:
%     u_out = u_in - Lp_k * (L_k * u_in - f_k)
%   where Lp_k = pinv(L_k) is computed via SVD for numerical stability.

    nt     = problem.nt;
    dt     = problem.dt;
    vareps = cfg.vareps;
    omega  = problem.ops.wavenumbers;   % (1 x nx)
    nx     = problem.nx;
    is_fe  = strcmp(cfg.time_disc, 'fe');

    sp.L         = cell(1, nx);
    sp.Lp        = cell(1, nx);      % pseudoinverse of L_k, size (2*nt-1) x nt
    sp.f_scale_0 = zeros(1, nx);     % coefficient of rho0_hat(k) in f_k(1)
    sp.f_scale_T = zeros(1, nx);     % coefficient of rho1_hat(k) in f_k(nt)

    for k = 1 : nx
        w     = omega(k);
        w2    = w * w;
        beta  = dt * 1i * w;
        alpha = 1 - dt * vareps * w2;   % FE coefficient
        gamma = 1 + dt * vareps * w2;   % IMEX/BE coefficient

        % Constraint matrix L_k: nt x (2*nt-1)
        %   Columns 1..nt-1     : rho_k at interior times t_1,...,t_{nt-1}
        %   Columns nt..2*nt-1  : mx_k  (LEFT edges for FE/IMEX; RIGHT for BE)
        %
        % Note: for nt=1 there are no rho columns (nt-1=0); the single column
        % is the mx column.  The row-1 code sets L_k(1,1)=coeff then
        % L_k(1,nt=1)=beta, so beta overwrites the nonexistent rho entry —
        % this gives the correct 1x1 system by coincidence of indexing.
        L_k = zeros(nt, 2*nt - 1);

        if is_fe
            % Row 1: rho(1) + beta*mx(0) = alpha*rho0
            L_k(1, 1)  = 1;
            L_k(1, nt) = beta;

            % Rows 2..nt-1
            for n = 2 : nt-1
                L_k(n, n)      = 1;
                L_k(n, n-1)    = -alpha;
                L_k(n, nt+n-1) = beta;
            end

            % Row nt: -alpha*rho(nt-1) + beta*mx(nt-1) = -rho1
            if nt >= 2
                L_k(nt, nt-1)   = -alpha;
                L_k(nt, 2*nt-1) = beta;
            end

            sp.f_scale_0(k) = alpha;
            sp.f_scale_T(k) = -1;
        else
            % IMEX / BE: same matrix structure
            % Row 1: gamma*rho(1) + beta*mx(?) = rho0
            L_k(1, 1)  = gamma;
            L_k(1, nt) = beta;

            % Rows 2..nt-1
            for n = 2 : nt-1
                L_k(n, n)      = gamma;
                L_k(n, n-1)    = -1;
                L_k(n, nt+n-1) = beta;
            end

            % Row nt: -rho(nt-1) + beta*mx(?) = -gamma*rho1
            if nt >= 2
                L_k(nt, nt-1)   = -1;
                L_k(nt, 2*nt-1) = beta;
            end

            sp.f_scale_0(k) = 1;
            sp.f_scale_T(k) = -gamma;
        end

        sp.L{k}  = L_k;
        sp.Lp{k} = pinv(L_k);
    end
end
