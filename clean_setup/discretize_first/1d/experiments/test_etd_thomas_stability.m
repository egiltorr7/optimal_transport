% TEST_ETD_THOMAS_STABILITY
%
% For a range of eps and grid sizes, checks the minimum eigenvalue and
% minimum Thomas pivot of T_k for the ETD scheme (precomp_expsemi_proj).
% Mirrors test_thomas_stability but for 1D ETD instead of 2D banded.

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..')));

%% Parameters to sweep
nt_vals  = [16, 32, 64, 128, 256];
eps_vals = [0.01, 0.1, 1.0, 5.0, 20.0, 100.0];
nx       = 32;

fprintf('%-6s  %-10s  %-14s  %-14s\n', 'nt', 'eps', 'min_pivot', 'min_eig');
fprintf('%s\n', repmat('-', 1, 50));

for nt = nt_vals
    dt = 1 / nt;

    problem.nt = nt;
    problem.nx = nx;
    problem.dt = dt;
    problem.dx = 1 / nx;
    problem.lambda_x = (2 - 2*cos(pi * (0:nx-1)  / nx)) / problem.dx^2;
    problem.lambda_t = (2 - 2*cos(pi * (0:nt-1)' / nt)) / problem.dt^2;

    for vareps = eps_vals
        ep = precomp_expsemi_proj(problem, vareps);

        min_pivot   = inf;
        min_eig_val = inf;

        for k = 2:nx
            cj  = ep.c_vals(k);
            phj = ep.phi_vals(k);
            lx  = problem.lambda_x(k);

            d      = ((1 + cj^2)/dt^2 + phj^2 * lx) * ones(nt, 1);
            d(1)   = 1/dt^2 + phj^2 * lx;
            d(nt)  = cj^2/dt^2 + phj^2 * lx;
            od     = -cj / dt^2;

            % Thomas forward sweep: track minimum pivot
            bmod = d;
            for i = 2:nt
                bmod(i) = bmod(i) - (od / bmod(i-1)) * od;
            end
            min_pivot = min(min_pivot, min(bmod));

            % Minimum eigenvalue
            T  = diag(d) + diag(od * ones(nt-1,1), 1) + diag(od * ones(nt-1,1), -1);
            ev = eig(T);
            min_eig_val = min(min_eig_val, min(real(ev)));
        end

        fprintf('%-6d  %-10.3g  %-14.4e  %-14.4e\n', nt, vareps, min_pivot, min_eig_val);
    end
end

%% Plot min eigenvalue vs eps for a fixed nt, all modes k=2..nx
nt  = 64;
dt  = 1 / nt;
nx2 = 64;

problem2.nt = nt;
problem2.nx = nx2;
problem2.dt = dt;
problem2.dx = 1 / nx2;
problem2.lambda_x = (2 - 2*cos(pi * (0:nx2-1)  / nx2)) / problem2.dx^2;
problem2.lambda_t = (2 - 2*cos(pi * (0:nt-1)'  / nt )) / problem2.dt^2;

eps_range    = logspace(-2, 2, 60);
min_eig_vs_eps = zeros(size(eps_range));

for ei = 1:numel(eps_range)
    vareps = eps_range(ei);
    ep     = precomp_expsemi_proj(problem2, vareps);
    mev    = inf;
    for k = 2:nx2
        cj  = ep.c_vals(k);
        phj = ep.phi_vals(k);
        lx  = problem2.lambda_x(k);
        d   = ((1 + cj^2)/dt^2 + phj^2 * lx) * ones(nt, 1);
        d(1)  = 1/dt^2 + phj^2 * lx;
        d(nt) = cj^2/dt^2 + phj^2 * lx;
        od  = -cj / dt^2;
        T   = diag(d) + diag(od*ones(nt-1,1),1) + diag(od*ones(nt-1,1),-1);
        mev = min(mev, min(real(eig(T))));
    end
    min_eig_vs_eps(ei) = mev;
end

figure;
loglog(eps_range, min_eig_vs_eps, 'b-o', 'MarkerSize', 4);
xlabel('eps', 'Interpreter', 'none');
ylabel('min eigenvalue of T_k', 'Interpreter', 'none');
title(sprintf('ETD: min eigenvalue over k=2..%d, nt=%d', nx2, nt), 'Interpreter', 'none');
grid on;
