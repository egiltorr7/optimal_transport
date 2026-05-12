% TEST_THOMAS_STABILITY
%
% For a range of eps/dt ratios and grid sizes, checks:
%   1. Minimum Thomas pivot across all (kx,ky) modes
%   2. Minimum eigenvalue of T_{kx,ky} (PSD check)
%   3. Diagonal dominance margin on the last row
%
% Run this script standalone — no problem struct needed.

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));

%% Parameters to sweep
nt_vals      = [16, 32, 64];
eps_dt_vals  = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
nx           = 16;   % fixed spatial grid for the sweep
ny           = 16;

fprintf('%-6s  %-8s  %-14s  %-14s  %-14s\n', ...
    'nt', 'eps/dt', 'min_pivot', 'min_eig', 'dd_margin_last');
fprintf('%s\n', repmat('-', 1, 65));

for nt = nt_vals
    dt = 1 / nt;

    % Minimal problem struct needed by precomp_banded_proj
    problem.nt = nt;
    problem.nx = nx;
    problem.ny = ny;
    problem.dt = dt;
    problem.dx = 1 / nx;
    problem.dy = 1 / ny;

    nxm = nx - 1;
    nym = ny - 1;
    ntm = nt - 1;
    problem.lambda_x = (2 - 2*cos(pi * problem.dx * (0:nxm))) / problem.dx^2;
    problem.lambda_y = (2 - 2*cos(pi * problem.dy * (0:nym))) / problem.dy^2;
    problem.lambda_t = (2 - 2*cos(pi * dt         * (0:ntm)'  )) / dt^2;

    for eps_dt = eps_dt_vals
        vareps = eps_dt * dt;

        bp = precomp_banded_proj(problem, vareps);

        min_pivot   = inf;
        min_eig_val = inf;
        min_dd      = inf;   % diagonal dominance margin on last row

        for kx = 1:nx
            for ky = 1:ny
                a = bp.lower_all(:, kx, ky);   % (nt-1 x 1)
                b = bp.main_all(:,  kx, ky);   % (nt   x 1)
                c = bp.upper_all(:, kx, ky);   % (nt-1 x 1)

                % --- Thomas forward sweep: track minimum pivot ---
                bmod = b;
                for i = 2:nt
                    bmod(i) = bmod(i) - (a(i-1) / bmod(i-1)) * c(i-1);
                end
                min_pivot = min(min_pivot, min(abs(bmod)));

                % --- PSD check via minimum eigenvalue ---
                T = diag(b) + diag(c, 1) + diag(a, -1);
                ev = eig(T);
                min_eig_val = min(min_eig_val, min(real(ev)));

                % --- Diagonal dominance margin on last row ---
                dd_last = abs(b(nt)) - abs(a(nt-1));   % no upper entry on last row
                min_dd  = min(min_dd, dd_last);
            end
        end

        fprintf('%-6d  %-8.2f  %-14.4e  %-14.4e  %-14.4e\n', ...
            nt, eps_dt, min_pivot, min_eig_val, min_dd);
    end
end

%% Visualize for a single nt: heatmaps over (kx,ky) at a problematic eps/dt
nt     = 32;
dt     = 1 / nt;
vareps = 2.0 * dt;   % eps/dt = 2

problem.nt = nt; problem.dt = dt;
ntm = nt - 1;
problem.lambda_t = (2 - 2*cos(pi * dt * (0:ntm)')) / dt^2;

bp = precomp_banded_proj(problem, vareps);

min_pivot_map = zeros(nx, ny);
min_eig_map   = zeros(nx, ny);
dd_map        = zeros(nx, ny);

for kx = 1:nx
    for ky = 1:ny
        a = bp.lower_all(:, kx, ky);
        b = bp.main_all(:,  kx, ky);
        c = bp.upper_all(:, kx, ky);

        bmod = b;
        for i = 2:nt
            bmod(i) = bmod(i) - (a(i-1) / bmod(i-1)) * c(i-1);
        end
        min_pivot_map(kx, ky) = min(abs(bmod));

        T = diag(b) + diag(c, 1) + diag(a, -1);
        min_eig_map(kx, ky) = min(real(eig(T)));

        dd_map(kx, ky) = abs(b(nt)) - abs(a(nt-1));
    end
end

figure('Name', sprintf('Thomas stability: nt=%d, eps/dt=%.1f', nt, vareps/dt));

subplot(1,3,1);
imagesc(log10(max(min_pivot_map, 1e-16)));
colorbar; axis xy;
title('log_{10}(min Thomas pivot)');
xlabel('ky index'); ylabel('kx index');

subplot(1,3,2);
imagesc(min_eig_map);
colorbar; axis xy;
title('min eigenvalue of T (PSD check)');
xlabel('ky index'); ylabel('kx index');

subplot(1,3,3);
imagesc(dd_map);
colorbar; axis xy;
title('DD margin last row');
xlabel('ky index'); ylabel('kx index');
