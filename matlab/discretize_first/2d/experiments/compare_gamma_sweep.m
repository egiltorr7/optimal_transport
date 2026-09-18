% COMPARE_GAMMA_SWEEP
%
% Groups banded-solver result .mat files by (nt, nx, ny, eps) and, for each
% group, produces overlay comparison figures across gamma values:
%
%   fp_residual       max FP residual vs time t
%   admm_res_x        ||x_{k+1} - x_k||  vs iteration
%   admm_res_y        ||y_{k+1} - y_k||  vs iteration
%   admm_res_primal   ||Ax - y||          vs iteration
%   mass_conservation total mass vs time t
%   diagslice         density along y=x at t = 0.25, 0.5, 0.75
%
% One set of figures is saved per (nt, nx, ny, eps) group.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
fig_dir = fullfile(res_dir, 'figures', 'gamma_sweep');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

ADMM_SKIP = 10;   % skip first N iterations in ADMM plots

% -------------------------------------------------------------------------
% 1. Scan all .mat files, keep only banded, index by group key
% -------------------------------------------------------------------------
mats = dir(fullfile(res_dir, 'result_*.mat'));
if isempty(mats)
    fprintf('No result_*.mat files found in %s\n', res_dir); return;
end

groups = struct();   % groups.(key).files, .gammas

for i = 1:numel(mats)
    MAT_FILE = fullfile(res_dir, mats(i).name);
    tmp = load(MAT_FILE, 'cfg');
    c   = tmp.cfg;

    proj_str = func2str(c.projection);
    if ~contains(proj_str, 'banded'), continue; end

    key = sprintf('nt%d_nx%d_ny%d_eps%s', c.nt, c.nx, c.ny, num2str(c.vareps, '%g'));
    key = matlab.lang.makeValidName(key);

    if ~isfield(groups, key)
        groups.(key).files  = {};
        groups.(key).gammas = [];
    end
    groups.(key).files{end+1}  = MAT_FILE;
    groups.(key).gammas(end+1) = c.gamma;
end

keys = fieldnames(groups);
if isempty(keys)
    fprintf('No banded result files found.\n'); return;
end
fprintf('Found %d group(s) of banded results.\n', numel(keys));

% -------------------------------------------------------------------------
% 2. Generate comparison figures for each group
% -------------------------------------------------------------------------
for ig = 1:numel(keys)
    key   = keys{ig};
    grp   = groups.(key);
    ng    = numel(grp.gammas);

    % Sort by gamma ascending
    [gammas_sorted, ord] = sort(grp.gammas);
    files_sorted = grp.files(ord);

    if ng < 2
        fprintf('Group %s: only 1 gamma value, skipping.\n', key); continue;
    end

    fprintf('\nGroup %s  (%d gamma values: %s)\n', key, ng, ...
        strjoin(arrayfun(@(g) sprintf('%.3g',g), gammas_sorted, 'UniformOutput', false), ', '));

    colors = lines(ng);
    savepng = @(fig, name) exportgraphics(fig, ...
        fullfile(fig_dir, sprintf('%s_%s.png', name, key)), 'Resolution', 150);

    % -- Preallocate per-file data structs --
    D = struct();

    for k = 1:ng
        s = load(files_sorted{k}, 'result', 'cfg', 'problem');
        D(k).result  = s.result;
        D(k).cfg     = s.cfg;
        D(k).problem = s.problem;
        D(k).gamma   = gammas_sorted(k);
        D(k).label   = sprintf('\\gamma=%.3g', gammas_sorted(k));
    end

    % Reference problem fields from first file
    prob = D(1).problem;
    nt = prob.nt;  dt = prob.dt;
    nx = prob.nx;  dx = prob.dx;
    ny = prob.ny;  dy = prob.dy;
    ops  = prob.ops;
    rho0 = prob.rho0;
    rho1 = prob.rho1;
    t_cc = ((1:nt)' - 0.5) * dt;

    % ---------------------------------------------------------------
    % Figure: FP residual vs time
    % ---------------------------------------------------------------
    fig = figure('Name', ['FP residual ' key], 'Position', [50 50 620 340]);
    hold on;
    zeros_x = zeros(nt, ny);
    zeros_y = zeros(nt, nx);
    for k = 1:ng
        r = D(k).result;
        x_stag.rho = r.rho_stag;
        x_stag.mx  = r.mx_stag;
        x_stag.my  = r.my_stag;
        rho_phi   = ops.interp_t_at_phi(x_stag.rho, rho0, rho1);
        nabla_rho = ops.deriv_x_at_phi(ops.deriv_x_at_m(rho_phi),  zeros_x, zeros_x) ...
                  + ops.deriv_y_at_phi(ops.deriv_y_at_m(rho_phi),  zeros_y, zeros_y);
        fp_res = ops.deriv_t_at_phi(x_stag.rho, rho0, rho1) ...
               + ops.deriv_x_at_phi(x_stag.mx, zeros_x, zeros_x) ...
               + ops.deriv_y_at_phi(x_stag.my, zeros_y, zeros_y) ...
               - D(k).cfg.vareps * nabla_rho;
        fp_per_t = squeeze(max(max(abs(fp_res), [], 2), [], 3));
        semilogy(t_cc, fp_per_t, '-', 'Color', colors(k,:), 'LineWidth', 1.5, ...
            'DisplayName', D(k).label);
    end
    set(gca, 'YScale', 'log');
    xlabel('t'); ylabel('max_{x,y} |FP residual|');
    title(sprintf('FP residual   %s', strrep(key,'_',' ')));
    legend('Location', 'best'); grid on;
    savepng(fig, 'fp_residual'); close(fig);

    % ---------------------------------------------------------------
    % Figures: ADMM residuals (three separate)
    % ---------------------------------------------------------------
    admm_specs = {
        'res_x',      'admm\_res\_x',      '||x^{k+1}-x^k||',  'admm_res_x';
        'res_y',      'admm\_res\_y',      '||y^{k+1}-y^k||',  'admm_res_y';
        'res_primal', 'admm\_res\_primal', '||Ax-y||',          'admm_res_primal';
    };

    for ia = 1:size(admm_specs, 1)
        field    = admm_specs{ia,1};
        ttl      = admm_specs{ia,2};
        ylbl     = admm_specs{ia,3};
        savename = admm_specs{ia,4};

        fig = figure('Name', [ttl ' ' key], 'Position', [50 50 620 340]);
        first = true;
        for k = 1:ng
            r = D(k).result;
            if ~isfield(r, field), continue; end
            data = r.(field);
            n    = numel(data);
            if n <= ADMM_SKIP, continue; end
            iters = (ADMM_SKIP+1 : n)';
            if first
                semilogy(iters, data(ADMM_SKIP+1:end), '-', 'Color', colors(k,:), ...
                    'LineWidth', 1.5, 'DisplayName', D(k).label);
                hold on; first = false;
            else
                semilogy(iters, data(ADMM_SKIP+1:end), '-', 'Color', colors(k,:), ...
                    'LineWidth', 1.5, 'DisplayName', D(k).label);
            end
        end
        set(gca, 'YScale', 'log');
        % draw tol line from first file (same for all in group)
        yline(D(1).cfg.tol, 'k--', sprintf('tol=%.0e', D(1).cfg.tol));
        xlabel('Iteration'); ylabel(ylbl);
        title(sprintf('%s   %s', ttl, strrep(key,'_',' ')));
        legend('Location', 'best'); grid on;
        savepng(fig, savename); close(fig);
    end

    % ---------------------------------------------------------------
    % Figure: mass conservation
    % ---------------------------------------------------------------
    fig = figure('Name', ['Mass ' key], 'Position', [50 50 620 340]);
    hold on;
    for k = 1:ng
        mass = squeeze(sum(sum(D(k).result.rho_cc, 2), 3)) * dx * dy;
        plot(t_cc, mass, '-', 'Color', colors(k,:), 'LineWidth', 1.5, ...
            'DisplayName', D(k).label);
    end
    xlabel('t'); ylabel('total mass');
    title(sprintf('Mass conservation   %s', strrep(key,'_',' ')));
    legend('Location', 'best'); grid on;
    savepng(fig, 'mass_conservation'); close(fig);

    % ---------------------------------------------------------------
    % Figure: diagonal slice at t = 0.25, 0.5, 0.75
    % ---------------------------------------------------------------
    if nx == ny
        t_snaps  = [0.25, 0.5, 0.75];
        k_snaps  = max(1, round(t_snaps * nt));
        xx = prob.xx;

        fig = figure('Name', ['Diagslice ' key], 'Position', [50 50 1100 320]);
        for ip = 1:numel(t_snaps)
            subplot(1, numel(t_snaps), ip); hold on;
            for k = 1:ng
                d = diag(squeeze(D(k).result.rho_cc(k_snaps(ip),:,:)));
                plot(xx, d, '-', 'Color', colors(k,:), 'LineWidth', 1.5, ...
                    'DisplayName', D(k).label);
            end
            xlabel('x'); ylabel('\rho(t,x,y=x)');
            title(sprintf('t \\approx %.2f', (k_snaps(ip)-0.5)*dt));
            legend('Location', 'best', 'FontSize', 7); grid on;
        end
        sgtitle(sprintf('Diagonal slice y=x   %s', strrep(key,'_',' ')));
        savepng(fig, 'diagslice'); close(fig);
    end

    fprintf('  Figures saved for group %s\n', key);
end

fprintf('\nAll done. Figures in: %s\n', fig_dir);
