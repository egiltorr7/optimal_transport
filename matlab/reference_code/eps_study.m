%% eps_study.m  –  Banded vs Woodbury across epsilon values
%
%  Fixed grid: nx = nt = 128.
%  eps values: 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1  (decade steps).
%
%  Produces:
%   1. rho_snapshots.pdf  – rho(x) at 4 time instances per eps panel (2x3 grid)
%   2. conv.pdf           – residual + FP violation vs iteration, per solver (2x2)
%   3. failure.pdf        – final FP violation vs eps for both solvers

clear;

set(groot, 'defaultTextInterpreter',         'latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter',       'latex');

%% ============================================================
%% Parameters
%% ============================================================
nx      = 128;
nt      = 128;
gamma   = 1;
maxIter = 2000;

eps_vals   = 10.^(-6:1:-1);      % [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
n_eps      = numel(eps_vals);

solvers    = {'banded'};
sol_labels = {'Banded'};
sol_ls     = {'-'};        % line style per solver
n_sol      = numel(solvers);

% Time fractions to plot in the snapshot panels
t_plot  = [0.2, 0.4, 0.6, 0.8];
n_t     = numel(t_plot);

%% ============================================================
%% Figure style
%% ============================================================
sty.lw   = 1.5;
sty.alw  = 0.75;
sty.fs   = 9;
sty.fw   = 17.2;    % double-column width (cm)
sty.fh_snap = 5.5;  % height per row in snapshot figure
sty.fh2     = 11.0; % two-row convergence figure
sty.fhF     = 6.0;  % failure summary

% Solver colour  (blue = banded)
sty.Csol = [0.1216 0.4667 0.7059];

% Per-time colours: 4 entries from lines() colormap
t_colors = lines(n_t);

% Per-eps colours for convergence curves: parula strip
eps_cmap = parula(n_eps + 2);
eps_cmap = eps_cmap(2:n_eps+1, :);

% Per-eps line styles (helps differentiate in B&W)
ls_eps = {'-', '--', ':', '-.', '-', '--'};

fig_dir = fullfile('figures', 'eps_study');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% ============================================================
%% Grid and boundary conditions
%% ============================================================
dx  = 1/nx;  dt = 1/nt;
x   = linspace(0, 1, nx+1);
xx  = (x(1:end-1) + x(2:end)) / 2;

mu0 = 1/3;  sigma0 = 0.05;
mu1 = 2/3;  sigma1 = 0.05;
gauss = @(xv, mu, sig) exp(-0.5*((xv-mu)/sig).^2) / (sqrt(2*pi)*sig);

rho0 = gauss(xx, mu0, sigma0);  rho0 = rho0 / sum(rho0);
rho1 = gauss(xx, mu1, sigma1);  rho1 = rho1 / sum(rho1);

tt_inner = linspace(0, 1, nt+1)';
tt_inner = tt_inner(2:end-1);    % interior time points, length nt-1

% Map t_plot fractions -> indices into tt_inner
t_idx = max(1, min(nt-1, round(t_plot * (nt-1))));

%% ============================================================
%% Run all (solver, eps) combinations
%% ============================================================
res_rho  = cell(n_sol, n_eps);
res_diff = cell(n_sol, n_eps);
res_viol = cell(n_sol, n_eps);
res_ok   = true(n_sol, n_eps);

for si = 1:n_sol
    for ei = 1:n_eps
        eps_str = fmt_eps(eps_vals(ei));
        fprintf('[%-8s / eps=%s]  ', sol_labels{si}, eps_str);

        opts = struct('nt', nt, 'nx', nx, 'maxIter', maxIter, ...
                      'gamma', gamma, 'vareps', eps_vals(ei), ...
                      'postprocess', true, 'proj_method', solvers{si});
        try
            [rho_out, ~, outs] = sb1d_admm(rho0, rho1, opts);
            res_rho{si,ei}  = rho_out;
            res_diff{si,ei} = outs.residual_diff;
            res_viol{si,ei} = outs.constraint_viol;
            fprintf('done  (final viol = %.2e)\n', outs.constraint_viol(end));
        catch ME
            fprintf('FAILED: %s\n', ME.message);
            res_ok(si,ei)   = false;
            res_rho{si,ei}  = nan(nt-1, nx);
            res_diff{si,ei} = nan(maxIter, 1);
            res_viol{si,ei} = nan(maxIter, 1);
        end
    end
end

%% ============================================================
%% Fig 1: rho snapshots  –  2x3 panels (one per eps)
%%
%%   Each panel: rho(x) at t = 0.2, 0.4, 0.6, 0.8
%%     Solid  line = Banded
%%     Dotted      = Exact SB  (same colour, darker)
%%   Colour encodes time instance.
%% ============================================================
fig1 = new_fig(sty.fw, sty.fh_snap * 2);
tl1  = tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for ei = 1:n_eps
    vareps  = eps_vals(ei);
    eps_str = fmt_eps(vareps);
    c_sb    = sqrt(vareps^2 + sigma0^2*sigma1^2) - vareps;

    nexttile;
    hold on;

    % Dummy handles for legend (created inside hold-on so they belong to this axes)
    h_time = gobjects(n_t, 1);
    h_ex   = gobjects(n_t, 1);

    for ti = 1:n_t
        tk      = tt_inner(t_idx(ti));
        mu_t    = (1-tk)*mu0 + tk*mu1;
        var_sb  = (1-tk)^2*sigma0^2 + 2*tk*(1-tk)*(vareps+c_sb) + tk^2*sigma1^2;
        rho_ex  = gauss(xx, mu_t, sqrt(var_sb));
        rho_ex  = rho_ex / sum(rho_ex);

        if res_ok(1,ei)
            plot(xx, res_rho{1,ei}(t_idx(ti),:), ...
                 '-', 'Color', t_colors(ti,:), 'LineWidth', sty.lw, ...
                 'HandleVisibility', 'off');
        end
        % Exact SB: dotted, same colour, slightly thinner
        plot(xx, rho_ex, ':', 'Color', t_colors(ti,:) * 0.6, 'LineWidth', 0.9, ...
             'HandleVisibility', 'off');

        % Dummy handles for legend
        h_time(ti) = plot(NaN, NaN, '-', 'Color', t_colors(ti,:), ...
                          'LineWidth', sty.lw, ...
                          'DisplayName', sprintf('$t = %.1f$', t_plot(ti)));
        h_ex(ti)   = plot(NaN, NaN, ':', 'Color', t_colors(ti,:) * 0.6, ...
                          'LineWidth', 0.9, ...
                          'DisplayName', sprintf('exact, $t = %.1f$', t_plot(ti)));
    end

    hold off;
    style_ax(gca, sty);
    xlabel('$x$');  ylabel('$\rho$');
    title(sprintf('$\\varepsilon = %s$', eps_str));
    grid on;

    if ei == 1
        legend(h_time, 'Location', 'northwest', 'FontSize', sty.fs - 1);
    end
end

save_fig(fig1, fullfile(fig_dir, 'rho_snapshots'));

%% ============================================================
%% Fig 2: Convergence  –  2x1 panels (banded solver)
%%   Top:    residual ||u_{k+1} - u_k||
%%   Bottom: FP violation
%%   Curves coloured by eps (parula)
%% ============================================================
fig2 = new_fig(sty.fw, sty.fh2);
tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

% Top: residuals
nexttile;
hold on;
for ei = 1:n_eps
    d = res_diff{1,ei};
    semilogy(2:maxIter, d(2:end), ls_eps{ei}, ...
             'Color', eps_cmap(ei,:), 'LineWidth', sty.lw, ...
             'DisplayName', sprintf('$\\varepsilon = %s$', fmt_eps(eps_vals(ei))));
end
hold off;
style_ax(gca, sty);
xlabel('Iteration');
ylabel('$\|u_{k+1} - u_k\|$');
title('Residual -- Banded');
legend('Location', 'northeast', 'FontSize', sty.fs - 2);
grid on;

% Bottom: FP violations
nexttile;
hold on;
for ei = 1:n_eps
    v = res_viol{1,ei};
    semilogy(1:maxIter, v, ls_eps{ei}, ...
             'Color', eps_cmap(ei,:), 'LineWidth', sty.lw, ...
             'DisplayName', sprintf('$\\varepsilon = %s$', fmt_eps(eps_vals(ei))));
end
hold off;
style_ax(gca, sty);
xlabel('Iteration');
ylabel('$\|\partial_t\rho + \partial_x m - \varepsilon\Delta\rho\|_{L^2}$');
title('FP violation -- Banded');
legend('Location', 'northeast', 'FontSize', sty.fs - 2);
grid on;

save_fig(fig2, fullfile(fig_dir, 'conv'));

%% ============================================================
%% Fig 3: Final FP violation vs eps  (failure summary)
%% ============================================================
fail_thr = 1e-4;

viol_final = nan(n_sol, n_eps);
for si = 1:n_sol
    for ei = 1:n_eps
        if res_ok(si,ei)
            viol_final(si,ei) = res_viol{si,ei}(end);
        end
    end
end

fig3 = new_fig(sty.fw * 0.6, sty.fhF);
hold on;
loglog(eps_vals, viol_final(1,:), '-o', ...
       'Color', sty.Csol(1,:), 'LineWidth', sty.lw, ...
       'MarkerSize', 5, 'MarkerFaceColor', sty.Csol(1,:), ...
       'DisplayName', 'Banded');
yline(fail_thr, 'k--', 'LineWidth', 1.0, ...
      'DisplayName', sprintf('Threshold $= 10^{%d}$', round(log10(fail_thr))));
hold off;
style_ax(gca, sty);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('$\varepsilon$');
ylabel('Final FP violation');
title('Final violation vs $\varepsilon$  ($N_x = N_t = 128$,  $K = 2000$)');
legend('Location', 'best', 'FontSize', sty.fs - 1);
grid on;

save_fig(fig3, fullfile(fig_dir, 'failure'));

%% ============================================================
%% Console summary table
%% ============================================================
fprintf('\n=== Final FP violation ===\n');
hdr = sprintf('%-10s', '');
for ei = 1:n_eps
    hdr = [hdr, sprintf('%10s', fmt_eps(eps_vals(ei)))]; %#ok<AGROW>
end
fprintf('%s\n', hdr);
for si = 1:n_sol
    row = sprintf('%-10s', sol_labels{si});
    for ei = 1:n_eps
        if isnan(viol_final(si,ei))
            row = [row, sprintf('%10s', 'FAIL')]; %#ok<AGROW>
        else
            tag = '';
            if viol_final(si,ei) > fail_thr, tag = '*'; end
            row = [row, sprintf('%9.2e%s', viol_final(si,ei), tag)]; %#ok<AGROW>
        end
    end
    fprintf('%s\n', row);
end
fprintf('(* above threshold %.0e)\n', fail_thr);

%% ============================================================
%% Local functions
%% ============================================================

function s = fmt_eps(e)
    s = regexprep(sprintf('%.0e', e), 'e([+-])0+(\d)', 'e$1$2');
end

function fig = new_fig(fw, fh)
    fig = figure('Units',         'centimeters', ...
                 'Position',      [5 5 fw fh],   ...
                 'PaperUnits',    'centimeters',  ...
                 'PaperSize',     [fw fh],        ...
                 'PaperPosition', [0 0 fw fh],    ...
                 'Color',         'white');
end

function style_ax(ax, sty)
    set(ax, 'FontSize',       sty.fs,  ...
            'LineWidth',      sty.alw, ...
            'TickDir',        'out',   ...
            'Box',            'off',   ...
            'GridAlpha',      0.15,    ...
            'MinorGridAlpha', 0.05);
end

function save_fig(fig, path)
    exportgraphics(fig, [path '.pdf'], 'ContentType', 'vector', ...
                   'BackgroundColor', 'white');
    exportgraphics(fig, [path '.png'], 'Resolution', 300, ...
                   'BackgroundColor', 'white');
    fprintf('Saved: %s (.pdf, .png)\n', path);
end
