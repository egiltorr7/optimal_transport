%% Main script: 1-D Schrödinger Bridge via ADMM
clear;

%% LaTeX renderer for all figure text
set(groot, 'defaultTextInterpreter',         'latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter',       'latex');

%% Grid
% CFL constraint
vareps  = 1e-1 ;      % Schrödinger regularisation strength
nx  = 128;
nt = 256;
% nt  = round(vareps*nx^2);
dx  = 1/nx;
dt  = 1/nt;
x   = linspace(0, 1, nx+1);
xx  = (x(1:end-1) + x(2:end)) / 2;   % cell-centre x-coordinates

%% Algorithm parameters
gamma   = 100;
maxIter = 5000;

%% Solver selection
%   'banded'   - DCT-x + tridiagonal solve (O(N log N), stable for all eps)
%   'pcg'      - preconditioned CG (iterative, usually slower)
solver = 'banded';

%% Test-case settings
test_case         = 'gaussian';
compute_reference = false;    % run a long solve and save as reference
use_reference     = false;    % track iterates against the reference solution
postprocess       = false;    % compute cost and constraint violation each iteration
maxIter_ref       = 10000;

%% Figure style  (journal single-column, ~8.6 cm wide)
sty.lw  = 1.5;    % data line width (pt)
sty.alw = 0.75;   % axes / frame line width (pt)
sty.fs  = 10;     % font size (pt)
sty.fw  = 8.6;    % figure width (cm)  -- one journal column
sty.fh1 = 5.5;    % single-panel height (cm)
sty.fh2 = 10.0;   % two-panel height (cm)
sty.fhT = 7.0;    % transport animation height (cm)

% Tab10 / matplotlib default palette: perceptually uniform, colorblind-safe
sty.C = [0.1216  0.4667  0.7059;   % blue
         0.8392  0.1529  0.1569;   % red
         0.1725  0.6275  0.1725;   % green
         0.5804  0.4039  0.7412];  % purple

%% Solver label (for figure titles and file paths)
switch solver
    case 'banded',    solver_label = 'Banded';     solver_tex = 'Banded';
    case 'pcg',       solver_label = 'PCG';        solver_tex = 'PCG';
    otherwise,        error('Unknown solver: %s', solver);
end

%% Figure output directory
%   figures/<test_case>/nx<nx>_nt<nt>_gamma<gamma>_eps<vareps>_<solver>/
%   Example: figures/gaussian/nx128_nt64_gamma10_eps1e-6_banded/
eps_str  = regexprep(sprintf('%.0e', vareps), 'e([+-])0+(\d)', 'e$1$2');
fig_dir  = fullfile('figures', test_case, ...
               sprintf('nx%d_nt%d_gamma%g_eps%s_%s', nx, nt, gamma, eps_str, solver));
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% Boundary densities: Gaussian to Gaussian
mu0 = 1/3;   sigma0 = 0.05;
mu1 = 2/3;   sigma1 = 0.05;
gauss = @(x, mu, sig) exp(-0.5*((x - mu)/sig).^2) / (sqrt(2*pi)*sig);
rho0  = gauss(xx, mu0, sigma0);
rho1  = gauss(xx, mu1, sigma1);
rho0  = rho0 / sum(rho0);
rho1  = rho1 / sum(rho1);

%% Reference solution
data_dir = 'data';
if ~exist(data_dir, 'dir'), mkdir(data_dir); end

ref_file = fullfile(data_dir, ...
    sprintf('refsoln_%s_gamma%.1f_nx%d_nt%d_eps%s.mat', test_case, gamma, nx, nt, eps_str));

if compute_reference
    fprintf('Computing reference solution (%d iterations)...\n', maxIter_ref);
    opts_ref.nt          = nt;
    opts_ref.nx          = nx;
    opts_ref.maxIter     = maxIter_ref;
    opts_ref.gamma       = gamma;
    opts_ref.vareps      = vareps;
    opts_ref.postprocess = false;   % skip expensive metrics for the reference run

    [rho_star, mx_star, outs_ref] = sb1d_admm(rho0, rho1, opts_ref);

    save(ref_file, 'rho_star', 'mx_star', 'outs_ref', ...
         'gamma', 'nx', 'nt', 'test_case', 'rho0', 'rho1');
    fprintf('Reference solution saved to %s\n', ref_file);

    fig = new_fig(sty.fw, sty.fh1);
    semilogy(2:maxIter_ref, outs_ref.residual_diff(2:end), ...
             'Color', sty.C(1,:), 'LineWidth', sty.lw);
    style_ax(gca, sty);
    xlabel('Iteration'); ylabel('Residual');
    title(sprintf('Reference convergence (%d iterations)', maxIter_ref));
    grid on;
    save_fig(fig, fullfile(fig_dir, 'reference_convergence'));
end

%% Main solve
opts.nt          = nt;
opts.nx          = nx;
opts.maxIter     = maxIter;
opts.gamma       = gamma;
opts.vareps      = vareps;
opts.postprocess = postprocess;

if use_reference
    if compute_reference
        opts.rho_star = rho_star;   % reuse in-memory result
        opts.mx_star  = mx_star;
    else
        if ~exist(ref_file, 'file')
            error('Reference file not found: %s\nSet compute_reference = true first.', ref_file);
        end
        fprintf('Loading reference solution from %s...\n', ref_file);
        ref = load(ref_file);
        opts.rho_star = ref.rho_star;
        opts.mx_star  = ref.mx_star;
    end
end

fprintf('Running ADMM with %s projection...\n', solver_label);
switch solver
    case 'pcg'
        [rho_admm, mx_admm, outs_admm] = sb1d_admm_pcg(rho0, rho1, opts);
    otherwise
        [rho_admm, mx_admm, outs_admm] = sb1d_admm(rho0, rho1, opts);
end

%% Analytical solutions (Gaussian-to-Gaussian only)
%
%  SB marginal:  N(mu_t, sigma_sb²(t))
%    sigma_sb²(t) = (1-t)²σ₀² + 2t(1-t)·√(ε²+σ₀²σ₁²) + t²σ₁²
%    (the √(ε²+σ₀²σ₁²) cross-term encodes both the OT coupling and
%     the Brownian bridge diffusion; they combine cleanly after cancellation)
%
%  OT marginal:  N(mu_t, sigma_ot²(t))
%    sigma_ot(t) = (1-t)σ₀ + t·σ₁   (linear interpolation of std devs)
%
has_exact = strcmp(test_case, 'gaussian');
if has_exact
    tt_inner = linspace(0, 1, nt+1)';
    tt_inner = tt_inner(2:end-1);   % interior time points, length nt-1

    % SB cross-covariance: c = √(ε²+σ₀²σ₁²) − ε
    % so  ε + c = √(ε²+σ₀²σ₁²), which is the cross-term in σ_sb²
    c_sb     = sqrt(vareps^2 + sigma0^2 * sigma1^2) - vareps;

    rho_sb = zeros(nt-1, nx);
    rho_ot = zeros(nt-1, nx);
    for k = 1:nt-1
        t_k  = tt_inner(k);
        mu_t = (1 - t_k)*mu0 + t_k*mu1;

        % SB variance
        var_sb = (1 - t_k)^2 * sigma0^2 ...
               + 2*t_k*(1 - t_k) * (vareps + c_sb) ...
               + t_k^2 * sigma1^2;
        rho_k       = gauss(xx, mu_t, sqrt(var_sb));
        rho_sb(k,:) = rho_k / sum(rho_k);

        % OT variance  (ε = 0 limit: linear interpolation of std devs)
        sig_ot      = (1 - t_k)*sigma0 + t_k*sigma1;
        rho_k       = gauss(xx, mu_t, sig_ot);
        rho_ot(k,:) = rho_k / sum(rho_k);
    end
end

%% Transport animation
fig = new_fig(sty.fw, sty.fhT);
ax  = gca;
hold(ax, 'on');
h = plot(ax, xx, rho0, '-',  'Color', sty.C(1,:), 'LineWidth', sty.lw);
if has_exact
    h_sb = plot(ax, xx, rho0, '--', 'Color', sty.C(2,:), 'LineWidth', sty.lw);
    h_ot = plot(ax, xx, rho0, ':',  'Color', sty.C(3,:), 'LineWidth', sty.lw);
    legend(ax, sprintf('$\\rho_{\\mathrm{ADMM}}$ (%s)', solver_tex), ...
               '$\rho_{\mathrm{SB}}$ (exact)', ...
               '$\rho_{\mathrm{OT}}$ (exact)', ...
               'Location', 'southoutside', 'Orientation', 'horizontal');
end
hold(ax, 'off');
style_ax(ax, sty);
ylim([0, max(rho_admm(:)) * 1.1]);
xlabel('$x$'); ylabel('$\rho(t,x)$');

vid_path = fullfile(fig_dir, 'transport.mp4');
v = VideoWriter(vid_path, 'MPEG-4');
v.FrameRate = 5;
open(v);
for k = 1:nt-1
    set(h, 'YData', rho_admm(k, :));
    if has_exact
        set(h_sb, 'YData', rho_sb(k, :));
        set(h_ot, 'YData', rho_ot(k, :));
    end
    title(sprintf('Density transport, $t = %d/%d$ (%s)', k, nt, solver_tex));
    drawnow;
    writeVideo(v, getframe(fig));
end
close(v);
fprintf('Video saved: %s\n', vid_path);

%% Convergence figure
has_true_error = use_reference && isfield(outs_admm, 'true_error');

if has_true_error
    fig = new_fig(sty.fw, sty.fh2);
    tl  = tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    semilogy(2:maxIter, outs_admm.residual_diff(2:end), ...
             'Color', sty.C(1,:), 'LineWidth', sty.lw);
    style_ax(gca, sty);
    xlabel('Iteration'); ylabel('$\|u_{k+1} - u_k\|$');
    title(sprintf('Successive difference (%s)', solver_tex));
    grid on;

    nexttile;
    semilogy(1:maxIter, outs_admm.true_error, ...
             'Color', sty.C(2,:), 'LineWidth', sty.lw);
    style_ax(gca, sty);
    xlabel('Iteration'); ylabel('$\|u_k - u^*\|$');
    title(sprintf('Error vs.\\ reference solution (%s)', solver_tex));
    grid on;
else
    fig = new_fig(sty.fw, sty.fh1);
    semilogy(2:maxIter, outs_admm.residual_diff(2:end), ...
             'Color', sty.C(1,:), 'LineWidth', sty.lw);
    style_ax(gca, sty);
    xlabel('Iteration'); ylabel('$\|u_{k+1} - u_k\|$');
    title(sprintf('Successive difference (%s)', solver_tex));
    grid on;
end
save_fig(fig, fullfile(fig_dir, 'convergence'));

%% Post-processing figure
if isfield(outs_admm, 'cost')
    fig = new_fig(sty.fw, sty.fh2);
    tl  = tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

    % Use deviation from converged value so the log scale reveals the
    % convergence rate rather than a flat line at J*.
    J_star   = outs_admm.cost(end);
    cost_dev = abs(outs_admm.cost - J_star);
    cost_dev(cost_dev == 0) = eps;   % guard against exact zeros on log scale

    nexttile;
    semilogy(1:maxIter, cost_dev, ...
             'Color', sty.C(1,:), 'LineWidth', sty.lw);
    style_ax(gca, sty);
    xlabel('Iteration');
    ylabel('$|J_k - J^*|$');
    title(sprintf('Kinetic energy convergence (%s)', solver_tex));
    % Annotate the converged value in the top-right corner
    text(0.97, 0.92, sprintf('$J^* \\approx %.4g$', J_star), ...
         'Units', 'normalized', 'HorizontalAlignment', 'right', ...
         'VerticalAlignment', 'top', 'FontSize', sty.fs);
    grid on;

    nexttile;
    semilogy(1:maxIter, outs_admm.constraint_viol, ...
             'Color', sty.C(2,:), 'LineWidth', sty.lw);
    style_ax(gca, sty);
    xlabel('Iteration');
    ylabel('$\|\partial_t\rho + \partial_x m - \varepsilon\Delta\rho\|_{L^2}$');
    title(sprintf('Fokker--Planck constraint violation (%s)', solver_tex));
    grid on;

    save_fig(fig, fullfile(fig_dir, 'postprocess'));
end

%% ====================================================================
%% Local functions
%% ====================================================================

function fig = new_fig(fw, fh)
    % Create a figure pre-configured for journal export (dimensions in cm).
    fig = figure('Units',         'centimeters', ...
                 'Position',      [5 5 fw fh],   ...
                 'PaperUnits',    'centimeters',  ...
                 'PaperSize',     [fw fh],        ...
                 'PaperPosition', [0 0 fw fh],    ...
                 'Color',         'white');
end

function style_ax(ax, sty)
    % Apply journal-quality styling to an axes object.
    set(ax, 'FontSize',       sty.fs,  ...
            'LineWidth',      sty.alw, ...
            'TickDir',        'out',   ...
            'Box',            'off',   ...
            'GridAlpha',      0.15,    ...
            'MinorGridAlpha', 0.05);
end

function save_fig(fig, path)
    % Export figure as vector PDF and 300 dpi PNG.
    exportgraphics(fig, [path '.pdf'], ...
                   'ContentType',     'vector', ...
                   'BackgroundColor', 'white');
    exportgraphics(fig, [path '.png'], ...
                   'Resolution',      300, ...
                   'BackgroundColor', 'white');
    fprintf('Saved: %s (.pdf, .png)\n', path);
end
