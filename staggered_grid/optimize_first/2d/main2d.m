%% Main script: 2-D Schrödinger Bridge via ADMM
clear;

%% LaTeX renderer
set(groot, 'defaultTextInterpreter',         'latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter',       'latex');

%% GPU toggle
use_gpu = false;   % set true to use GPU (requires R2022a+ for dct on gpuArray)

%% Grid
vareps = 1e-4;     % Schrödinger regularisation strength
nx     = 64;       % x-space intervals
ny     = 64;       % y-space intervals
nt     = 64;       % time intervals

dx = 1/nx;  dy = 1/ny;  dt = 1/nt;
xx = (0.5:nx-0.5) * dx;   % cell-centre x-coordinates
yy = (0.5:ny-0.5) * dy;   % cell-centre y-coordinates
[XX, YY] = meshgrid(xx, yy);   % ny x nx  (meshgrid convention)

%% Algorithm parameters
gamma   = 1;
maxIter = 3000;

%% Boundary densities: Gaussian to Gaussian in 2-D
mu0x = 1/3;  mu0y = 1/2;  sigma0 = 0.07;
mu1x = 2/3;  mu1y = 1/2;  sigma1 = 0.07;

gauss2d = @(X, Y, mux, muy, sig) ...
    exp(-0.5*((X - mux).^2 + (Y - muy).^2) / sig^2) / (2*pi*sig^2);

% rho0, rho1 are (nx x ny)  (transpose of meshgrid output)
rho0 = gauss2d(XX, YY, mu0x, mu0y, sigma0)';   % nx x ny
rho1 = gauss2d(XX, YY, mu1x, mu1y, sigma1)';

% Normalise to discrete probability: sum(rho(:)) = 1  (matches 1-D convention)
rho0 = rho0 / sum(rho0(:));
rho1 = rho1 / sum(rho1(:));

%% Figure style  (journal single-column)
sty.lw  = 1.5;
sty.alw = 0.75;
sty.fs  = 10;
sty.fw  = 8.6;    % cm
sty.fh1 = 6.0;
sty.fhT = 8.0;
sty.C   = [0.1216  0.4667  0.7059;
           0.8392  0.1529  0.1569;
           0.1725  0.6275  0.1725;
           0.5804  0.4039  0.7412];

%% Output directory
eps_str = regexprep(sprintf('%.0e', vareps), 'e([+-])0+(\d)', 'e$1$2');
fig_dir = fullfile('figures', ...
    sprintf('nx%d_ny%d_nt%d_gamma%g_eps%s', nx, ny, nt, gamma, eps_str));
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% Solve
opts.nt          = nt;
opts.maxIter     = maxIter;
opts.gamma       = gamma;
opts.vareps      = vareps;
opts.use_gpu     = use_gpu;
opts.postprocess = true;

fprintf('Running 2-D SB-ADMM: nx=%d  ny=%d  nt=%d  eps=%g  gamma=%g\n', ...
        nx, ny, nt, vareps, gamma);
tic;
[rho_admm, mx_admm, my_admm, outs_admm] = sb2d_admm(rho0, rho1, opts);
t_solve = toc;
fprintf('Solve time: %.1f s\n', t_solve);

%% Density animation (imagesc, one frame per time slice)
fig = new_fig(sty.fw, sty.fhT);
ax  = gca;
clim_max = max(rho_admm(:)) * 1.05;

% Prepend rho0, interior slices, append rho1 for the full transport sequence
h_img = imagesc(ax, xx, yy, rho0');   % ny x nx for imagesc (row = y, col = x)
colormap(ax, 'hot');
clim(ax, [0, clim_max]);
colorbar(ax);
axis(ax, 'xy');   % y-axis pointing up
axis(ax, 'equal');
xlim(ax, [0 1]);  ylim(ax, [0 1]);
style_ax(ax, sty);
xlabel(ax, '$x$');  ylabel(ax, '$y$');

vid_path = fullfile(fig_dir, 'transport.mp4');
v = VideoWriter(vid_path, 'MPEG-4');
v.FrameRate = 8;
open(v);

% t=0 boundary
title(ax, '$\rho(t=0)$', 'FontSize', sty.fs);
drawnow;  writeVideo(v, getframe(fig));

% Interior time slices
for k = 1:nt-1
    set(h_img, 'CData', squeeze(rho_admm(k,:,:))');   % ny x nx
    title(ax, sprintf('$\\rho(t=%d/%d)$', k, nt), 'FontSize', sty.fs);
    drawnow;  writeVideo(v, getframe(fig));
end

% t=1 boundary
set(h_img, 'CData', rho1');
title(ax, '$\rho(t=1)$', 'FontSize', sty.fs);
drawnow;  writeVideo(v, getframe(fig));
close(v);
fprintf('Animation saved: %s\n', vid_path);

%% Convergence figure
fig = new_fig(sty.fw, sty.fh1);
semilogy(2:maxIter, outs_admm.residual_diff(2:end), ...
         'Color', sty.C(1,:), 'LineWidth', sty.lw);
style_ax(gca, sty);
xlabel('Iteration');  ylabel('$\|u_{k+1}-u_k\|$');
title('Successive difference (convergence)');
grid on;
save_fig(fig, fullfile(fig_dir, 'convergence'));

%% Post-processing figure
if isfield(outs_admm, 'cost')
    fig = new_fig(sty.fw, 2*sty.fh1);
    tl  = tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

    J_star   = outs_admm.cost(end);
    cost_dev = abs(outs_admm.cost - J_star);
    cost_dev(cost_dev == 0) = eps;

    nexttile;
    semilogy(1:maxIter, cost_dev, 'Color', sty.C(1,:), 'LineWidth', sty.lw);
    style_ax(gca, sty);
    xlabel('Iteration');  ylabel('$|J_k - J^*|$');
    title('Kinetic energy convergence');
    text(0.97, 0.92, sprintf('$J^* \\approx %.4g$', J_star), ...
         'Units','normalized','HorizontalAlignment','right', ...
         'VerticalAlignment','top','FontSize',sty.fs);
    grid on;

    nexttile;
    semilogy(1:maxIter, outs_admm.constraint_viol, ...
             'Color', sty.C(2,:), 'LineWidth', sty.lw);
    style_ax(gca, sty);
    xlabel('Iteration');
    ylabel('$\|\partial_t\rho + \nabla\cdot m - \varepsilon\Delta\rho\|_{L^2}$');
    title('Fokker--Planck constraint violation');
    grid on;

    save_fig(fig, fullfile(fig_dir, 'postprocess'));
end

%% Snapshots at selected time slices
t_snaps = [0.25, 0.5, 0.75];
fig = new_fig(sty.fw * numel(t_snaps), sty.fhT);
tl  = tiledlayout(1, numel(t_snaps), 'Padding', 'compact', 'TileSpacing', 'compact');

for kk = 1:numel(t_snaps)
    k_idx = round(t_snaps(kk) * (nt-1));
    k_idx = max(1, min(k_idx, nt-1));

    nexttile;
    ax2 = gca;
    imagesc(ax2, xx, yy, squeeze(rho_admm(k_idx,:,:))');
    colormap(ax2, 'hot');
    clim(ax2, [0, clim_max]);
    colorbar(ax2);
    axis(ax2, 'xy');  axis(ax2, 'equal');
    xlim(ax2, [0 1]);  ylim(ax2, [0 1]);
    style_ax(ax2, sty);
    xlabel(ax2, '$x$');  ylabel(ax2, '$y$');
    title(ax2, sprintf('$t = %.2f$', t_snaps(kk)));
end
save_fig(fig, fullfile(fig_dir, 'snapshots'));


%% ====================================================================
%% Local functions
%% ====================================================================

function fig = new_fig(fw, fh)
    fig = figure('Units','centimeters', 'Position',[5 5 fw fh], ...
                 'PaperUnits','centimeters', 'PaperSize',[fw fh], ...
                 'PaperPosition',[0 0 fw fh], 'Color','white');
end

function style_ax(ax, sty)
    set(ax, 'FontSize', sty.fs, 'LineWidth', sty.alw, ...
            'TickDir', 'out', 'Box', 'off', ...
            'GridAlpha', 0.15, 'MinorGridAlpha', 0.05);
end

function save_fig(fig, path)
    exportgraphics(fig, [path '.pdf'], 'ContentType','vector', ...
                   'BackgroundColor','white');
    exportgraphics(fig, [path '.png'], 'Resolution',300, ...
                   'BackgroundColor','white');
    fprintf('Saved: %s (.pdf, .png)\n', path);
end
