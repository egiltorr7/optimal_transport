% POSTPROCESS_SB_GAUSSIAN  Load a saved result and generate all figures locally.
%
%   Run from experiments/ after pulling the .mat file from the remote machine:
%
%     rsync user@remote:.../results/result_*.mat ./results/
%
%   Then run this script.  Set fields in the 'sel' struct to filter which
%   result to load (nt, nx, ny, eps, gam, tau).  Leave a field empty ([])
%   to match any value.  If multiple files match, the most recent is used.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
fig_dir = fullfile(res_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Select result ---
% Set any field to filter; leave empty ([]) to match anything.
% If multiple files match, the most recent is loaded.
sel.nt  = 64;      % e.g. 64
sel.nx  = 128;      % e.g. 128
sel.ny  = 128;      %e.g. 128
sel.eps = 1e-1;      % e.g. 0.01
sel.gam = 0.1;      % e.g. 100
sel.tau = 0.11;      % e.g. 101

mats = dir(fullfile(res_dir, 'result_*.mat'));
if isempty(mats)
    error('No result_*.mat files found in %s', res_dir);
end

keep = true(numel(mats), 1);
tokens = {'nt','nx','ny','gam','tau','eps'};
for i = 1:numel(mats)
    for j = 1:numel(tokens)
        f   = tokens{j};
        val = sel.(f);
        if isempty(val), continue; end
        tok = regexp(mats(i).name, [f '([\d.eE+-]+)'], 'tokens', 'once');
        if isempty(tok) || abs(str2double(tok{1}) - val) > 1e-10*(abs(val)+1)
            keep(i) = false;  break;
        end
    end
end

mats = mats(keep);
if isempty(mats)
    error('No result_*.mat files match the selection criteria.');
end
if numel(mats) > 1
    fprintf('Multiple matches — loading most recent:\n');
    for i = 1:numel(mats), fprintf('  %s\n', mats(i).name); end
end
[~, idx] = max([mats.datenum]);
MAT_FILE = fullfile(res_dir, mats(idx).name);

fprintf('Loading %s ...\n', MAT_FILE);
load(MAT_FILE, 'result', 'cfg', 'problem', 'rho_ana_cc', 'obj', 'ftag');

nt = problem.nt;  dt = problem.dt;
nx = problem.nx;  dx = problem.dx;
ny = problem.ny;  dy = problem.dy;
xx = problem.xx;  yy = problem.yy;
ops  = problem.ops;
rho0 = problem.rho0;
rho1 = problem.rho1;

rho_num_cc = result.rho_cc;

fprintf('  Grid: nt=%d  nx=%d  ny=%d  eps=%.4g\n', nt, nx, ny, cfg.vareps);
fprintf('  iters=%d  converged=%d  error=%.2e  wall=%.1fs\n', ...
    result.iters, result.converged, result.error, result.walltime);
if isfield(result, 'time_per_iter')
    fprintf('  time/iter=%.3fs  throughput=%.1f iter/s\n', ...
        result.time_per_iter, result.throughput);
end
if isfield(result, 'gpu_total_mb') && result.gpu_total_mb > 0
    fprintf('  GPU mem: %.0f / %.0f MB used/total\n', ...
        result.gpu_mem_post_mb, result.gpu_total_mb);
end
fprintf('  Kinetic energy: %.6f\n', obj);

savefig = @(fig, name) exportgraphics(fig, ...
    fullfile(fig_dir, sprintf('%s_%s.png', name, ftag)), 'Resolution', 150);

t_snap  = [0.1, 0.3, 0.5, 0.7, 0.9];
n_snap  = numel(t_snap);
k_snap  = max(1, round(t_snap * nt));
colors  = parula(n_snap);
clim_max = max(rho_ana_cc(:));

% -------------------------------------------------------------------------
%% Figure 1: density heatmaps -- analytical vs numerical
% -------------------------------------------------------------------------
fig1 = figure('Name', 'Density snapshots', 'Position', [50 50 1200 500]);
for p = 1:n_snap
    k = k_snap(p);
    subplot(2, n_snap, p);
    imagesc(xx, yy, squeeze(rho_ana_cc(k,:,:))');
    axis xy; colorbar; clim([0 clim_max]);
    title(sprintf('Ana  t=%.2f', (k-0.5)*dt));
    xlabel('x'); ylabel('y');

    subplot(2, n_snap, n_snap + p);
    imagesc(xx, yy, squeeze(rho_num_cc(k,:,:))');
    axis xy; colorbar; clim([0 clim_max]);
    title(sprintf('LADMM  t=%.2f', (k-0.5)*dt));
    xlabel('x'); ylabel('y');
end
sgtitle(sprintf('Density  eps=%.4g  iters=%d', cfg.vareps, result.iters));
savefig(fig1, 'density');

% -------------------------------------------------------------------------
%% Figure 2: x-slice at y = 0.5 vs analytical
% -------------------------------------------------------------------------
[~, iy] = min(abs(yy - 0.5));

fig2 = figure('Name', 'x-slice at y=0.5', 'Position', [50 600 700 400]);
hold on;
leg_handles = gobjects(n_snap, 2);
for p = 1:n_snap
    k = k_snap(p);
    leg_handles(p,1) = plot(xx, rho_ana_cc(k,:,iy), '-',  'Color', colors(p,:), 'LineWidth', 1.5);
    leg_handles(p,2) = plot(xx, rho_num_cc(k,:,iy), 'o--','Color', colors(p,:), 'MarkerSize', 4);
end
leg_str = cell(n_snap, 2);
for p = 1:n_snap
    k = k_snap(p);
    leg_str{p,1} = sprintf('Ana   t=%.2f', (k-0.5)*dt);
    leg_str{p,2} = sprintf('LADMM t=%.2f', (k-0.5)*dt);
end
legend(leg_handles(:), leg_str(:), 'Location', 'best', 'FontSize', 7);
xlabel('x'); ylabel('\rho(t,x,y=0.5)');
title(sprintf('x-slice at y=0.5   eps=%.4g', cfg.vareps));
grid on;
savefig(fig2, 'xslice');

% -------------------------------------------------------------------------
%% Figure 2b: diagonal slice along y = x (connects the two Gaussian centres)
% -------------------------------------------------------------------------
% Only meaningful when nx == ny (diagonal indices i==j)
if nx == ny
    % For each snapshot extract rho along the main diagonal i==j
    fig2b = figure('Name', 'Diagonal slice y=x', 'Position', [50 1050 700 400]);
    hold on;
    leg_handles2 = gobjects(n_snap, 2);
    for p = 1:n_snap
        k = k_snap(p);
        diag_ana = diag(squeeze(rho_ana_cc(k,:,:)));   % (nx x 1)
        diag_num = diag(squeeze(rho_num_cc(k,:,:)));
        leg_handles2(p,1) = plot(xx, diag_ana, '-',  'Color', colors(p,:), 'LineWidth', 1.5);
        leg_handles2(p,2) = plot(xx, diag_num, 'o--','Color', colors(p,:), 'MarkerSize', 4);
    end
    leg_str2 = cell(n_snap, 2);
    for p = 1:n_snap
        k = k_snap(p);
        leg_str2{p,1} = sprintf('Ana   t=%.2f', (k-0.5)*dt);
        leg_str2{p,2} = sprintf('LADMM t=%.2f', (k-0.5)*dt);
    end
    legend(leg_handles2(:), leg_str2(:), 'Location', 'best', 'FontSize', 7);
    xlabel('x  (along diagonal y=x)');
    ylabel('\rho(t,x,y=x)');
    title(sprintf('Diagonal slice y=x   eps=%.4g', cfg.vareps));
    grid on;
    savefig(fig2b, 'diagslice');
end

% -------------------------------------------------------------------------
%% Figure 3: momentum magnitude heatmaps + quiver at selected times
% -------------------------------------------------------------------------
zeros_x = zeros(nt, ny);
zeros_y = zeros(nt, nx);

mx_cc = result.mx_cc;   % (nt x nx x ny)  cell-centre momentum
my_cc = result.my_cc;

n_quiv = 3;
k_quiv = max(1, round([0.25, 0.5, 0.75] * nt));
stride = max(1, floor(min(nx, ny) / 16));   % thin out quiver arrows

fig3 = figure('Name', 'Momentum field', 'Position', [800 50 900 300]);
for p = 1:n_quiv
    k = k_quiv(p);
    m_mag = sqrt(squeeze(mx_cc(k,:,:)).^2 + squeeze(my_cc(k,:,:)).^2);

    subplot(1, n_quiv, p);
    imagesc(xx, yy, m_mag');
    axis xy; colorbar;
    hold on;

    % Quiver on coarser grid
    xi = 1:stride:nx;   yi = 1:stride:ny;
    [Xq, Yq] = meshgrid(xx(xi), yy(yi));
    Uq = squeeze(mx_cc(k, xi, yi))';
    Vq = squeeze(my_cc(k, xi, yi))';
    quiver(Xq, Yq, Uq, Vq, 0.8, 'w', 'LineWidth', 0.8);

    title(sprintf('|m|  t=%.2f', (k-0.5)*dt));
    xlabel('x'); ylabel('y');
end
sgtitle(sprintf('Momentum magnitude + direction   eps=%.4g', cfg.vareps));
savefig(fig3, 'momentum');

% -------------------------------------------------------------------------
%% Figure 4: FP residual over domain (staggered output x)
% -------------------------------------------------------------------------
x_stag.rho = result.rho_stag;
x_stag.mx  = result.mx_stag;
x_stag.my  = result.my_stag;

rho_phi   = ops.interp_t_at_phi(x_stag.rho, rho0, rho1);
nabla_rho = ops.deriv_x_at_phi(ops.deriv_x_at_m(rho_phi), zeros_x, zeros_x) ...
          + ops.deriv_y_at_phi(ops.deriv_y_at_m(rho_phi), zeros_y, zeros_y);
fp_res = ops.deriv_t_at_phi(x_stag.rho, rho0, rho1) ...
       + ops.deriv_x_at_phi(x_stag.mx, zeros_x, zeros_x) ...
       + ops.deriv_y_at_phi(x_stag.my, zeros_y, zeros_y) ...
       - cfg.vareps * nabla_rho;   % (nt x nx x ny)

fp_per_time = squeeze(max(max(abs(fp_res), [], 2), [], 3));   % (nt x 1)
t_phi = ((1:nt)' - 0.5) * dt;

fig4 = figure('Name', 'FP residual', 'Position', [50 200 600 300]);
semilogy(t_phi, fp_per_time, 'b-', 'LineWidth', 1.5);
xlabel('t'); ylabel('max_{x,y} |FP residual|');
title(sprintf('FP constraint residual (max over space)   max=%.2e', max(fp_per_time)));
grid on;
savefig(fig4, 'fp_residual');

% -------------------------------------------------------------------------
%% Figure 5: ADMM convergence
% -------------------------------------------------------------------------
fig5 = figure('Name', 'ADMM residual', 'Position', [700 200 600 300]);
semilogy(result.residual, 'b-', 'LineWidth', 1.5);
yline(cfg.tol, 'r--', sprintf('tol = %.1e', cfg.tol));
xlabel('Iteration'); ylabel('||y^{k+1} - y^k||');
title(sprintf('ADMM convergence   iters=%d  converged=%d', result.iters, result.converged));
grid on;
savefig(fig5, 'admm_residual');

% -------------------------------------------------------------------------
%% Figure 6: L2 error in rho vs analytical over time
% -------------------------------------------------------------------------
l2_err = sqrt(dx * dy * sum(sum((rho_num_cc - rho_ana_cc).^2, 2), 3));
t_cc   = ((1:nt)' - 0.5) * dt;

fig6 = figure('Name', 'L2 error', 'Position', [700 550 600 300]);
plot(t_cc, l2_err, 'b-', 'LineWidth', 1.5);
xlabel('t'); ylabel('||\rho_{num} - \rho_{ana}||_{L^2(x,y)}');
title(sprintf('L^2 error vs analytical SB   mean=%.2e', mean(l2_err)));
grid on;
savefig(fig6, 'l2error');

% -------------------------------------------------------------------------
%% Figure 7: mass conservation  sum_xy rho(t) * dx * dy  vs time
% -------------------------------------------------------------------------
mass = squeeze(sum(sum(rho_num_cc, 2), 3)) * dx * dy;   % (nt x 1)

fig7 = figure('Name', 'Mass conservation', 'Position', [50 550 600 300]);
plot(t_cc, mass, 'b-', 'LineWidth', 1.5);
yline(mass(1), 'r--', sprintf('initial = %.4f', mass(1)));
xlabel('t'); ylabel('\int\int \rho\, dx\, dy');
title(sprintf('Mass conservation   drift = %.2e', max(mass) - min(mass)));
grid on;
savefig(fig7, 'mass_conservation');

% -------------------------------------------------------------------------
fprintf('\nFigures saved to %s\n', fig_dir);
