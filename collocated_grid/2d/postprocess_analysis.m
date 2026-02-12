%% Post-Processing Script for Optimal Transport Analysis
% This script analyzes data from main_simple.m and creates:
% 1. Error analysis plots (true error and residual difference vs iterations)
% 2. Animation showing dynamic transportation between Gaussians

clear; close all;

%% Configuration
folder = "./data/";
figures_folder = "./figures/";
case_name = "gaussian_";

% Grid parameters (should match main_simple.m)
tau = 0.1;
nt = 63;
nx = 127;
ny = nx;

grid_name = sprintf('tau%0.2f_nt%d_nx%d_ny%d', tau, nt, nx, ny);
file_name = folder + case_name + grid_name;

% Animation settings
animate = true;
save_animation = true;  % Set to true to save animation as GIF
animation_file = figures_folder + "transport_animation_" + grid_name + ".gif";
fps = 10;  % Frames per second for animation

%% Load Data
fprintf('Loading data from %s...\n', file_name);
data = load(file_name + ".mat");
post_data = load(file_name + "_post.mat");

rho_admm = data.rho_admm;
mx_admm = data.mx_admm;
my_admm = data.my_admm;
maxIter = data.maxIter;

iterations = post_data.iterations;
residual_diff = post_data.outs_admm.residual_diff;
residual_numerical = post_data.outs_admm.residual_numerical;

% Trim to actual iterations used
iterations = iterations(1:maxIter);
residual_diff = residual_diff(1:maxIter);
residual_numerical = residual_numerical(1:maxIter);

fprintf('Data loaded successfully. Total iterations: %d\n', maxIter);

%% Setup spatial and temporal grid
ntp = nt + 1;
nxp = nx + 1;
nyp = ny + 1;
dx = 1/nxp;
dy = 1/nyp;
dt = 1/ntp;

% Spatial grids
x = (0:nxp-1) * dx;  % x_j = j*dx, j = 0,1,...,nxp-1
y = (0:nyp-1) * dy;  % y_k = k*dy, k = 0,1,...,nyp-1

% Temporal grids
t_rho = (0:nt-1) * dt;  % t_i = i*dt for rho, i = 0,1,...,nt-1
t_m = (0:nt) * dt;      % t_i = i*dt for mx,my, i = 0,1,...,nt

[X, Y] = meshgrid(y, x);  % Note: meshgrid uses (y,x) ordering

%% Error Analysis Plots

% Figure 1: Cumulative Error (Residual Difference)
figure('Position', [100, 100, 800, 600]);
semilogy(iterations, residual_diff, 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);
grid on;
xlabel('Iteration $k$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\| u^{k+1} - u^k \|_2$', 'Interpreter', 'latex', 'FontSize', 14);
title('Cumulative Error (Residual Difference)', 'Interpreter', 'latex', 'FontSize', 16);
set(gca, 'FontSize', 12);
xlim([1, maxIter]);

% Save figure
saveas(gcf, figures_folder + "error_residual_diff_" + grid_name + ".png");
fprintf('Saved: error_residual_diff_%s.png\n', grid_name);

% Figure 2: True Numerical Error (excluding last point where reference solution was obtained)
figure('Position', [150, 150, 800, 600]);
marker_interval = max(1, floor((maxIter-1)/50));  % Show ~50 markers
semilogy(iterations(1:end-1), residual_numerical(1:end-1), '-o', 'LineWidth', 2, ...
    'Color', [0.8, 0.2, 0.2], 'MarkerIndices', 1:marker_interval:length(iterations)-1, ...
    'MarkerSize', 6, 'MarkerFaceColor', [0.8, 0.2, 0.2]);
grid on;
xlabel('Iteration $k$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\| u^k - u^* \|_2$', 'Interpreter', 'latex', 'FontSize', 14);
title('True Numerical Error', 'Interpreter', 'latex', 'FontSize', 16);
set(gca, 'FontSize', 12);
xlim([1, maxIter-1]);

% Save figure
saveas(gcf, figures_folder + "error_numerical_" + grid_name + ".png");
fprintf('Saved: error_numerical_%s.png\n', grid_name);

% Figure 3: Combined Error Plot
figure('Position', [200, 200, 800, 600]);
marker_interval_combined = max(1, floor(maxIter/50));  % Show ~50 markers
semilogy(iterations, residual_diff, 'LineWidth', 2, ...
    'DisplayName', '$\| u^{k+1} - u^k \|_2$ (Residual)', 'Color', [0.2, 0.4, 0.8]);
hold on;
% Exclude last point from true error (where reference solution was obtained)
semilogy(iterations(1:end-1), residual_numerical(1:end-1), '-o', 'LineWidth', 2, ...
    'DisplayName', '$\| u^k - u^* \|_2$ (True Error)', 'Color', [0.8, 0.2, 0.2], ...
    'MarkerIndices', 1:marker_interval_combined:length(iterations)-1, ...
    'MarkerSize', 6, 'MarkerFaceColor', [0.8, 0.2, 0.2]);
grid on;
xlabel('Iteration $k$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Error', 'Interpreter', 'latex', 'FontSize', 14);
title('Error Analysis: ADMM Convergence', 'Interpreter', 'latex', 'FontSize', 16);
legend('Interpreter', 'latex', 'FontSize', 12, 'Location', 'best');
set(gca, 'FontSize', 12);
xlim([1, maxIter]);

% Save figure
saveas(gcf, figures_folder + "error_combined_" + grid_name + ".png");
fprintf('Saved: error_combined_%s.png\n', grid_name);

%% Compute velocity field from momentum
% v = m/rho (handle division by zero carefully)
% Forward differencing staggered grid:
% - rho: x_j = j*dx (j=0,...,nx), y_k = k*dy (k=0,...,ny) -> size (nt, nxp, nyp)
% - mx:  x_j = j*dx (j=0,...,nx-1), y_k = k*dy (k=0,...,ny) -> size (ntp, nx, nyp)
% - my:  x_j = j*dx (j=0,...,nx), y_k = k*dy (k=0,...,ny-1) -> size (ntp, nxp, ny)
% mx and rho share the same points at (j=0:nx-1, k=0:ny)
% my and rho share the same points at (j=0:nx, k=0:ny-1)
% Boundary conditions: at t=ntp, rho=rho1; missing spatial points have v·n=0

% Reconstruct rho0 and rho1 from problem setup
rho0 = zeros(nxp, nyp);
rho1 = zeros(nxp, nyp);
rho0((x'-0.25).^2 + (y-0.25).^2 < 0.01) = 1;
rho1((x'-0.75).^2 + (y-0.75).^2 < 0.01) = 1;
rho0_mass = sum(rho0, 'all') * dx * dy;
rho1_mass = sum(rho1, 'all') * dx * dy;
rho1 = rho1 * (rho0_mass / rho1_mass);  % Normalize mass

vx_admm = zeros(ntp, nx, nyp);
vy_admm = zeros(ntp, nxp, ny);

% Compute velocity for interior times (tt = 1 to nt)
for tt = 1:nt
    % For vx = mx/rho: they share the same spatial points!
    safe_rho_x = max(rho_admm(tt, 1:nx, :), 1e-12);
    vx_admm(tt, :, :) = mx_admm(tt, :, :) ./ safe_rho_x;

    % For vy = my/rho: they share the same spatial points!
    safe_rho_y = max(rho_admm(tt, :, 1:ny), 1e-12);
    vy_admm(tt, :, :) = my_admm(tt, :, :) ./ safe_rho_y;
end

% For the final time point (ntp), use rho1 boundary condition
safe_rho_x = max(rho1(1:nx, :), 1e-12);
vx_admm(ntp, :, :) = squeeze(mx_admm(ntp, :, :)) ./ safe_rho_x;

safe_rho_y = max(rho1(:, 1:ny), 1e-12);
vy_admm(ntp, :, :) = squeeze(my_admm(ntp, :, :)) ./ safe_rho_y;

%% Animation of Dynamic Transport

if animate
    fprintf('\nCreating animation...\n');

    % Setup figure for animation
    fig_anim = figure('Position', [250, 250, 900, 700]);

    % Find common color scale for density
    rho_min = 0;
    rho_max = max(rho_admm(:));

    % Animation loop - iterate over all ntp time points
    for tt = 1:ntp
        clf;

        % Get density at current time
        if tt <= nt
            rho_current = squeeze(rho_admm(tt, :, :));
            current_time = t_rho(tt);
        else
            % Final time point: use rho1 boundary condition
            rho_current = rho1;
            current_time = t_m(tt);
        end

        % Plot density as heatmap
        subplot(1, 2, 1);
        imagesc(y, x, rho_current);
        set(gca, 'YDir', 'normal');
        colorbar;
        caxis([rho_min, rho_max]);
        colormap('hot');
        xlabel('$y$', 'Interpreter', 'latex', 'FontSize', 14);
        ylabel('$x$', 'Interpreter', 'latex', 'FontSize', 14);
        title(sprintf('Density $\\rho(t=%.3f)$', current_time), ...
            'Interpreter', 'latex', 'FontSize', 14);
        axis square;

        % Plot velocity field (subsampled for clarity)
        subplot(1, 2, 2);
        imagesc(y, x, rho_current);
        set(gca, 'YDir', 'normal');
        colorbar;
        caxis([rho_min, rho_max]);
        colormap('hot');
        hold on;

        % Get velocity at current time
        vx_current = squeeze(vx_admm(tt, :, :));  % size (nx, nyp)
        vy_current = squeeze(vy_admm(tt, :, :));  % size (nxp, ny)

        % For visualization, average vx and vy to get velocity on the common interior grid
        % Common grid is (nx, ny) where both vx and vy are defined
        v_x_interior = vx_current(:, 1:ny);  % (nx, ny)
        v_y_interior = vy_current(1:nx, :);  % (nx, ny)

        % Subsample for quiver plot
        skip = max(1, floor(min(nx, ny) / 20));  % Show ~20 arrows per dimension
        x_plot = x(1:skip:end-1);  % Use x points where velocity is defined (0 to (nx-1)*dx)
        y_plot = y(1:skip:end-1);  % Use y points where velocity is defined (0 to (ny-1)*dy)
        [Y_sub, X_sub] = meshgrid(y_plot, x_plot);
        vx_sub = v_x_interior(1:skip:end, 1:skip:end);
        vy_sub = v_y_interior(1:skip:end, 1:skip:end);

        % Plot velocity vectors
        quiver(Y_sub, X_sub, vy_sub, vx_sub, 1.5, 'w', 'LineWidth', 1.5);

        xlabel('$y$', 'Interpreter', 'latex', 'FontSize', 14);
        ylabel('$x$', 'Interpreter', 'latex', 'FontSize', 14);
        title(sprintf('Density + Velocity $(t=%.3f)$', current_time), ...
            'Interpreter', 'latex', 'FontSize', 14);
        axis square;
        hold off;

        sgtitle(sprintf('Optimal Transport Dynamics (t = %.3f)', current_time), ...
            'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold');

        drawnow;

        % Save frame to GIF
        if save_animation
            frame = getframe(fig_anim);
            im = frame2im(frame);
            [imind, cm] = rgb2ind(im, 256);

            if tt == 1
                imwrite(imind, cm, animation_file, 'gif', ...
                    'Loopcount', inf, 'DelayTime', 1/fps);
            else
                imwrite(imind, cm, animation_file, 'gif', ...
                    'WriteMode', 'append', 'DelayTime', 1/fps);
            end
        end

        fprintf('Frame %d/%d (t = %.3f)\n', tt, ntp, current_time);
    end

    if save_animation
        fprintf('\nAnimation saved to: %s\n', animation_file);
    end
end

%% Summary Statistics
fprintf('\n=== Summary Statistics ===\n');
fprintf('Grid: nt=%d, nx=%d, ny=%d\n', nt, nx, ny);
fprintf('Total iterations: %d\n', maxIter);
fprintf('\n--- Convergence ---\n');
fprintf('Final residual difference: %.4e\n', residual_diff(end));
fprintf('Final numerical error: %.4e\n', residual_numerical(end));
fprintf('Initial residual difference: %.4e\n', residual_diff(1));
fprintf('Initial numerical error: %.4e\n', residual_numerical(1));
fprintf('Convergence factor (residual): %.4f\n', residual_diff(end)/residual_diff(1));
fprintf('Convergence factor (numerical): %.4f\n', residual_numerical(end)/residual_numerical(1));

% Compute masses at different times
mass_initial = sum(rho_admm(1, :, :), 'all') * dx * dy;
mass_final = sum(rho_admm(nt, :, :), 'all') * dx * dy;
fprintf('\n--- Mass Conservation ---\n');
fprintf('Mass at t=%.3f: %.6f\n', t_rho(1), mass_initial);
fprintf('Mass at t=%.3f: %.6f\n', t_rho(nt), mass_final);
fprintf('Mass conservation error: %.4e\n', abs(mass_final - mass_initial));

fprintf('\nPost-processing complete!\n');
