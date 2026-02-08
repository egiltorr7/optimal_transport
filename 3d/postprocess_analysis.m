%% Post-Processing Script for Optimal Transport Analysis - 3D
% This script analyzes data from main_simple.m and creates:
% 1. Error analysis plots (true error and residual difference vs iterations)
% 2. Visualization of 3D density distribution and velocity fields

clear; close all;

%% Configuration
folder = "./data/";
figures_folder = "./figures/";
case_name = "gaussian_";

% Grid parameters (should match main_simple.m)
tau = 0.1;
nt = 31;
nx = 31;
ny = nx;
nz = nx;

grid_name = sprintf('tau%0.2f_nt%d_nx%d_ny%d_nz%d', tau, nt, nx, ny, nz);
file_name = folder + case_name + grid_name;

% Visualization settings
visualize = true;
save_figures = true;
time_snapshots = [1, floor(nt/4), floor(nt/2), floor(3*nt/4), nt];  % Times to visualize

%% Load Data
fprintf('Loading data from %s...\n', file_name);
data = load(file_name + ".mat");
post_data = load(file_name + "_post.mat");

rho_admm = data.rho_admm;
mx_admm = data.mx_admm;
my_admm = data.my_admm;
mz_admm = data.mz_admm;
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
nzp = nz + 1;
dx = 1/nxp;
dy = 1/nyp;
dz = 1/nzp;
dt = 1/ntp;

% Spatial grids
x = (0:nxp-1) * dx;
y = (0:nyp-1) * dy;
z = (0:nzp-1) * dz;

% Temporal grids
t_rho = (0:nt-1) * dt;
t_m = (0:nt) * dt;

[X, Y, Z] = meshgrid(y, x, z);  % Note: meshgrid uses (y,x,z) ordering

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

if save_figures
    saveas(gcf, figures_folder + "error_residual_diff_" + grid_name + ".png");
    fprintf('Saved: error_residual_diff_%s.png\n', grid_name);
end

% Figure 2: True Numerical Error (excluding last point)
figure('Position', [150, 150, 800, 600]);
marker_interval = max(1, floor((maxIter-1)/50));
semilogy(iterations(1:end-1), residual_numerical(1:end-1), '-o', 'LineWidth', 2, ...
    'Color', [0.8, 0.2, 0.2], 'MarkerIndices', 1:marker_interval:length(iterations)-1, ...
    'MarkerSize', 6, 'MarkerFaceColor', [0.8, 0.2, 0.2]);
grid on;
xlabel('Iteration $k$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\| u^k - u^* \|_2$', 'Interpreter', 'latex', 'FontSize', 14);
title('True Numerical Error', 'Interpreter', 'latex', 'FontSize', 16);
set(gca, 'FontSize', 12);
xlim([1, maxIter-1]);

if save_figures
    saveas(gcf, figures_folder + "error_numerical_" + grid_name + ".png");
    fprintf('Saved: error_numerical_%s.png\n', grid_name);
end

% Figure 3: Combined Error Plot
figure('Position', [200, 200, 800, 600]);
marker_interval_combined = max(1, floor(maxIter/50));
semilogy(iterations, residual_diff, 'LineWidth', 2, ...
    'DisplayName', '$\| u^{k+1} - u^k \|_2$ (Residual)', 'Color', [0.2, 0.4, 0.8]);
hold on;
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

if save_figures
    saveas(gcf, figures_folder + "error_combined_" + grid_name + ".png");
    fprintf('Saved: error_combined_%s.png\n', grid_name);
end

%% Compute velocity field from momentum
fprintf('\nComputing velocity fields...\n');

% Reconstruct rho0 and rho1 from problem setup
rho0 = zeros(nxp, nyp, nzp);
rho1 = zeros(nxp, nyp, nzp);
rho0((x'-0.25).^2 + (y-0.25).^2 + (reshape(z,1,1,nzp)-0.25).^2 < 0.01) = 1;
rho1((x'-0.75).^2 + (y-0.75).^2 + (reshape(z,1,1,nzp)-0.75).^2 < 0.01) = 1;
rho0_mass = sum(rho0, 'all') * dx * dy * dz;
rho1_mass = sum(rho1, 'all') * dx * dy * dz;
rho1 = rho1 * (rho0_mass / rho1_mass);

vx_admm = zeros(ntp, nx, nyp, nzp);
vy_admm = zeros(ntp, nxp, ny, nzp);
vz_admm = zeros(ntp, nxp, nyp, nz);

% Compute velocity for interior times (tt = 1 to nt)
for tt = 1:nt
    safe_rho_x = max(rho_admm(tt, 1:nx, :, :), 1e-12);
    vx_admm(tt, :, :, :) = mx_admm(tt, :, :, :) ./ safe_rho_x;

    safe_rho_y = max(rho_admm(tt, :, 1:ny, :), 1e-12);
    vy_admm(tt, :, :, :) = my_admm(tt, :, :, :) ./ safe_rho_y;

    safe_rho_z = max(rho_admm(tt, :, :, 1:nz), 1e-12);
    vz_admm(tt, :, :, :) = mz_admm(tt, :, :, :) ./ safe_rho_z;
end

% For the final time point (ntp), use rho1 boundary condition
safe_rho_x = max(rho1(1:nx, :, :), 1e-12);
vx_admm(ntp, :, :, :) = squeeze(mx_admm(ntp, :, :, :)) ./ safe_rho_x;

safe_rho_y = max(rho1(:, 1:ny, :), 1e-12);
vy_admm(ntp, :, :, :) = squeeze(my_admm(ntp, :, :, :)) ./ safe_rho_y;

safe_rho_z = max(rho1(:, :, 1:nz), 1e-12);
vz_admm(ntp, :, :, :) = squeeze(mz_admm(ntp, :, :, :)) ./ safe_rho_z;

%% 3D Visualization

if visualize
    fprintf('\nCreating 3D visualizations...\n');

    for snap_idx = 1:length(time_snapshots)
        tt = time_snapshots(snap_idx);
        if tt > nt
            continue;
        end

        rho_current = squeeze(rho_admm(tt, :, :, :));
        current_time = t_rho(tt);

        % Create figure with multiple subplots
        fig = figure('Position', [250, 250, 1200, 800]);

        % Subplot 1: Isosurface of density
        subplot(2, 2, 1);
        iso_value = max(rho_current(:)) * 0.3;  % 30% of max density
        if iso_value > 0
            p = patch(isosurface(Y, X, Z, rho_current, iso_value));
            isonormals(Y, X, Z, rho_current, p);
            p.FaceColor = 'red';
            p.EdgeColor = 'none';
            camlight;
            lighting gouraud;
            alpha(0.7);
        end
        xlabel('y'); ylabel('x'); zlabel('z');
        title(sprintf('Isosurface: $t=%.3f$', current_time), 'Interpreter', 'latex');
        axis equal tight;
        view(3);
        grid on;

        % Subplot 2: Slice at z = 0.5
        subplot(2, 2, 2);
        z_slice_idx = round(nzp/2);
        rho_slice = squeeze(rho_current(:, :, z_slice_idx));
        imagesc(y, x, rho_slice);
        set(gca, 'YDir', 'normal');
        colorbar;
        colormap(gca, 'hot');
        xlabel('y'); ylabel('x');
        title(sprintf('Density slice (z=%.2f, t=%.3f)', z(z_slice_idx), current_time), ...
            'Interpreter', 'latex');
        axis equal tight;

        % Subplot 3: Slice at y = 0.5
        subplot(2, 2, 3);
        y_slice_idx = round(nyp/2);
        rho_slice = squeeze(rho_current(:, y_slice_idx, :));
        imagesc(z, x, rho_slice);
        set(gca, 'YDir', 'normal');
        colorbar;
        colormap(gca, 'hot');
        xlabel('z'); ylabel('x');
        title(sprintf('Density slice (y=%.2f, t=%.3f)', y(y_slice_idx), current_time), ...
            'Interpreter', 'latex');
        axis equal tight;

        % Subplot 4: Slice at x = 0.5
        subplot(2, 2, 4);
        x_slice_idx = round(nxp/2);
        rho_slice = squeeze(rho_current(x_slice_idx, :, :));
        imagesc(z, y, rho_slice');
        set(gca, 'YDir', 'normal');
        colorbar;
        colormap(gca, 'hot');
        xlabel('z'); ylabel('y');
        title(sprintf('Density slice (x=%.2f, t=%.3f)', x(x_slice_idx), current_time), ...
            'Interpreter', 'latex');
        axis equal tight;

        sgtitle(sprintf('3D Optimal Transport: t = %.3f', current_time), ...
            'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold');

        if save_figures
            filename = sprintf('%sdensity_3d_t%03d_%s.png', figures_folder, tt, grid_name);
            saveas(fig, filename);
            fprintf('Saved: %s\n', filename);
        end

        fprintf('Visualization %d/%d complete (t=%.3f)\n', snap_idx, length(time_snapshots), current_time);
    end
end

%% Summary Statistics
fprintf('\n=== Summary Statistics ===\n');
fprintf('Grid: nt=%d, nx=%d, ny=%d, nz=%d\n', nt, nx, ny, nz);
fprintf('Total iterations: %d\n', maxIter);
fprintf('\n--- Convergence ---\n');
fprintf('Final residual difference: %.4e\n', residual_diff(end));
fprintf('Final numerical error: %.4e\n', residual_numerical(end));
fprintf('Initial residual difference: %.4e\n', residual_diff(1));
fprintf('Initial numerical error: %.4e\n', residual_numerical(1));
fprintf('Convergence factor (residual): %.4f\n', residual_diff(end)/residual_diff(1));
fprintf('Convergence factor (numerical): %.4f\n', residual_numerical(end)/residual_numerical(1));

% Compute masses at different times
mass_initial = sum(rho_admm(1, :, :, :), 'all') * dx * dy * dz;
mass_final = sum(rho_admm(nt, :, :, :), 'all') * dx * dy * dz;
fprintf('\n--- Mass Conservation ---\n');
fprintf('Mass at t=%.3f: %.6f\n', t_rho(1), mass_initial);
fprintf('Mass at t=%.3f: %.6f\n', t_rho(nt), mass_final);
fprintf('Mass conservation error: %.4e\n', abs(mass_final - mass_initial));

fprintf('\nPost-processing complete!\n');
