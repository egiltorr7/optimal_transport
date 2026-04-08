% SWEEP_GRID_SEPARATE  Grid refinement studies: nx-only, nt-only, and joint.
%
%   Three separate convergence sweeps against the analytical SB solution:
%
%     1. Sweep nx  (fix nt=128): isolates spatial convergence rate
%     2. Sweep nt  (fix nx=128): isolates temporal convergence rate
%     3. Joint     (nt=nx):      combined rate
%
%   For each sweep: L2 error ||(rho,m) - analytical|| vs grid parameter (log-log),
%   kinetic energy vs grid, and density evolution to see oscillation convergence.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% --- Shared config ---
cfg_base = cfg_ladmm_gaussian();
prob_def = prob_gaussian();

NT_FIX = 128;   % fixed nt when sweeping nx
NX_FIX = 128;   % fixed nx when sweeping nt

nx_vals   = [32,  64,  128, 256, 512];
nt_vals   = [16,  32,  64,  128, 256];
nj_vals   = [16,  32,  64,  128, 256];   % joint: nt=nx=nj

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
ftag = sprintf('sweep_separate_eps%g_gam%g_tau%g', ...
    cfg_base.vareps, cfg_base.gamma, cfg_base.tau);

ke_ylabel = '$\int\!\!\int m^2/\rho\,\mathrm{d}t\,\mathrm{d}x$';

%% -----------------------------------------------------------------------
%% Sweep 1: nx only  (nt fixed at NT_FIX)
%% -----------------------------------------------------------------------
fprintf('\n=== Sweep 1: nx only (nt=%d fixed) ===\n', NT_FIX);
n1 = numel(nx_vals);
err1 = zeros(n1, 2);   ke1 = zeros(n1, 2);
res1 = cell(n1, 1);    prob1 = cell(n1, 1);   rho_ana1 = cell(n1, 1);

for k = 1:n1
    cfg_k    = cfg_base;
    cfg_k.nt = NT_FIX;
    cfg_k.nx = nx_vals(k);
    [err1(k,1), err1(k,2), ke1(k,1), ke1(k,2), res1{k}, prob1{k}, rho_ana1{k}] = ...
        run_level(cfg_k, prob_def);
end

%% -----------------------------------------------------------------------
%% Sweep 2: nt only  (nx fixed at NX_FIX)
%% -----------------------------------------------------------------------
fprintf('\n=== Sweep 2: nt only (nx=%d fixed) ===\n', NX_FIX);
n2 = numel(nt_vals);
err2 = zeros(n2, 2);   ke2 = zeros(n2, 2);
res2 = cell(n2, 1);    prob2 = cell(n2, 1);   rho_ana2 = cell(n2, 1);

for k = 1:n2
    cfg_k    = cfg_base;
    cfg_k.nt = nt_vals(k);
    cfg_k.nx = NX_FIX;
    [err2(k,1), err2(k,2), ke2(k,1), ke2(k,2), res2{k}, prob2{k}, rho_ana2{k}] = ...
        run_level(cfg_k, prob_def);
end

%% -----------------------------------------------------------------------
%% Sweep 3: joint  (nt = nx = nj)
%% -----------------------------------------------------------------------
fprintf('\n=== Sweep 3: joint (nt=nx) ===\n');
n3 = numel(nj_vals);
err3 = zeros(n3, 2);   ke3 = zeros(n3, 2);
res3 = cell(n3, 1);    prob3 = cell(n3, 1);   rho_ana3 = cell(n3, 1);

for k = 1:n3
    cfg_k    = cfg_base;
    cfg_k.nt = nj_vals(k);
    cfg_k.nx = nj_vals(k);
    [err3(k,1), err3(k,2), ke3(k,1), ke3(k,2), res3{k}, prob3{k}, rho_ana3{k}] = ...
        run_level(cfg_k, prob_def);
end

%% -----------------------------------------------------------------------
%% Plot A: L2 error for all three sweeps (one figure, three subplots)
%% -----------------------------------------------------------------------
err1_tot = sqrt(err1(:,1).^2 + err1(:,2).^2);
err2_tot = sqrt(err2(:,1).^2 + err2(:,2).^2);
err3_tot = sqrt(err3(:,1).^2 + err3(:,2).^2);

figure('Position', [100 100 1100 340]);

subplot(1, 3, 1);
loglog(nx_vals, err1_tot, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
loglog(nx_vals, err1_tot(1)*(nx_vals(1)./nx_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$L^2$ error', 'Interpreter', 'latex', 'FontSize', 12);
title(sprintf('Spatial ($n_t=%d$ fixed)', NT_FIX), 'Interpreter', 'latex');
legend('error', '$\mathcal{O}(n_x^{-2})$', 'Interpreter', 'latex', 'Location', 'southwest');
grid on;

subplot(1, 3, 2);
loglog(nt_vals, err2_tot, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
loglog(nt_vals, err2_tot(1)*(nt_vals(1)./nt_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_t$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$L^2$ error', 'Interpreter', 'latex', 'FontSize', 12);
title(sprintf('Temporal ($n_x=%d$ fixed)', NX_FIX), 'Interpreter', 'latex');
legend('error', '$\mathcal{O}(n_t^{-2})$', 'Interpreter', 'latex', 'Location', 'southwest');
grid on;

subplot(1, 3, 3);
loglog(nj_vals, err3_tot, 'k^-', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
loglog(nj_vals, err3_tot(1)*(nj_vals(1)./nj_vals).^2, 'k--', 'LineWidth', 1.2);
xlabel('$n_t = n_x$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$L^2$ error', 'Interpreter', 'latex', 'FontSize', 12);
title('Joint ($n_t = n_x$)', 'Interpreter', 'latex');
legend('error', '$\mathcal{O}(n^{-2})$', 'Interpreter', 'latex', 'Location', 'southwest');
grid on;

sgtitle(sprintf('Grid refinement   $\\varepsilon=%g,\\; \\gamma=%g,\\; \\tau=%g$', ...
    cfg_base.vareps, cfg_base.gamma, cfg_base.tau), 'Interpreter', 'latex', 'FontSize', 13);

saveas(gcf, fullfile(fig_dir, sprintf('convergence_%s.png', ftag)));

%% -----------------------------------------------------------------------
%% Plot B: Kinetic energy for all three sweeps
%% -----------------------------------------------------------------------
figure('Position', [100 500 1100 340]);

subplot(1, 3, 1);
semilogx(nx_vals, ke1(:,1), 'bo-', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogx(nx_vals, ke1(:,2), 'r--', 'LineWidth', 1.5);
xlabel('$n_x$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel(ke_ylabel, 'Interpreter', 'latex', 'FontSize', 12);
title(sprintf('Spatial ($n_t=%d$ fixed)', NT_FIX), 'Interpreter', 'latex');
legend('LADMM', 'Analytical', 'Interpreter', 'latex', 'Location', 'best');
grid on;

subplot(1, 3, 2);
semilogx(nt_vals, ke2(:,1), 'bo-', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogx(nt_vals, ke2(:,2), 'r--', 'LineWidth', 1.5);
xlabel('$n_t$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel(ke_ylabel, 'Interpreter', 'latex', 'FontSize', 12);
title(sprintf('Temporal ($n_x=%d$ fixed)', NX_FIX), 'Interpreter', 'latex');
legend('LADMM', 'Analytical', 'Interpreter', 'latex', 'Location', 'best');
grid on;

subplot(1, 3, 3);
semilogx(nj_vals, ke3(:,1), 'bo-', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogx(nj_vals, ke3(:,2), 'r--', 'LineWidth', 1.5);
xlabel('$n_t = n_x$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel(ke_ylabel, 'Interpreter', 'latex', 'FontSize', 12);
title('Joint ($n_t = n_x$)', 'Interpreter', 'latex');
legend('LADMM', 'Analytical', 'Interpreter', 'latex', 'Location', 'best');
grid on;

sgtitle(sprintf('Kinetic energy   $\\varepsilon=%g,\\; \\gamma=%g,\\; \\tau=%g$', ...
    cfg_base.vareps, cfg_base.gamma, cfg_base.tau), 'Interpreter', 'latex', 'FontSize', 13);

saveas(gcf, fullfile(fig_dir, sprintf('ke_%s.png', ftag)));

%% -----------------------------------------------------------------------
%% Plot C: Density evolution — Sweep 1 (nx only, nt=NT_FIX)
%% -----------------------------------------------------------------------
plot_density_sweep(res1, prob1, rho_ana1, nx_vals, 'n_x', NT_FIX, 'n_t', ...
    cfg_base, fig_dir, sprintf('density_sweep1_nx_%s.png', ftag));

%% -----------------------------------------------------------------------
%% Plot D: Density evolution — Sweep 2 (nt only, nx=NX_FIX)
%% -----------------------------------------------------------------------
plot_density_sweep(res2, prob2, rho_ana2, nt_vals, 'n_t', NX_FIX, 'n_x', ...
    cfg_base, fig_dir, sprintf('density_sweep2_nt_%s.png', ftag));

%% -----------------------------------------------------------------------
%% Plot E: Density evolution — Sweep 3 (joint nt=nx)
%% -----------------------------------------------------------------------
plot_density_sweep(res3, prob3, rho_ana3, nj_vals, 'n_t=n_x', [], '', ...
    cfg_base, fig_dir, sprintf('density_sweep3_joint_%s.png', ftag));

%% --- Summary tables ---
fprintf('\n--- Sweep 1: nx only (nt=%d) ---\n', NT_FIX);
fprintf('%-6s  %-10s  %-10s\n', 'nx', 'err_total', 'KE');
for k = 1:n1
    fprintf('%-6d  %-10.3e  %-10.6f\n', nx_vals(k), err1_tot(k), ke1(k,1));
end

fprintf('\n--- Sweep 2: nt only (nx=%d) ---\n', NX_FIX);
fprintf('%-6s  %-10s  %-10s\n', 'nt', 'err_total', 'KE');
for k = 1:n2
    fprintf('%-6d  %-10.3e  %-10.6f\n', nt_vals(k), err2_tot(k), ke2(k,1));
end

fprintf('\n--- Sweep 3: joint (nt=nx) ---\n');
fprintf('%-6s  %-10s  %-10s\n', 'n', 'err_total', 'KE');
for k = 1:n3
    fprintf('%-6d  %-10.3e  %-10.6f\n', nj_vals(k), err3_tot(k), ke3(k,1));
end

%% -----------------------------------------------------------------------
%% Helper: run one grid level; return errors, KE, full result, problem, rho_a
%% -----------------------------------------------------------------------
function [err_rho, err_mx, ke_num, ke_ref, r, prob, rho_a] = run_level(cfg, prob_def)
    prob          = setup_problem(cfg, prob_def);
    r             = cfg.pipeline(cfg, prob);
    [rho_a, mx_a] = analytical_sb_gaussian(prob, cfg.vareps);
    dt = prob.dt;   dx = prob.dx;
    err_rho = norm(r.rho_stag(:) - rho_a(:)) * sqrt(dt * dx);
    err_mx  = norm(r.mx_stag(:)  - mx_a(:))  * sqrt(dt * dx);
    ke_num  = compute_objective(r.rho_stag, r.mx_stag, prob) / dx;
    ke_ref  = compute_objective(rho_a, mx_a, prob) / dx;
    fprintf('  nt=%3d  nx=%3d  iters=%d  err=%.2e  time=%.1fs\n', ...
        cfg.nt, cfg.nx, r.iters, r.error, r.walltime);
end

%% -----------------------------------------------------------------------
%% Helper: density evolution subplots for one sweep
%%   sweep_vals  — the varying grid parameter values
%%   sweep_label — LaTeX name of the varying parameter (e.g. 'n_x')
%%   fixed_val   — scalar value of the fixed parameter ([] if joint)
%%   fixed_label — LaTeX name of the fixed parameter ('' if joint)
%% -----------------------------------------------------------------------
function plot_density_sweep(results, problems, rho_anas, sweep_vals, ...
        sweep_label, fixed_val, fixed_label, cfg_base, fig_dir, fname)

    n       = numel(sweep_vals);
    t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
    colors  = lines(numel(t_fracs));
    ncols   = ceil((n + 1) / 2);

    figure('Position', [100 100 300*ncols 500]);

    for k = 1:n
        subplot(2, ncols, k);
        hold on;

        prob   = problems{k};
        ntm    = prob.nt - 1;
        nx     = prob.nx;
        xx     = prob.xx;
        dt     = prob.dt;
        stride = max(1, floor(nx / 40));
        idx    = 1:stride:nx;

        for j = 1:numel(t_fracs)
            ti = max(1, min(ntm, round(t_fracs(j) / dt)));
            plot(xx, rho_anas{k}(ti,:), '-', 'Color', colors(j,:), 'LineWidth', 1.5);
            plot(xx(idx), results{k}.rho_stag(ti,idx), 'o', 'Color', colors(j,:), ...
                 'MarkerSize', 5, 'MarkerFaceColor', colors(j,:));
        end

        if ~isempty(fixed_val)
            ttl = sprintf('$%s=%d,\\; %s=%d$', sweep_label, sweep_vals(k), ...
                          fixed_label, fixed_val);
        else
            ttl = sprintf('$%s=%d$', sweep_label, sweep_vals(k));
        end
        title(ttl, 'Interpreter', 'latex');
        xlabel('$x$', 'Interpreter', 'latex');
        if mod(k-1, ncols) == 0
            ylabel('$\rho$', 'Interpreter', 'latex');
        end
        grid on;
    end

    % Legend panel
    ax_leg = subplot(2, ncols, n + 1); %#ok<NASGU>
    hold on; axis off;
    leg = {};
    for j = 1:numel(t_fracs)
        plot(nan, nan, '-',  'Color', colors(j,:), 'LineWidth', 1.5);
        plot(nan, nan, 'o',  'Color', colors(j,:), 'MarkerFaceColor', colors(j,:), 'MarkerSize', 5);
        leg{end+1} = sprintf('Ana $t=%.2f$',   t_fracs(j)); %#ok<AGROW>
        leg{end+1} = sprintf('LADMM $t=%.2f$', t_fracs(j)); %#ok<AGROW>
    end
    legend(leg, 'Interpreter', 'latex', 'FontSize', 8, 'Location', 'best', 'NumColumns', 2);

    if ~isempty(fixed_val)
        stitle = sprintf('Density: sweep $%s$ ($%s=%d$ fixed),   $\\varepsilon=%g$', ...
            sweep_label, fixed_label, fixed_val, cfg_base.vareps);
    else
        stitle = sprintf('Density: joint sweep $%s$,   $\\varepsilon=%g$', ...
            sweep_label, cfg_base.vareps);
    end
    sgtitle(stitle, 'Interpreter', 'latex', 'FontSize', 12);

    saveas(gcf, fullfile(fig_dir, fname));
end
