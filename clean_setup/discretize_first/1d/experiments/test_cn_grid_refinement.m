% TEST_CN_GRID_REFINEMENT  Grid refinement study for CN-ADMM vs Sinkhorn (SB, eps=1).
%
%   Sweep 1: nx=256 fixed, nt refined  [64, 128, 256, 512, 1024, 2048]
%   Sweep 2: mutual refinement nt=nx   [16, 32, 64, 128, 256, 512, 1024, 2048]
%
%   Both sweeps use the GPU Crank-Nicolson solver (cfg_ladmm_gaussian_gpu).
%   Figures are saved to results/figures/ and NOT displayed (headless-safe).
%
%   Saved figures:
%     cn_refine_nt_err.png    max L2 error vs nt (sweep 1)
%     cn_refine_nt_time.png   wall time vs nt (sweep 1)
%     cn_refine_nn_err.png    max L2 error vs n=nt=nx (sweep 2)
%     cn_refine_nn_time.png   wall time vs n (sweep 2)

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));



fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

VAREPS   = 1.0;
prob_def = prob_gaussian();

cfg_base            = cfg_ladmm_gaussian_gpu();
cfg_base.gpu_device = 3;   % whichever GPU is free
cfg_base.vareps     = VAREPS;
gpuDevice(cfg_base.gpu_device);   % select once before any GPU allocation

cfg_sink_tmpl.vareps       = VAREPS;
cfg_sink_tmpl.max_iter     = 50;
cfg_sink_tmpl.tol          = -1;
cfg_sink_tmpl.precomp_heat = @precomp_heat_neumann;

FS = 11;  LW = 1.5;

%% ========================================================================
%% Sweep 1: nx = 256 fixed, nt varies
%% ========================================================================
NX1  = 256;
NT1  = [64, 128, 256, 512, 1024];
ns1  = numel(NT1);

max_err1   = nan(1, ns1);
wall_cn1   = nan(1, ns1);
wall_sink1 = nan(1, ns1);
iters1     = nan(1, ns1);
conv1      = false(1, ns1);

fprintf('=== Sweep 1: nx=%d fixed, nt varies ===\n', NX1);
for i = 1:ns1
    nt_i = NT1(i);
    fprintf('  nt=%-5d nx=%d  ', nt_i, NX1);

    cfg_i      = cfg_base;
    cfg_i.nt   = nt_i;
    cfg_i.nx   = NX1;
    prob_i     = setup_problem(cfg_i, prob_def);

    cfg_sk_i   = cfg_sink_tmpl;

    try
        r_sk = sinkhorn_hopf_cole(prob_i, cfg_sk_i);
        wall_sink1(i) = r_sk.walltime;

        r_cn = discretize_then_optimize(cfg_i, prob_i);
        wall_cn1(i) = r_cn.walltime;
        iters1(i)   = r_cn.iters;
        conv1(i)    = r_cn.converged;

        rho_ref = r_sk.rho(2:prob_i.nt, :);   % (ntm x nx)
        e_vec   = sqrt(prob_i.dx * sum((r_cn.rho_stag - rho_ref).^2, 2));
        max_err1(i) = max(e_vec);
    catch ME
        fprintf('[FAILED: %s]  ', ME.message);
    end

    fprintf('err=%.2e  t_cn=%.1fs  t_sk=%.1fs  iters=%d  conv=%d\n', ...
        max_err1(i), wall_cn1(i), wall_sink1(i), iters1(i), conv1(i));
end

%% ========================================================================
%% Sweep 2: mutual refinement nt = nx = n
%% ========================================================================
N2   = [16, 32, 64, 128, 256, 512, 1024];
ns2  = numel(N2);

max_err2   = nan(1, ns2);
wall_cn2   = nan(1, ns2);
wall_sink2 = nan(1, ns2);
iters2     = nan(1, ns2);
conv2      = false(1, ns2);

fprintf('\n=== Sweep 2: nt = nx = n varies ===\n');
for i = 1:ns2
    n_i = N2(i);
    fprintf('  nt=nx=%-5d  ', n_i);

    cfg_i      = cfg_base;
    cfg_i.nt   = n_i;
    cfg_i.nx   = n_i;
    prob_i     = setup_problem(cfg_i, prob_def);

    cfg_sk_i   = cfg_sink_tmpl;

    try
        r_sk = sinkhorn_hopf_cole(prob_i, cfg_sk_i);
        wall_sink2(i) = r_sk.walltime;

        r_cn = discretize_then_optimize(cfg_i, prob_i);
        wall_cn2(i) = r_cn.walltime;
        iters2(i)   = r_cn.iters;
        conv2(i)    = r_cn.converged;

        rho_ref = r_sk.rho(2:prob_i.nt, :);   % (ntm x nx)
        e_vec   = sqrt(prob_i.dx * sum((r_cn.rho_stag - rho_ref).^2, 2));
        max_err2(i) = max(e_vec);
    catch ME
        fprintf('[FAILED: %s]  ', ME.message);
    end

    fprintf('err=%.2e  t_cn=%.1fs  t_sk=%.1fs  iters=%d  conv=%d\n', ...
        max_err2(i), wall_cn2(i), wall_sink2(i), iters2(i), conv2(i));
end

%% ========================================================================
%% Figures
%% ========================================================================

% --- Helper: reference slope line anchored to first valid point ----------
ref_slope = @(xv, yv, s) yv(1) * (xv / xv(1)).^s;

% --- Figure 1: max L2 error vs nt (sweep 1) ------------------------------
valid1 = ~isnan(max_err1);
fig1 = figure('Visible','off','Units','centimeters','Position',[2 2 14 9]);
hold on; set(gca, 'XScale','log', 'YScale','log');
if any(valid1)
    plot(NT1(valid1), max_err1(valid1), 'b^-', 'LineWidth', LW, 'MarkerSize', 7, ...
        'DisplayName', 'CN-ADMM (GPU)');
    ref1 = ref_slope(NT1(valid1), max_err1(valid1), -1);
    plot(NT1(valid1), ref1, 'k--', 'LineWidth', 1, 'DisplayName', '$O(N_T^{-1})$');
end
xlabel('$N_T$', 'FontSize', FS);
ylabel('$\max_t \|\rho_\mathrm{CN} - \rho_\mathrm{Sink}\|_{L^2}$', 'FontSize', FS);
title(sprintf('Sweep 1: $N_x=%d$ fixed, $\\varepsilon=%.4g$', NX1, VAREPS), 'FontSize', FS);
legend('Location','best','FontSize',FS-1,'Box','off');
set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;
saveas(fig1, fullfile(fig_dir, 'cn_refine_nt_err.png'));

% --- Figure 2: wall time vs nt (sweep 1) ---------------------------------
fig2 = figure('Visible','off','Units','centimeters','Position',[2 2 14 9]);
hold on; set(gca, 'XScale','log', 'YScale','log');
if any(valid1)
    plot(NT1(valid1), wall_cn1(valid1),   'b^-', 'LineWidth', LW, 'MarkerSize', 7, ...
        'DisplayName', 'CN-ADMM (GPU)');
    plot(NT1(valid1), wall_sink1(valid1), 'rs-', 'LineWidth', LW, 'MarkerSize', 7, ...
        'DisplayName', 'Sinkhorn (CPU)');
end
xlabel('$N_T$', 'FontSize', FS);
ylabel('Wall time (s)', 'FontSize', FS);
title(sprintf('Sweep 1: wall time  ($N_x=%d$, $\\varepsilon=%.4g$)', NX1, VAREPS), 'FontSize', FS);
legend('Location','best','FontSize',FS-1,'Box','off');
set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;
saveas(fig2, fullfile(fig_dir, 'cn_refine_nt_time.png'));

% --- Figure 3: max L2 error vs n=nt=nx (sweep 2) -------------------------
valid2 = ~isnan(max_err2);
fig3 = figure('Visible','off','Units','centimeters','Position',[2 2 14 9]);
hold on; set(gca, 'XScale','log', 'YScale','log');
if any(valid2)
    plot(N2(valid2), max_err2(valid2), 'b^-', 'LineWidth', LW, 'MarkerSize', 7, ...
        'DisplayName', 'CN-ADMM (GPU)');
    ref2 = ref_slope(N2(valid2), max_err2(valid2), -1);
    plot(N2(valid2), ref2, 'k--', 'LineWidth', 1, 'DisplayName', '$O(n^{-1})$');
end
xlabel('$n = N_T = N_x$', 'FontSize', FS);
ylabel('$\max_t \|\rho_\mathrm{CN} - \rho_\mathrm{Sink}\|_{L^2}$', 'FontSize', FS);
title(sprintf('Sweep 2: mutual refinement  ($\\varepsilon=%.4g$)', VAREPS), 'FontSize', FS);
legend('Location','best','FontSize',FS-1,'Box','off');
set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;
saveas(fig3, fullfile(fig_dir, 'cn_refine_nn_err.png'));

% --- Figure 4: wall time vs n=nt=nx (sweep 2) ----------------------------
fig4 = figure('Visible','off','Units','centimeters','Position',[2 2 14 9]);
hold on; set(gca, 'XScale','log', 'YScale','log');
if any(valid2)
    plot(N2(valid2), wall_cn2(valid2),   'b^-', 'LineWidth', LW, 'MarkerSize', 7, ...
        'DisplayName', 'CN-ADMM (GPU)');
    plot(N2(valid2), wall_sink2(valid2), 'rs-', 'LineWidth', LW, 'MarkerSize', 7, ...
        'DisplayName', 'Sinkhorn (CPU)');
end
xlabel('$n = N_T = N_x$', 'FontSize', FS);
ylabel('Wall time (s)', 'FontSize', FS);
title(sprintf('Sweep 2: wall time  ($\\varepsilon=%.4g$)', VAREPS), 'FontSize', FS);
legend('Location','best','FontSize',FS-1,'Box','off');
set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;
saveas(fig4, fullfile(fig_dir, 'cn_refine_nn_time.png'));

%% ========================================================================
%% Summary tables
%% ========================================================================
fprintf('\n--- Sweep 1 summary (nx=%d, eps=%.4g) ---\n', NX1, VAREPS);
fprintf('  %6s  %8s  %8s  %8s  %6s  %5s\n', 'nt', 'max_err', 't_cn', 't_sink', 'iters', 'conv');
for i = 1:ns1
    fprintf('  %6d  %8.2e  %8.2f  %8.2f  %6d  %5d\n', ...
        NT1(i), max_err1(i), wall_cn1(i), wall_sink1(i), iters1(i), conv1(i));
end

fprintf('\n--- Sweep 2 summary (nt=nx, eps=%.4g) ---\n', VAREPS);
fprintf('  %6s  %8s  %8s  %8s  %6s  %5s\n', 'n', 'max_err', 't_cn', 't_sink', 'iters', 'conv');
for i = 1:ns2
    fprintf('  %6d  %8.2e  %8.2f  %8.2f  %6d  %5d\n', ...
        N2(i), max_err2(i), wall_cn2(i), wall_sink2(i), iters2(i), conv2(i));
end

fprintf('\nFigures saved to: %s\n', fig_dir);
