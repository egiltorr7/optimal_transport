%% compare_projections_gaussian.m
%
% Compare projection methods for the 1D Gaussian SB problem.
%
% Methods compared:
%   1. Spectral (DCT) approximate  -- proj_fokker_planck         [cfg_spectral_proj]
%   2. Exact banded                -- proj_fokker_planck_banded  [cfg_banded_proj]
%   5. PCG with DCT preconditioner -- proj_fp_pcg                [cfg_fp_pcg]
%
% For each method: run ADMM to convergence, record iterations, wall time,
% and final ADMM residual. Plot convergence histories.

clear; clc;

base = fileparts(mfilename('fullpath'));
run(fullfile(base, '..', 'setup_paths.m'));

fig_dir = fullfile(base, '..', 'results', 'figures', 'compare_projections');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% Problem
prob_def = prob_gaussian();

%% Configurations to compare
cfgs = { cfg_spectral_proj(), cfg_banded_proj(), cfg_fp_dr(), cfg_fp_pcg() };
labels = {'Spectral (DCT)', 'Banded (exact)', 'DR (C1/C2 split)', 'PCG (DCT precond)'};
n_methods = length(cfgs);

results = cell(n_methods, 1);

%% Run each method
for mi = 1:n_methods
    cfg     = cfgs{mi};
    problem = setup_problem(cfg, prob_def);

    fprintf('\n--- Method %d: %s ---\n', mi, labels{mi});
    t0 = tic;
    results{mi} = cfg.pipeline(cfg, problem);
    results{mi}.walltime_total = toc(t0);

    fprintf('  iters    = %d\n', results{mi}.iters);
    fprintf('  residual = %.2e\n', results{mi}.error);
    fprintf('  time     = %.2f s\n', results{mi}.walltime_total);
end

%% -------------------------------------------------------------------------
%% Figure 1: ADMM residual vs iteration
%% -------------------------------------------------------------------------
figure('Name', 'ADMM convergence');
colors = {'b', 'r', 'g', 'm'};
for mi = 1:n_methods
    res = results{mi}.residual;
    semilogy(1:length(res), res, [colors{mi} '-'], ...
             'LineWidth', 1.5, 'DisplayName', labels{mi});
    hold on;
end
xlabel('ADMM iteration', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Residual', 'Interpreter', 'latex', 'FontSize', 13);
title('Convergence: Gaussian SB', 'Interpreter', 'latex', 'FontSize', 13);
legend('Location', 'northeast', 'Interpreter', 'latex');
grid on;
saveas(gcf, fullfile(fig_dir, 'admm_convergence.png'));

%% -------------------------------------------------------------------------
%% Figure 2: Summary bar chart (iters and time)
%% -------------------------------------------------------------------------
iters = cellfun(@(r) r.iters, results);
times = cellfun(@(r) r.walltime_total, results);

figure('Name', 'Summary');
tiledlayout(1, 2);

nexttile;
bar(iters);
set(gca, 'XTickLabel', labels, 'XTickLabelRotation', 15);
ylabel('ADMM iterations', 'Interpreter', 'latex');
title('Iterations to convergence', 'Interpreter', 'latex');

nexttile;
bar(times);
set(gca, 'XTickLabel', labels, 'XTickLabelRotation', 15);
ylabel('Wall time (s)', 'Interpreter', 'latex');
title('Total wall time', 'Interpreter', 'latex');

saveas(gcf, fullfile(fig_dir, 'summary.png'));

fprintf('\nFigures saved to: %s\n', fig_dir);
