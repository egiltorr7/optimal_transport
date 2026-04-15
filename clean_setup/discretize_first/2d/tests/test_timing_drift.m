% TEST_TIMING_DRIFT  Diagnose growing per-iteration time on CPU vs GPU.
%
%   Runs a small fixed problem for 500 iterations on CPU, then on GPU
%   (if available), and plots sec/iter over time for both.
%
%   Interpretation:
%     CPU flat, GPU grows  -> GPU thermal throttling or CUDA context overhead
%     Both grow            -> struct/GC overhead in the algorithm (pre-allocation needed)
%     Both flat            -> drift only appears at larger grid sizes (memory pressure)

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Small fixed config ---
cfg_base             = cfg_ladmm_gaussian();
cfg_base.nt          = 32;
cfg_base.nx          = 32;
cfg_base.ny          = 32;
cfg_base.max_iter    = 500;
cfg_base.tol         = 0;       % run all 500 iters — never stop early
cfg_base.print_every = 0;       % suppress output

prob_def = prob_gaussian();

%% --- CPU run ---
fprintf('Running CPU  (nt=%d nx=%d ny=%d, 500 iters)...\n', ...
    cfg_base.nt, cfg_base.nx, cfg_base.ny);
cfg_cpu         = cfg_base;
cfg_cpu.use_gpu = false;
problem_cpu     = setup_problem(cfg_cpu, prob_def);
res_cpu         = discretize_then_optimize(cfg_cpu, problem_cpu);
t_cpu           = res_cpu.iter_times;
fprintf('  done.  mean=%.4fs  max=%.4fs  drift=%.1f%%\n', ...
    mean(t_cpu), max(t_cpu), 100*(t_cpu(end)-t_cpu(1))/t_cpu(1));

%% --- GPU run ---
has_gpu = gpuDeviceCount() > 0;
if has_gpu
    fprintf('Running GPU  (nt=%d nx=%d ny=%d, 500 iters)...\n', ...
        cfg_base.nt, cfg_base.nx, cfg_base.ny);
    cfg_gpu         = cfg_base;
    cfg_gpu.use_gpu = true;
    problem_gpu     = setup_problem(cfg_gpu, prob_def);
    res_gpu         = discretize_then_optimize(cfg_gpu, problem_gpu);
    t_gpu           = res_gpu.iter_times;
    fprintf('  done.  mean=%.4fs  max=%.4fs  drift=%.1f%%\n', ...
        mean(t_gpu), max(t_gpu), 100*(t_gpu(end)-t_gpu(1))/t_gpu(1));
else
    fprintf('No GPU found — skipping GPU run.\n');
    t_gpu = [];
end

%% --- Plot ---
figure('Name', 'Timing drift: CPU vs GPU', 'Position', [100 100 800 380]);
hold on;

iters = (1:numel(t_cpu))';
plot(iters, t_cpu * 1000, 'b-', 'LineWidth', 1.2, 'DisplayName', 'CPU');

p_cpu   = polyfit(iters, t_cpu * 1000, 1);
plot(iters, polyval(p_cpu, iters), 'b--', 'LineWidth', 0.8, ...
    'DisplayName', sprintf('CPU fit  slope=%.3f ms/iter', p_cpu(1)));

if ~isempty(t_gpu)
    iters_g = (1:numel(t_gpu))';
    plot(iters_g, t_gpu * 1000, 'r-', 'LineWidth', 1.2, 'DisplayName', 'GPU');
    p_gpu = polyfit(iters_g, t_gpu * 1000, 1);
    plot(iters_g, polyval(p_gpu, iters_g), 'r--', 'LineWidth', 0.8, ...
        'DisplayName', sprintf('GPU fit  slope=%.3f ms/iter', p_gpu(1)));
end

xlabel('Iteration');
ylabel('ms / iter');
title(sprintf('Per-iteration time   nt=%d nx=%d ny=%d', ...
    cfg_base.nt, cfg_base.nx, cfg_base.ny));
legend('Location', 'northwest');
grid on;
saveas(gcf, fullfile(fig_dir, 'timing_drift.png'));

%% --- Diagnosis ---
fprintf('\n--- Diagnosis ---\n');
slope_cpu_pct = 100 * p_cpu(1) * 500 / (p_cpu(2) + p_cpu(1));
fprintf('  CPU slope: %.3f ms/iter  (%.1f%% growth over 500 iters)\n', ...
    p_cpu(1), slope_cpu_pct);
if ~isempty(t_gpu)
    slope_gpu_pct = 100 * p_gpu(1) * 500 / (p_gpu(2) + p_gpu(1));
    fprintf('  GPU slope: %.3f ms/iter  (%.1f%% growth over 500 iters)\n', ...
        p_gpu(1), slope_gpu_pct);

    if abs(slope_cpu_pct) < 10 && slope_gpu_pct > 20
        fprintf('\n  -> CPU flat, GPU grows.\n');
        fprintf('     Cause: GPU thermal throttling or CUDA context state accumulation.\n');
    elseif slope_cpu_pct > 20 && slope_gpu_pct > 20
        fprintf('\n  -> Both CPU and GPU grow.\n');
        fprintf('     Cause: struct allocation / GC overhead in the algorithm itself.\n');
        fprintf('     Fix: pre-allocate intermediate arrays before the ADMM loop.\n');
    else
        fprintf('\n  -> Both approximately flat at this grid size.\n');
        fprintf('     Drift may only appear at your production grid size (memory pressure).\n');
        fprintf('     Try re-running with nt=64, nx=64, ny=64 to reproduce.\n');
    end
end
