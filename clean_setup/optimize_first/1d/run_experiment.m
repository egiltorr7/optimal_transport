function result = run_experiment(cfg, problem)
% RUN_EXPERIMENT  Run a single experiment given a config and problem.
%
%   result = run_experiment(cfg, problem)
%
%   Calls cfg.pipeline(cfg, problem), saves result to results/raw/,
%   and prints a one-line summary.

    fprintf('[running] %s ... ', problem.name);
    result = cfg.pipeline(cfg, problem);
    fprintf('done  iters=%d  error=%.2e  time=%.1fs\n', ...
            result.iters, result.error, result.walltime);

    % Save result (named by problem + algorithm)
    raw_dir = fullfile(fileparts(mfilename('fullpath')), 'results', 'raw');
    if ~exist(raw_dir, 'dir'), mkdir(raw_dir); end
    save(fullfile(raw_dir, [problem.name '.mat']), 'result');
end
