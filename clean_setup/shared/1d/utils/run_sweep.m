function sweep = run_sweep(cfg, problem, param_name, param_values)
% RUN_SWEEP  Run a parameter sweep over a single config field.
%
%   sweep = run_sweep(cfg, problem, param_name, param_values)
%
%   For each value in param_values, overrides cfg.(param_name), runs the
%   pipeline, and collects results.
%
%   Inputs:
%     cfg          base config struct (not modified)
%     problem      problem struct from setup_problem
%     param_name   name of the cfg field to sweep (string), e.g. 'vareps'
%     param_values vector of values to sweep over
%
%   Output:
%     sweep        struct array with fields:
%       sweep(k).value   the parameter value used
%       sweep(k).cfg     the config used (with override applied)
%       sweep(k).result  the result struct from the pipeline

    n = length(param_values);
    sweep = struct('value', cell(n,1), 'cfg', cell(n,1), 'result', cell(n,1));

    fprintf('Sweeping %s over %d values...\n', param_name, n);

    for k = 1:n
        cfg_k = cfg;
        cfg_k.(param_name) = param_values(k);

        fprintf('  [%d/%d]  %s = %g ... ', k, n, param_name, param_values(k));
        result_k = cfg_k.pipeline(cfg_k, problem);
        fprintf('iters=%d  error=%.2e  time=%.1fs\n', ...
                result_k.iters, result_k.error, result_k.walltime);

        sweep(k).value  = param_values(k);
        sweep(k).cfg    = cfg_k;
        sweep(k).result = result_k;
    end

    fprintf('Sweep complete.\n');
end
