%% main.m  --  Optimize-first 1D experiments
%
% Usage:
%   1. Add configs and problem definitions to the lists below
%   2. Run this script
%   3. Results saved to results/raw/  figures to results/figures/

clear; clc;

%% Add paths
run(fullfile(fileparts(mfilename('fullpath')), 'setup_paths.m'));

%% Define experiments
cfg = cfg_staggered_gaussian();

% problems = {
%     prob_gaussian(),
%     prob_bimodal(),
% };

problems = {
    prob_gaussian(),
};

%% Run
results = cell(numel(problems), 1);
for i = 1:numel(problems)
    problem    = setup_problem(cfg, problems{i});
    results{i} = run_experiment(cfg, problem);
end

%% Plot
for i = 1:numel(results)
    problem = setup_problem(cfg, problems{i});
    plot_results(results{i}, problem, cfg);
end
