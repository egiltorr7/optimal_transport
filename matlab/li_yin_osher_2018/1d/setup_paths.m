% SETUP_PATHS  Add all required folders to the MATLAB path for the
%   Li-Yin-Osher (2018) 1D Newton solver, plus the existing 1D codebase
%   it's benchmarked against.
%
%   Run this once at the start of any session working in
%   li_yin_osher_2018/1d/.

base = fileparts(mfilename('fullpath'));

addpath(fullfile(base, 'core'));
addpath(fullfile(base, 'experiments'));

% Existing discretize_first/1d codebase (prob_gaussian, setup_problem,
% cfg_ladmm_gaussian, discretize_then_optimize, ...), used for a shared
% problem instance and the ADMM comparison.
existing_1d = fullfile(base, '..', '..', 'discretize_first', '1d');
run(fullfile(existing_1d, 'setup_paths.m'));
