% SETUP_PATHS  Add all required folders to the MATLAB path.
%
%   Run this once at the start of any session working in discretize_first/1d/.

base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', 'shared');
sh1d    = fullfile(sh_base, '1d');

% Shared (dimension-agnostic): mirt_dctn, mirt_idctn, solve_cubic
addpath(fullfile(sh_base, 'utils'));

% Shared (1D-specific)
addpath(sh1d);                                   % setup_problem
addpath(fullfile(sh1d, 'utils'));                 % admm_solve, ladmm_solve, s_*, precomp_banded_proj, ...
addpath(fullfile(sh1d, 'problems'));              % prob_*, analytical_*
addpath(fullfile(sh1d, 'discretization'));        % disc_staggered_1st
addpath(fullfile(sh1d, 'prox'));                  % prox_ke_exact
addpath(fullfile(sh1d, 'projection'));            % proj_fokker_planck_banded
addpath(fullfile(sh1d, 'pipelines'));            % discretize_then_optimize

% Local (discretize_first-specific)
addpath(base);
addpath(fullfile(base, 'config'));               % cfg_ladmm_gaussian
addpath(fullfile(base, 'pipelines'));            % discretize_then_optimize
addpath(fullfile(base, 'prox'));                 % prox_ke_cc
addpath(fullfile(base, 'experiments'));
