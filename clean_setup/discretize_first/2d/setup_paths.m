% SETUP_PATHS  Add all required folders to the MATLAB path.
%
%   Run this once at the start of any session working in discretize_first/2d/.

base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', 'shared');
sh2d    = fullfile(sh_base, '2d');

% Shared (dimension-agnostic): mirt_dctn, mirt_idctn, solve_cubic
addpath(fullfile(sh_base, 'utils'));

% Shared (2D-specific)
addpath(sh2d);                                   % setup_problem
addpath(fullfile(sh2d, 'utils'));                 % admm_solve, ladmm_solve, s_*, precomp_banded_proj, ...
addpath(fullfile(sh2d, 'problems'));              % prob_*, analytical_*
addpath(fullfile(sh2d, 'discretization'));        % disc_staggered_1st
addpath(fullfile(sh2d, 'prox'));                  % prox_ke_exact
addpath(fullfile(sh2d, 'projection'));            % proj_fokker_planck_banded
addpath(fullfile(sh2d, 'pipelines'));            % discretize_then_optimize

% Local (discretize_first-specific)
addpath(base);
addpath(fullfile(base, 'config'));               % cfg_ladmm_gaussian
addpath(fullfile(base, 'prox'));                 % prox_ke_cc
addpath(fullfile(base, 'experiments'));
