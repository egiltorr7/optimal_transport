% SETUP_PATHS  Add all required folders to the MATLAB path.
%
%   Run this once at the start of any session working in discretize_first/1d_spectral/.

base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', 'shared');
sh1d    = fullfile(sh_base, '1d');

% Shared (dimension-agnostic): mirt_dctn, mirt_idctn, solve_cubic
addpath(fullfile(sh_base, 'utils'));

% Shared (1D-specific)
addpath(sh1d);                                   % setup_problem
addpath(fullfile(sh1d, 'utils'));                 % admm_solve, s_*, precomp_banded_proj, ...
addpath(fullfile(sh1d, 'problems'));              % prob_gaussian, analytical_sb_gaussian, ...

% Local folders (discretize_first/1d_spectral/)
addpath(base);
addpath(fullfile(base, 'config'));
addpath(fullfile(base, 'discretization'));
addpath(fullfile(base, 'pipelines'));
addpath(fullfile(base, 'prox'));
addpath(fullfile(base, 'projection'));
addpath(fullfile(base, 'utils'));
addpath(fullfile(base, 'problems'));
addpath(fullfile(base, 'experiments'));
