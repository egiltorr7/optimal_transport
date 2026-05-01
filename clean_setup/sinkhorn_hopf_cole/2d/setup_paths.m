% SETUP_PATHS  Add all required folders to the MATLAB path.
%
%   Run this once at the start of any session working in sinkhorn_hopf_cole/2d/.

% mfilename('fullpath') is unreliable when called via run() — use a caller-
% supplied base if one is available, otherwise fall back to mfilename.
if ~exist('setup_paths_base', 'var')
    setup_paths_base = fileparts(mfilename('fullpath'));
end
base    = setup_paths_base;
sh_base = fullfile(base, '..', '..', 'shared');
sh2d    = fullfile(sh_base, '2d');

% Shared (2D-specific)
addpath(sh2d);                                   % setup_problem
addpath(fullfile(sh2d, 'utils'));                 % precomp_heat_neumann_2d, precomp_heat_free_space_2d, ...
addpath(fullfile(sh2d, 'problems'));              % prob_*, analytical_*
addpath(fullfile(sh2d, 'discretization'));        % disc_staggered_1st
addpath(fullfile(sh2d, 'pipelines'));             % sinkhorn_hopf_cole

% Local (sinkhorn_hopf_cole-specific)
addpath(base);
addpath(fullfile(base, 'config'));
addpath(fullfile(base, 'experiments'));
