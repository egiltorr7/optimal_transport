% SETUP_PATHS  Add all required folders to the MATLAB path.
%
%   Run this once at the start of any session working in sinkhorn_hopf_cole/1d/.

if ~exist('setup_paths_base', 'var')
    setup_paths_base = fileparts(mfilename('fullpath'));
end
base    = setup_paths_base;
sh_base = fullfile(base, '..', '..', 'shared');
sh1d    = fullfile(sh_base, '1d');

% Shared (dimension-agnostic): mirt_dctn, mirt_idctn, solve_cubic
addpath(fullfile(sh_base, 'utils'));

% Shared (1D-specific)
addpath(sh1d);                                   % setup_problem
addpath(fullfile(sh1d, 'utils'));                 % ladmm_solve, precomp_banded_proj, ...
addpath(fullfile(sh1d, 'problems'));              % prob_*, analytical_*
addpath(fullfile(sh1d, 'discretization'));        % disc_staggered_1st
addpath(fullfile(sh1d, 'pipelines'));             % sinkhorn_hopf_cole

% Local (sinkhorn_hopf_cole-specific)
addpath(base);
addpath(fullfile(base, 'config'));
addpath(fullfile(base, 'experiments'));
