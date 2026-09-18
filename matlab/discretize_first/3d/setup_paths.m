% SETUP_PATHS  Add all required folders to the MATLAB path for discretize_first/3d.
%
%   Run this once at the start of any 3D session.
%
%   3D-specific paths are added BEFORE 2D paths so that same-named functions
%   (precomp_expsemi_proj, proj_fokker_planck_expsemi_gpu, setup_problem, etc.)
%   resolve to the 3D version.

base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', 'shared');
sh2d    = fullfile(sh_base, '2d');
sh3d    = fullfile(sh_base, '3d');

% Shared (dimension-agnostic): solve_cubic
addpath(fullfile(sh_base, 'utils'));

% Shared (3D-specific) — must come before sh2d for name-shadowing to work
addpath(sh3d);                                  % setup_problem (3D)
addpath(fullfile(sh3d, 'utils'));               % precomp_expsemi_proj (3D)
addpath(fullfile(sh3d, 'problems'));            % prob_gaussian, analytical_sb_gaussian
addpath(fullfile(sh3d, 'discretization'));      % disc_staggered_1st_3d
addpath(fullfile(sh3d, 'prox'));               % prox_ke_cc
addpath(fullfile(sh3d, 'projection'));         % proj_fokker_planck_expsemi_gpu (3D)
addpath(fullfile(sh3d, 'pipelines'));          % discretize_then_optimize (3D)

% Shared (2D utilities reused verbatim): ladmm_solve, s_zeros, s_scale, s_sub, s_add
addpath(fullfile(sh2d, 'utils'));

% Local (experiment-specific)
addpath(base);
addpath(fullfile(base, 'config'));
addpath(fullfile(base, 'experiments'));
