% REPLOT_SELECTED
%
% Regenerates specific figures for a filtered subset of result .mat files.
% Edit FIG_NAMES and PROJ_FILTER below, then run.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

% -----------------------------------------------------------------------
% Configure what to regenerate
% -----------------------------------------------------------------------
FIG_NAMES   = {'admm_residual'};      % figures to regenerate (see generate_result_figures for names)
PROJ_FILTER = {'banded', 'spike2'};   % projection short-names to include (empty = all)
% -----------------------------------------------------------------------

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
fig_dir = fullfile(res_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

mats = dir(fullfile(res_dir, 'result_*.mat'));
if isempty(mats)
    fprintf('No result_*.mat files found in %s\n', res_dir);
    return;
end

n_done = 0;  n_skip = 0;  n_fail = 0;

for i = 1:numel(mats)
    MAT_FILE = fullfile(res_dir, mats(i).name);

    % Load only cfg to check projection without reading the full file
    tmp       = load(MAT_FILE, 'cfg');
    proj_str  = func2str(tmp.cfg.projection);
    proj_short = strrep(proj_str, 'proj_fokker_planck_', '');

    if ~isempty(PROJ_FILTER) && ~any(strcmp(proj_short, PROJ_FILTER))
        fprintf('[%d/%d] Skip (proj=%s): %s\n', i, numel(mats), proj_short, mats(i).name);
        n_skip = n_skip + 1;
        continue;
    end

    fprintf('[%d/%d] Replotting %s (proj=%s) ...\n', i, numel(mats), mats(i).name, proj_short);
    try
        generate_result_figures(MAT_FILE, fig_dir, FIG_NAMES);
        n_done = n_done + 1;
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        n_fail = n_fail + 1;
    end
end

fprintf('\nDone.  Updated: %d   Skipped: %d   Failed: %d\n', n_done, n_skip, n_fail);
