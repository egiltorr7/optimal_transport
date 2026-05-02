% BATCH_POSTPROCESS  Generate figures for all result_*.mat files not yet processed.
%
%   A result is considered processed if density_<ftag>.png already exists
%   in the figures directory.  Safe to re-run — already-processed files are
%   skipped.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
fig_dir = fullfile(res_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

mats = dir(fullfile(res_dir, 'result_*.mat'));
if isempty(mats)
    fprintf('No result_*.mat files found in %s\n', res_dir);
    return;
end

n_total     = numel(mats);
n_processed = 0;
n_skipped   = 0;
n_failed    = 0;

for i = 1:n_total
    MAT_FILE = fullfile(res_dir, mats(i).name);
    ftag     = strrep(strrep(mats(i).name, 'result_', ''), '.mat', '');
    sentinel = fullfile(fig_dir, sprintf('density_%s.png', ftag));

    if exist(sentinel, 'file')
        fprintf('[%d/%d] Skipping (already done): %s\n', i, n_total, mats(i).name);
        n_skipped = n_skipped + 1;
        continue;
    end

    fprintf('[%d/%d] Processing: %s\n', i, n_total, mats(i).name);
    try
        generate_result_figures(MAT_FILE, fig_dir);
        n_processed = n_processed + 1;
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        n_failed = n_failed + 1;
    end
end

fprintf('\nDone.  Processed: %d   Skipped: %d   Failed: %d   Total: %d\n', ...
    n_processed, n_skipped, n_failed, n_total);
fprintf('Figures in: %s\n', fig_dir);
