% POSTPROCESS_SB_GAUSSIAN  Load a saved result and generate all figures locally.
%
%   Run from experiments/ after pulling the .mat file from the remote machine:
%
%     rsync user@remote:.../results/result_*.mat ./results/
%
%   Then run this script.  Set fields in the 'sel' struct to filter which
%   result to load (nt, nx, ny, eps, gam, tau).  Leave a field empty ([])
%   to match any value.  If multiple files match, the most recent is used.

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
fig_dir = fullfile(res_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Select result ---
% Set any field to filter; leave empty ([]) to match anything.
% If multiple files match, the most recent is loaded.
sel.nt  = 128;      % e.g. 64
sel.nx  = 128;      % e.g. 128
sel.ny  = 128;      % e.g. 128
sel.eps = 0.1;      % e.g. 0.01
sel.gam = 100;      % e.g. 100
sel.tau = 101;      % e.g. 101

mats = dir(fullfile(res_dir, 'result_*.mat'));
if isempty(mats)
    error('No result_*.mat files found in %s', res_dir);
end

keep = true(numel(mats), 1);
tokens = {'nt','nx','ny','gam','tau','eps'};
for i = 1:numel(mats)
    for j = 1:numel(tokens)
        f   = tokens{j};
        val = sel.(f);
        if isempty(val), continue; end
        tok = regexp(mats(i).name, [f '(\d+\.?\d*(?:[eE][+-]?\d+)?)'], 'tokens', 'once');
        if isempty(tok) || abs(str2double(tok{1}) - val) > 1e-10*(abs(val)+1)
            keep(i) = false;  break;
        end
    end
end

mats = mats(keep);
if isempty(mats)
    error('No result_*.mat files match the selection criteria.');
end
if numel(mats) > 1
    fprintf('Multiple matches — loading most recent:\n');
    for i = 1:numel(mats), fprintf('  %s\n', mats(i).name); end
end
[~, idx] = max([mats.datenum]);
MAT_FILE = fullfile(res_dir, mats(idx).name);

generate_result_figures(MAT_FILE, fig_dir);
