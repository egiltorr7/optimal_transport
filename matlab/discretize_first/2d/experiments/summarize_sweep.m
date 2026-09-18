% SUMMARIZE_SWEEP  Print a timing/convergence table for all result_*.mat files.
%
%   Run from any directory after results are synced locally.
%   Prints a table sorted by grid size then eps.

clear;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
mats    = dir(fullfile(res_dir, 'result_*.mat'));

if isempty(mats)
    fprintf('No result_*.mat files found in %s\n', res_dir);
    return;
end

rows = struct('nt',{}, 'nx',{}, 'ny',{}, 'eps',{}, ...
              'iters',{}, 'converged',{}, 'error',{}, ...
              'walltime',{}, 'time_per_iter',{}, 'throughput',{}, ...
              'gpu_mb',{}, 'obj',{}, 'fname',{});

for i = 1:numel(mats)
    fpath = fullfile(res_dir, mats(i).name);
    try
        d = load(fpath, 'cfg', 'result', 'obj');
    catch
        fprintf('Skipping (load failed): %s\n', mats(i).name);
        continue;
    end
    r.nt           = d.cfg.nt;
    r.nx           = d.cfg.nx;
    r.ny           = d.cfg.ny;
    r.eps          = d.cfg.vareps;
    r.iters        = d.result.iters;
    r.converged    = d.result.converged;
    r.error        = d.result.error;
    r.walltime     = d.result.walltime;
    r.time_per_iter = d.result.time_per_iter;
    r.throughput   = d.result.throughput;
    r.gpu_mb       = d.result.gpu_total_mb;
    r.obj          = d.obj;
    r.fname        = mats(i).name;
    rows(end+1)    = r; %#ok<AGROW>
end

if isempty(rows)
    fprintf('No valid result files.\n');
    return;
end

% Sort by grid size (nt*nx*ny) then eps
nvox  = [rows.nt] .* [rows.nx] .* [rows.ny];
eps_v = [rows.eps];
[~, ord] = sortrows([nvox(:), eps_v(:)]);
rows = rows(ord);

% Print table
hdr = sprintf('%-6s %-6s %-6s %-8s %-8s %-5s %-10s %-10s %-12s %-10s %-10s\n', ...
    'nt','nx','ny','eps','wall(s)','iters','conv','error','t/iter(s)','tput(it/s)','obj');
fprintf('%s', repmat('-', 1, length(hdr)-1)); fprintf('\n');
fprintf('%s', hdr);
fprintf('%s', repmat('-', 1, length(hdr)-1)); fprintf('\n');

for i = 1:numel(rows)
    r = rows(i);
    fprintf('%-6d %-6d %-6d %-8.2g %-8.1f %-5d %-10d %-10.2e %-12.4f %-10.1f %-10.6f\n', ...
        r.nt, r.nx, r.ny, r.eps, r.walltime, r.iters, r.converged, ...
        r.error, r.time_per_iter, r.throughput, r.obj);
end
fprintf('%s\n', repmat('-', 1, length(hdr)-1));
fprintf('Total runs: %d\n', numel(rows));
