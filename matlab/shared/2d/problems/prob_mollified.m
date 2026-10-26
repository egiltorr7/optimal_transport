function prob = prob_mollified(matfile)
% PROB_MOLLIFIED  Problem definition from a PRECOMPUTED mollified density pair.
%
%   prob = prob_mollified('.../pikachu_raichu_w0.04_128.mat')
%
%   Same contract as prob_gaussian / prob_silhouette, but the marginals are
%   read from a .mat written by make_mollified_marginals.m instead of being
%   rebuilt from the source sprites. Use this for production runs: the
%   interp2 resample plus the bump convolution are pure setup, identical on
%   every run, and there is no reason to pay for them (or to risk them
%   silently changing) once they are settled.
%
%   The stored arrays are tied to ONE grid. rho0_func/rho1_func therefore
%   check the grid they are handed and refuse a mismatch rather than
%   returning a wrong-sized array for setup_problem to fail on obscurely --
%   asking a 128 cache for a 256 run is the easy mistake here.
%
%   Regenerate with make_mollified_marginals.m if mollify_w, margin or the
%   source images change.

    arguments
        matfile (1,:) char
    end
    if exist(matfile, 'file') ~= 2
        error('prob_mollified:missingCache', ...
              ['cache not found: %s\n' ...
               'Run make_mollified_marginals.m to build it.'], matfile);
    end
    S = load(matfile);
    for f = {'rho0','rho1','nx','ny','mollify_w','margin','name'}
        if ~isfield(S, f{1})
            error('prob_mollified:badCache', ...
                  '%s is missing field "%s" -- regenerate it', matfile, f{1});
        end
    end

    prob.name       = S.name;
    prob.rho0_func  = @(xx, yy) local_fetch(S.rho0, xx, yy, S, matfile, 0);
    prob.rho1_func  = @(xx, yy) local_fetch(S.rho1, xx, yy, S, matfile, 1);
    % Carried through so an experiment can record what it actually ran with.
    prob.mollify_w  = S.mollify_w;
    prob.margin     = S.margin;
    prob.source_mat = matfile;
end


function r = local_fetch(r, xx, yy, S, matfile, which)
    if numel(xx) ~= S.nx || numel(yy) ~= S.ny
        error('prob_mollified:gridMismatch', ...
              ['rho%d cache is %dx%d but this run asks for %dx%d.\n' ...
               'Cache: %s\n' ...
               'Load the cache built for this grid, or rebuild with ' ...
               'make_mollified_marginals.m.'], ...
              which, S.nx, S.ny, numel(xx), numel(yy), matfile);
    end
end
