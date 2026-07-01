function [t_grid, dt_vec] = make_uniform_tgrid(nt, ~)
% MAKE_UNIFORM_TGRID  Uniform time grid on [0,1].
%
%   Drop-in replacement for make_nonuniform_tgrid when no refinement is
%   needed (e.g. small vareps where the power law gives p=1 anyway).
%   The second argument (vareps) is ignored.

    t_grid = linspace(0, 1, nt + 1)';
    dt_vec = diff(t_grid);
end
