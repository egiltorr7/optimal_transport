function [t_grid, dt_vec] = make_nonuniform_tgrid(nt, vareps, pow_scale)
% MAKE_NONUNIFORM_TGRID  Power-law time grid refined near t=1.
%
%   [t_grid, dt_vec] = make_nonuniform_tgrid(nt, vareps)
%   [t_grid, dt_vec] = make_nonuniform_tgrid(nt, vareps, pow_scale)
%
%   Grid: t = 1 - (1-u)^p,  u = linspace(0,1,nt+1)
%   Power: p = max(1, 1 + pow_scale * log(max(vareps,1)) / log(nt))
%
%   pow_scale=1 (default): dt_min ~ 1/(nt*vareps).
%   pow_scale=2:           dt_min ~ 1/(nt*vareps^2), steeper near t=1.
%
%   Outputs:
%     t_grid  (nt+1 x 1)  node positions, t_grid(1)=0, t_grid(end)=1
%     dt_vec  (nt   x 1)  step sizes dt_vec(n) = t_grid(n+1) - t_grid(n)

    if nargin < 3, pow_scale = 1; end
    p      = max(1, 1 + pow_scale * log(max(vareps, 1)) / log(nt));
    u      = linspace(0, 1, nt + 1)';
    t_grid = 1 - (1 - u).^p;
    dt_vec = diff(t_grid);
end
