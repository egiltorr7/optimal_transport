function problem = setup_problem_spectral(cfg, prob_def)
% SETUP_PROBLEM_SPECTRAL  Build the problem struct for spectral 1D OT / SB.
%
%   problem = setup_problem_spectral(cfg, prob_def)
%
%   Calls setup_problem for the shared grid / density setup, then overrides
%   the spatial grid to use FFT-compatible node points  x_j = j/nx (j=0..nx-1)
%   which include x=0 but not x=1 (correct for the periodic FFT on [0,1)).
%   setup_problem uses cell-centres suited to the DCT/staggered scheme; here
%   we re-evaluate rho0 and rho1 at the node grid before precomputing the
%   spectral projection factors.
%
%   Requires:
%     cfg.disc       = @disc_spectral_1d
%     cfg.projection = @proj_fokker_planck_spectral
%     cfg.vareps     scalar regularisation parameter
%     cfg.time_disc  'forward_euler' | 'backward_euler'

    % Base grid / ops from shared setup_problem
    problem = setup_problem(cfg, prob_def);

    % Override spatial grid: node points x_j = (j-1)/nx, j=1..nx
    % i.e. [0, dx, 2*dx, ..., (nx-1)*dx]  (includes 0, excludes 1)
    nx = problem.nx;
    dx = problem.dx;
    problem.xx = (0 : nx-1) * dx;          % (1 x nx)  node points

    % Re-evaluate and re-normalise boundary densities on the node grid
    rho0 = prob_def.rho0_func(problem.xx);
    rho1 = prob_def.rho1_func(problem.xx);
    problem.rho0 = rho0 / sum(rho0);
    problem.rho1 = rho1 / sum(rho1);

    % Store time-stepping scheme so prox and other functions can query it
    problem.time_disc = cfg.time_disc;

    % Precompute per-mode projection factors
    if isequal(cfg.projection, @proj_fokker_planck_spectral)
        problem.spectral_proj = precomp_spectral_proj(problem, cfg);
    end
end
