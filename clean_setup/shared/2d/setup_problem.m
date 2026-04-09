function problem = setup_problem(cfg, prob_def)
% SETUP_PROBLEM  Build the problem struct from a config and problem definition.
%
%   problem = setup_problem(cfg, prob_def)
%
%   problem fields:
%     nt, nx, ny, dt, dx, dy         grid dimensions and step sizes
%     xx, yy                         cell-center spatial coordinates  
%     rho0, rho1                     boundary densities               (nx x ny)
%     lambda_x, lambda_y, lambda_t   DCT eigenvalues for spectral solve
%     ops                            staggered-grid operator handles (from cfg.disc)
%     name                           combined identifier, e.g. 'gaussian_staggered_gaussian'

    problem.nt = cfg.nt;
    problem.nx = cfg.nx;
    problem.ny = cfg.ny;
    problem.dt = 1 / cfg.nt;
    problem.dx = 1 / cfg.nx;
    problem.dy = 1 / cfg.ny;

    ntm = problem.nt - 1;
    nxm = problem.nx - 1;
    nym = problem.ny - 1;

    % Cell-center spatial grid
    x = linspace(0, 1, problem.nx + 1)';
    problem.xx = (x(2:end) + x(1:end-1)) / 2;   % (nx x 1)
    y = linspace(0, 1, problem.ny + 1);
    problem.yy = (y(2:end) + y(1:end-1)) / 2;   % (1 x ny)

    % Boundary densities from problem definition
    rho0 = prob_def.rho0_func(problem.xx, problem.yy);
    rho1 = prob_def.rho1_func(problem.xx, problem.yy);
    problem.rho0 = rho0 / (sum(rho0(:))*problem.dx*problem.dy);
    problem.rho1 = rho1 / (sum(rho1(:))*problem.dx*problem.dy);

    % Combined name for saving results
    problem.name = sprintf('%s_%s', prob_def.name, cfg.name);

    % DCT eigenvalues for the spectral solver in projection
    problem.lambda_y = (2 - 2*cos(pi * problem.dy * (0:nym)))  / problem.dy^2;  % (1 x ny)
    problem.lambda_x = (2 - 2*cos(pi * problem.dx * (0:nxm)))  / problem.dx^2;  % (1 x nx)
    problem.lambda_t = (2 - 2*cos(pi * problem.dt * (0:ntm)')) / problem.dt^2;  % (nt x 1)

    % Precomputed staggered-grid operators (discretization scheme from config)
    problem.ops = cfg.disc(problem);
end
