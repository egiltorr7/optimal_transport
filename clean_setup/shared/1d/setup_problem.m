function problem = setup_problem(cfg, prob_def)
% SETUP_PROBLEM  Build the problem struct from a config and problem definition.
%
%   problem = setup_problem(cfg, prob_def)
%
%   problem fields:
%     nt, nx, dt, dx       grid dimensions and step sizes
%     xx                   cell-center spatial coordinates  (1 x nx)
%     rho0, rho1           boundary densities               (1 x nx)
%     lambda_x, lambda_t   DCT eigenvalues for spectral solve
%     ops                  staggered-grid operator handles (from cfg.disc)
%     name                 combined identifier, e.g. 'gaussian_staggered_gaussian'

    problem.nt = cfg.nt;
    problem.nx = cfg.nx;
    problem.dt = 1 / cfg.nt;
    problem.dx = 1 / cfg.nx;

    ntm = problem.nt - 1;
    nxm = problem.nx - 1;

    % Cell-center spatial grid
    x = linspace(0, 1, problem.nx + 1);
    problem.xx = (x(2:end) + x(1:end-1)) / 2;   % (1 x nx)

    % Boundary densities from problem definition
    rho0 = prob_def.rho0_func(problem.xx);
    rho1 = prob_def.rho1_func(problem.xx);
    problem.rho0_pdf = rho0;              % raw PDF from prob_def (integrates to ~1 over R)
    problem.rho1_pdf = rho1;
    problem.rho0     = rho0 / sum(rho0); % discrete probability mass (sums to 1 over [0,1])
    problem.rho1     = rho1 / sum(rho1);

    % Combined name for saving results
    problem.name = sprintf('%s_%s', prob_def.name, cfg.name);

    % DCT eigenvalues for the spectral solver in projection
    problem.lambda_x = (2 - 2*cos(pi * problem.dx * (0:nxm)))  / problem.dx^2;  % (1 x nx)
    problem.lambda_t = (2 - 2*cos(pi * problem.dt * (0:ntm)')) / problem.dt^2;  % (nt x 1)

    % Precomputed staggered-grid operators (discretization scheme from config)
    problem.ops = cfg.disc(problem);
end
