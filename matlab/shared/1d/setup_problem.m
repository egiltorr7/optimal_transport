function problem = setup_problem(cfg, prob_def)
% SETUP_PROBLEM  Build the problem struct from a config and problem definition.
%
%   problem = setup_problem(cfg, prob_def)
%
%   problem fields:
%     nt, nx, dt, dx       grid dimensions and step sizes
%     L                    domain length (default 1)
%     xx                   cell-center spatial coordinates  (1 x nx)  on [0,L]
%     rho0, rho1           boundary densities               (1 x nx)
%     lambda_x, lambda_t   DCT eigenvalues for spectral solve
%     ops                  staggered-grid operator handles (from cfg.disc)
%     name                 combined identifier, e.g. 'gaussian_staggered_gaussian'
%
%   Optional cfg fields:
%     cfg.L   domain length (default 1); xx spans [0,L], dx=L/nx

    problem.nt = cfg.nt;
    problem.nx = cfg.nx;
    problem.dt = 1 / cfg.nt;

    L = 1;
    if isfield(cfg, 'L'), L = cfg.L; end
    problem.L  = L;
    problem.dx = L / cfg.nx;

    nt  = problem.nt;
    nx  = problem.nx;
    ntm = nt - 1;
    nxm = nx - 1;

    % Cell-center spatial grid on [0, L]
    x = linspace(0, L, nx + 1);
    problem.xx = (x(2:end) + x(1:end-1)) / 2;   % (1 x nx)

    % Boundary densities from problem definition
    rho0 = prob_def.rho0_func(problem.xx);
    rho1 = prob_def.rho1_func(problem.xx);
    problem.rho0_pdf = rho0;              % raw PDF from prob_def (integrates to ~1 over R)
    problem.rho1_pdf = rho1;
    problem.rho0     = rho0 / sum(rho0); % discrete probability mass (sums to 1 over domain)
    problem.rho1     = rho1 / sum(rho1);

    % Copy analytical parameters from prob_def if present (mu0, mu1, sigma)
    for fld = {'mu0','mu1','sigma'}
        if isfield(prob_def, fld{1}), problem.(fld{1}) = prob_def.(fld{1}); end
    end

    % Combined name for saving results
    problem.name = sprintf('%s_%s', prob_def.name, cfg.name);

    % DCT eigenvalues for the spectral solver in projection.
    % Correct formula for [0,L] with Neumann BCs:
    %   lambda_k = (k*pi/L)^2  (continuous),  discrete: (2-2cos(k*pi/nx))/dx^2
    % Mode index k = 0..nx-1 (0..nt-1), normalised by grid count (not by dx).
    problem.lambda_x = (2 - 2*cos(pi * (0:nxm)  / nx)) / problem.dx^2;  % (1 x nx)
    problem.lambda_t = (2 - 2*cos(pi * (0:ntm)' / nt)) / problem.dt^2;  % (nt x 1)

    % Precomputed staggered-grid operators (discretization scheme from config)
    problem.ops = cfg.disc(problem);
end
