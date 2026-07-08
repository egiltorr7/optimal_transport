function problem = setup_problem(cfg, prob_def)
% SETUP_PROBLEM  Build the 3D problem struct from a config and problem definition.
%
%   problem = setup_problem(cfg, prob_def)
%
%   problem fields:
%     nt, nx, ny, nz            grid point counts
%     dt, dx, dy, dz            step sizes
%     xx, yy, zz                cell-centre spatial coordinates
%     rho0, rho1                boundary densities  (nx x ny x nz)
%     lambda_x, lambda_y, lambda_z, lambda_t   DCT eigenvalues
%     ops                       staggered-grid operator handles (from cfg.disc)
%     name                      combined identifier string

    problem.nt = cfg.nt;
    problem.nx = cfg.nx;
    problem.ny = cfg.ny;
    problem.nz = cfg.nz;
    problem.dt = 1 / cfg.nt;
    problem.dx = 1 / cfg.nx;
    problem.dy = 1 / cfg.ny;
    problem.dz = 1 / cfg.nz;

    ntm = problem.nt - 1;
    nxm = problem.nx - 1;
    nym = problem.ny - 1;
    nzm = problem.nz - 1;

    % Cell-centre spatial grids
    x = linspace(0, 1, problem.nx + 1)';
    problem.xx = (x(2:end) + x(1:end-1)) / 2;   % (nx x 1)
    y = linspace(0, 1, problem.ny + 1);
    problem.yy = (y(2:end) + y(1:end-1)) / 2;   % (1 x ny)
    z = reshape(linspace(0, 1, problem.nz + 1), 1, 1, []);
    problem.zz = (z(2:end) + z(1:end-1)) / 2;   % (1 x 1 x nz)

    % Boundary densities — prob_def.rho0_func(xx, yy, zz) returns (nx x ny x nz)
    rho0 = prob_def.rho0_func(problem.xx, problem.yy, problem.zz);
    rho1 = prob_def.rho1_func(problem.xx, problem.yy, problem.zz);
    problem.rho0_pdf = rho0;
    problem.rho1_pdf = rho1;
    dV = problem.dx * problem.dy * problem.dz;
    problem.rho0 = rho0 / (sum(rho0(:)) * dV);
    problem.rho1 = rho1 / (sum(rho1(:)) * dV);

    problem.name = sprintf('%s_%s', prob_def.name, cfg.name);

    % DCT eigenvalues for the spectral solver
    problem.lambda_x = (2 - 2*cos(pi * problem.dx * (0:nxm)))  / problem.dx^2;  % (1 x nx)
    problem.lambda_y = (2 - 2*cos(pi * problem.dy * (0:nym)))  / problem.dy^2;  % (1 x ny) -> stored as row
    problem.lambda_z = (2 - 2*cos(pi * problem.dz * (0:nzm)))  / problem.dz^2;  % (1 x nz)
    problem.lambda_t = (2 - 2*cos(pi * problem.dt * (0:ntm)')) / problem.dt^2;  % (nt x 1)

    % Staggered-grid operators from config
    problem.ops = cfg.disc(problem);
end
