% TEST_MASS_CONSERVATION  Check that the FP projection conserves total mass.
%
%   For the FP equation  d_t rho + d_x mx = eps * d_xx rho  on a periodic
%   domain, integrating in x gives  d_t (sum rho) = 0  because both
%   d_x mx and d_xx rho integrate to zero over a full period.
%
%   So after projection, dx * sum(rho(n,:)) should be constant in t
%   (equal to dx * sum(rho0) = 1 after normalisation in setup_problem).
%
%   This test also checks that the BC rows pin rho at t=0 and t=T
%   to the correct total mass.

clear;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

for time_disc = {'forward_euler', 'backward_euler'}
    td = time_disc{1};
    fprintf('=== test_mass_conservation  [%s] ===\n\n', td);

    if strcmp(td, 'forward_euler')
        cfg = cfg_admm_spectral_fe();
    else
        cfg = cfg_admm_spectral_be();
    end
    cfg.nt = 32;  cfg.nx = 64;

    prob_def = prob_gaussian();
    problem  = setup_problem_spectral(cfg, prob_def);

    nt = problem.nt;  dx = problem.dx;
    rho0 = problem.rho0;  rho1 = problem.rho1;

    %% Random input
    rng(7);
    x_in.rho = abs(randn(nt+1, problem.nx)) + 0.1;
    x_in.mx  = randn(nt+1, problem.nx);

    x_out = proj_fokker_planck_spectral(x_in, problem, cfg);

    %% Mass at each time step
    mass      = dx * sum(x_out.rho, 2);   % (nt+1 x 1)
    mass_rho0 = dx * sum(rho0);
    mass_rho1 = dx * sum(rho1);

    mass_variation = max(mass) - min(mass);
    tol = 1e-10;

    report('Mass constant in t (max - min)', mass_variation, tol);
    report('Mass(t=0)  = mass(rho0)',  abs(mass(1)    - mass_rho0), tol);
    report('Mass(t=T)  = mass(rho1)',  abs(mass(end)  - mass_rho1), tol);

    fprintf('  INFO  mass range: [%.6f, %.6f]  (rho0 mass = %.6f)\n\n', ...
        min(mass), max(mass), mass_rho0);
end

fprintf('Done.\n');

function report(name, err, tol)
    if err < tol
        fprintf('  PASS  %-48s  err = %.2e\n', name, err);
    else
        fprintf('  FAIL  %-48s  err = %.2e  (tol = %.2e)\n', name, err, tol);
    end
end
