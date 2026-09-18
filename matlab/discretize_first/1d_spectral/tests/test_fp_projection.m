% TEST_FP_PROJECTION  Verify proj_fokker_planck_spectral correctness.
%
%   Three properties that must hold for a correct projection:
%
%   1. BC satisfaction  -- rho_out(1,:) = rho0, rho_out(nt+1,:) = rho1
%
%   2. FP residual = 0  -- the projected (rho, mx) satisfies the discretised
%                          FP equation at every interior time step
%
%   3. Idempotency      -- proj(proj(u)) = proj(u) to machine precision
%                          (a second projection should change nothing)
%
%   Run for both FE and BE configs.

clear;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

for time_disc = {'forward_euler', 'backward_euler'}
    td = time_disc{1};
    fprintf('=== test_fp_projection  [%s] ===\n\n', td);

    %% Setup
    if strcmp(td, 'forward_euler')
        cfg = cfg_admm_spectral_fe();
    else
        cfg = cfg_admm_spectral_be();
    end
    cfg.nt = 16;   cfg.nx = 32;   % small grid for speed

    prob_def = prob_gaussian();
    problem  = setup_problem_spectral(cfg, prob_def);

    nt = problem.nt;  nx = problem.nx;
    dt = problem.dt;  dx = problem.dx;
    ops = problem.ops;
    rho0 = problem.rho0;  rho1 = problem.rho1;

    tol_bc   = 1e-10;
    tol_fp   = 1e-10;
    tol_idem = 1e-10;

    %% Random input (nt+1 x nx for both rho and mx)
    rng(42);
    x_in.rho = abs(randn(nt+1, nx)) + 0.1;   % keep rho positive
    x_in.mx  = randn(nt+1, nx);

    x_out = proj_fokker_planck_spectral(x_in, problem, cfg);

    %% --- Test 1: BC satisfaction ---
    err_rho0 = max(abs(x_out.rho(1,:)    - rho0));
    err_rho1 = max(abs(x_out.rho(nt+1,:) - rho1));
    report('BC: rho(1,:) = rho0',    err_rho0, tol_bc);
    report('BC: rho(nt+1,:) = rho1', err_rho1, tol_bc);

    %% --- Test 2: FP residual ---
    % For each interval n=1,...,nt compute the FP residual in physical space.
    % FE: (rho(n+1)-rho(n))/dt + deriv_x(mx(n))   - eps*deriv_xx(rho(n))   = 0
    % BE: (rho(n+1)-rho(n))/dt + deriv_x(mx(n+1)) - eps*deriv_xx(rho(n+1)) = 0
    rho = x_out.rho;   mx = x_out.mx;
    use_be = strcmp(td, 'backward_euler');

    fp_res = zeros(nt, nx);
    for n = 1:nt
        drho_dt = (rho(n+1,:) - rho(n,:)) / dt;
        if use_be
            div_mx   = ops.deriv_x(mx(n+1,:));
            lap_rho  = ops.deriv_xx(rho(n+1,:));
        else
            div_mx   = ops.deriv_x(mx(n,:));
            lap_rho  = ops.deriv_xx(rho(n,:));
        end
        fp_res(n,:) = drho_dt + div_mx - cfg.vareps * lap_rho;
    end
    err_fp = max(abs(fp_res(:)));
    report('FP residual = 0 (all intervals)', err_fp, tol_fp);

    %% --- Test 3: Idempotency ---
    x_out2 = proj_fokker_planck_spectral(x_out, problem, cfg);
    err_rho = max(abs(x_out2.rho(:) - x_out.rho(:)));
    err_mx  = max(abs(x_out2.mx(:)  - x_out.mx(:)));
    report('Idempotency: rho unchanged by 2nd proj', err_rho, tol_idem);
    report('Idempotency: mx  unchanged by 2nd proj', err_mx,  tol_idem);

    %% --- Bonus: analytical solution should be near-feasible ---
    % The analytical solution satisfies the CONTINUOUS FP eq; the discrete
    % residual is O(dt) for FE/BE and O(dt^2) with midpoint rule.
    if cfg.vareps > 0 || true
        [rho_a, ~] = analytical_sb_gaussian_spectral(problem, cfg.vareps);
        x_ana.rho  = rho_a;
        x_ana.mx   = zeros(nt+1, nx);   % only rho is used here

        fp_res_ana = zeros(nt, nx);
        for n = 1:nt
            drho_dt = (rho_a(n+1,:) - rho_a(n,:)) / dt;
            if use_be
                % BE uses analytical mx at right edge -- compute from formula
                [~, mx_a] = analytical_sb_gaussian_spectral(problem, cfg.vareps);
                lap_rho   = ops.deriv_xx(rho_a(n+1,:));
                div_mx    = ops.deriv_x(mx_a(n+1,:));
            else
                [~, mx_a] = analytical_sb_gaussian_spectral(problem, cfg.vareps);
                lap_rho   = ops.deriv_xx(rho_a(n,:));
                div_mx    = ops.deriv_x(mx_a(n,:));
            end
            fp_res_ana(n,:) = drho_dt + div_mx - cfg.vareps * lap_rho;
        end
        err_ana = max(abs(fp_res_ana(:)));
        fprintf('  INFO  %-45s  err = %.2e  (expect O(dt)=%.2e)\n', ...
            'FP residual of analytical solution', err_ana, dt);
    end

    fprintf('\n');
end

fprintf('Done.\n');

%% -----------------------------------------------------------------------
function report(name, err, tol)
    if err < tol
        fprintf('  PASS  %-48s  err = %.2e\n', name, err);
    else
        fprintf('  FAIL  %-48s  err = %.2e  (tol = %.2e)\n', name, err, tol);
    end
end
