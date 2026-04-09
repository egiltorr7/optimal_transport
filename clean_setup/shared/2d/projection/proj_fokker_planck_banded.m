function x_out = proj_fokker_planck_banded(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_BANDED  Project onto the FP constraint via DCT-xy + tridiagonal-t.
%
%   x_out = proj_fokker_planck_banded(x_in, problem, cfg)
%
%   Solves AA*phi = f exactly using:
%     1. Compute FP residual  f = d_t mu + d_x psi_x + d_y psi_y - eps*Delta mu
%     2. 2D DCT in x and y
%     3. Batched Thomas (TDMA) solve for all (kx,ky) modes simultaneously
%        (kx=1,ky=1) DC mode is handled via DCT-t and then inserted
%     4. 2D IDCT back to physical space
%     5. Apply A* corrections
%
%   Requires problem.banded_proj from precomp_banded_proj.
%   All operations are element-wise; compatible with gpuArray inputs.

    ops    = problem.ops;
    rho0   = problem.rho0;
    rho1   = problem.rho1;
    nt     = problem.nt;
    nx     = problem.nx;
    ny     = problem.ny;
    vareps = cfg.vareps;
    bp     = problem.banded_proj;

    mu    = x_in.rho;
    psi_x = x_in.mx;
    psi_y = x_in.my;

    zeros_x = zeros(nt, ny, 'like', mu);
    zeros_y = zeros(nt, nx, 'like', mu);

    %% --- FP residual  f = d_t mu + d_x psi_x + d_y psi_y - eps*Delta mu ---
    mu_phi   = ops.interp_t_at_phi(mu, rho0, rho1);
    nabla_mu = ops.deriv_x_at_phi(ops.deriv_x_at_m(mu_phi), zeros_x, zeros_x) ...
             + ops.deriv_y_at_phi(ops.deriv_y_at_m(mu_phi), zeros_y, zeros_y);

    f = ops.deriv_t_at_phi(mu, rho0, rho1) ...
      + ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x) ...
      + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y) ...
      - vareps * nabla_mu;

    if norm(f(:)) * sqrt(problem.dt * problem.dx * problem.dy) < 1e-12
        x_out = x_in;
        return;
    end

    %% --- 2D DCT in x and y ---
    f_hat = dct2_xy(f, nt, nx, ny);   % (nt x nx x ny)

    %% --- Thomas solve for all modes; zero RHS for (1,1) ---
    rhs          = f_hat;
    rhs(:, 1, 1) = 0;

    phi_hat = thomas_solve(bp.lower_all, bp.main_all, bp.upper_all, rhs, nt, nx, ny);

    %% --- (kx=1, ky=1): DC spatial mode, lxy=0 -> singular T -> DCT-t ---
    f1_t           = dct(f_hat(:, 1, 1));
    phi1_t         = zeros(nt, 1, 'like', f_hat);
    phi1_t(2:end)  = f1_t(2:end) ./ problem.lambda_t(2:end);
    phi_hat(:,1,1) = idct(phi1_t);

    %% --- 2D IDCT back to physical space ---
    phi = idct2_xy(phi_hat, nt, nx, ny);   % (nt x nx x ny)

    %% --- Apply A* corrections ---
    dphi_dx   = ops.deriv_x_at_m(phi);
    dphi_dy   = ops.deriv_y_at_m(phi);

    nabla_phi = ops.interp_t_at_rho( ...
        ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x) + ...
        ops.deriv_y_at_phi(dphi_dy, zeros_y, zeros_y));

    x_out.rho = mu    + ops.deriv_t_at_rho(phi) + vareps .* nabla_phi;
    x_out.mx  = psi_x + dphi_dx;
    x_out.my  = psi_y + dphi_dy;
end

% ---------------------------------------------------------------------------

function phi_hat = thomas_solve(lower_all, main_all, upper_all, f_hat, nt, nx, ny)
% THOMAS_SOLVE  Batched Thomas (TDMA) algorithm for all (kx,ky) modes at once.
%
%   lower_all  (nt-1 x nx x ny)  lower diagonals
%   main_all   (nt   x nx x ny)  main  diagonals
%   upper_all  (nt-1 x nx x ny)  upper diagonals
%   f_hat      (nt   x nx x ny)  right-hand sides
%
%   Solves T_{kx,ky} * phi(:,kx,ky) = f_hat(:,kx,ky) for every (kx,ky)
%   simultaneously via nt sequential element-wise sweeps over (nx x ny).
%   Compatible with gpuArray inputs.

    % Work copies (forward sweep modifies main and RHS in place)
    b = main_all;
    d = f_hat;

    % Forward sweep
    for i = 2:nt
        w        = lower_all(i-1,:,:) ./ b(i-1,:,:);
        b(i,:,:) = b(i,:,:) - w .* upper_all(i-1,:,:);
        d(i,:,:) = d(i,:,:) - w .* d(i-1,:,:);
    end

    % Back substitution
    phi_hat = zeros(nt, nx, ny, 'like', f_hat);
    phi_hat(nt,:,:) = d(nt,:,:) ./ b(nt,:,:);
    for i = nt-1:-1:1
        phi_hat(i,:,:) = (d(i,:,:) - upper_all(i,:,:) .* phi_hat(i+1,:,:)) ./ b(i,:,:);
    end
end

% ---------------------------------------------------------------------------

function f_hat = dct2_xy(f, nt, nx, ny)
% DCT-II along dim 2 (x) and dim 3 (y) of a (nt x nx x ny) array.
    f2 = permute(f, [2, 1, 3]);
    f2 = reshape(f2, nx, nt*ny);
    f2 = dct(f2);
    f2 = reshape(f2, nx, nt, ny);
    f2 = permute(f2, [2, 1, 3]);

    f3 = permute(f2, [3, 1, 2]);
    f3 = reshape(f3, ny, nt*nx);
    f3 = dct(f3);
    f3 = reshape(f3, ny, nt, nx);
    f_hat = permute(f3, [2, 3, 1]);
end

function f = idct2_xy(f_hat, nt, nx, ny)
% IDCT-II along dim 2 (x) and dim 3 (y) of a (nt x nx x ny) array.
    f3 = permute(f_hat, [3, 1, 2]);
    f3 = reshape(f3, ny, nt*nx);
    f3 = idct(f3);
    f3 = reshape(f3, ny, nt, nx);
    f2 = permute(f3, [2, 3, 1]);

    f2 = permute(f2, [2, 1, 3]);
    f2 = reshape(f2, nx, nt*ny);
    f2 = idct(f2);
    f2 = reshape(f2, nx, nt, ny);
    f  = permute(f2, [2, 1, 3]);
end
