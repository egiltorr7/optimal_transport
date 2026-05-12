% TEST_SCHUR_DIAGNOSTIC
%
% Sweeps over (nt, nx/ny, eps/dt) and reports two linked diagnostics:
%
%   Matrix diagnostic (no random input needed):
%     min_S        -- min Schur complement min_{kx,ky} S(kx,ky)
%     min_Tnn      -- min diagonal entry   min_{kx,ky} T(nt,nt,kx,ky)
%     ratio        -- min_S / min_Tnn  (how much S is suppressed vs the diagonal)
%
%   Projection diagnostic (random x_in, spike2 solve):
%     fp_max       -- max |FP residual| over all (t,x,y) after projection
%     fp_tT        -- max |FP residual| at the t=T boundary slice only
%     fp_int       -- max |FP residual| at interior time slices (t != T)
%     grow_ratio   -- fp_tT / fp_int  (> 1 means residual grows toward t=T)
%
%   Verification diagnostic (backward-error decomposition):
%     norm_f       -- ||f||_F : Frobenius norm of the RHS f = FP-residual(x_in)
%     max_cond     -- max_{(kx,ky) != (1,1)} cond(T_{kx,ky})
%     predicted    -- machine_eps * max_cond * norm_f  (backward-error upper bound)
%     ratio_pred   -- fp_max / predicted  (should be O(1) and roughly constant in eps)
%
% The backward-error claim: fp_max <= machine_eps * max_cond * norm_f.
% If ratio_pred stays O(1) while fp_max grows, then both max_cond and norm_f
% are growing with eps and their product drives the residual.

clear; clc;

base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', '..', 'shared');
addpath(fullfile(sh_base, 'utils'));
addpath(fullfile(sh_base, '2d', 'discretization'));
addpath(fullfile(sh_base, '2d', 'projection'));
addpath(fullfile(sh_base, '2d', 'utils'));

rng(42);

%% --- Sweep parameters ---
% vareps_vals: absolute epsilon values (not eps/dt ratios).
% eps/dt is derived per (nt, vareps) pair for display.
nt_vals     = [16, 32, 64];
nxy_vals    = [8, 16, 32];     % nx = ny for simplicity
vareps_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10];

%% --- Table header ---
fprintf('%-4s  %-4s  %-10s  %-8s  %-12s  %-12s  %-10s  %-12s  %-12s  %-12s  %-10s  %-12s  %-12s  %-12s  %-10s\n', ...
    'nt', 'nxy', 'eps', 'eps/dt', ...
    'min_S', 'min_T(nt,nt)', 'S/Tnn', ...
    'fp_max', 'fp_tT', 'fp_int', 'grow_ratio', ...
    'norm_f', 'max_cond', 'predicted', 'ratio_pred');
fprintf('%s\n', repmat('-', 1, 175));

for nt = nt_vals
    dt  = 1 / nt;
    ntm = nt - 1;

    for nxy = nxy_vals
        nx = nxy;  ny = nxy;
        nxm = nx - 1;  nym = ny - 1;
        dx = 1/nx;  dy = 1/ny;

        % --- Minimal problem struct ---
        prob.nt = nt;  prob.nx = nx;  prob.ny = ny;
        prob.dt = dt;  prob.dx = dx;  prob.dy = dy;
        prob.lambda_t = (2 - 2*cos(pi*dt*(0:ntm)' )) / dt^2;
        prob.lambda_x = (2 - 2*cos(pi*dx*(0:nxm)  )) / dx^2;
        prob.lambda_y = (2 - 2*cos(pi*dy*(0:nym)   )) / dy^2;

        % Gaussian marginals
        xx = ((1:nx)' - 0.5) * dx;
        yy = ((1:ny)  - 0.5) * dy;
        G  = @(x, mu, s) exp(-0.5*((x - mu)/s).^2);
        rho0 = G(xx, 0.3, 0.08) .* G(yy, 0.5, 0.10);   % xx col (nx×1), yy row (1×ny) → (nx×ny)
        rho1 = G(xx, 0.7, 0.08) .* G(yy, 0.5, 0.10);
        prob.rho0 = rho0 / sum(rho0(:));
        prob.rho1 = rho1 / sum(rho1(:));
        prob.ops  = disc_staggered_1st(prob);

        for vareps = vareps_vals
            eps_dt     = vareps / dt;   % ratio for display only
            cfg.vareps = vareps;

            bp = precomp_banded_proj_spike2(prob, vareps);

            %% --- Matrix diagnostic: Schur complement ---
            % S(kx,ky) = T(nt,nt) - T(nt,nt-1) * spike_v(nt-1)
            S    = bp.main_all(nt,:,:) - bp.lower_all(ntm,:,:) .* bp.spike_v(ntm,:,:);
            Tnn  = bp.main_all(nt,:,:);

            % Exclude DC mode (kx=1,ky=1): handled by DCT-t in spike2, not Schur.
            S_vals   = S(:);    S_vals(1)   = [];
            Tnn_vals = Tnn(:);  Tnn_vals(1) = [];

            min_S   = min(S_vals);
            min_Tnn = min(Tnn_vals);
            ratio   = min_S / min_Tnn;

            %% --- Projection diagnostic: FP residual after spike2 ---
            t_rho   = reshape((1:ntm)*dt, ntm, 1, 1);
            rho0_3d = reshape(prob.rho0, 1, nx, ny);
            rho1_3d = reshape(prob.rho1, 1, nx, ny);
            x_in.rho = (1-t_rho).*rho0_3d + t_rho.*rho1_3d + 0.01*randn(ntm, nx, ny);
            x_in.mx  = 0.01*randn(nt, nxm, ny);
            x_in.my  = 0.01*randn(nt, nx,  nym);

            prob.banded_proj = bp;
            x_out = proj_fokker_planck_spike2(x_in, prob, cfg);

            fp_res = fp_residual(x_out, prob, vareps);

            fp_max = max(abs(fp_res(:)));
            fp_tT  = max(abs(reshape(fp_res(end,:,:), [], 1)));   % last phi time = t=T
            if nt > 2
                fp_int = max(abs(reshape(fp_res(1:end-1,:,:), [], 1)));
            else
                fp_int = NaN;
            end
            grow_ratio = fp_tT / max(fp_int, eps(fp_int));

            %% --- Verification diagnostic: backward-error decomposition ---
            % Claim: fp_max <= machine_eps * max_cond(T) * ||f||_F
            %
            % norm_f: RHS norm of f = FP-residual(x_in) before projection.
            %   f grows with eps because it contains -eps*Laplacian(rho),
            %   whose magnitude scales as eps * ||rho|| / dx^2.
            %
            % max_cond: condition number of the worst (kx,ky) tridiagonal T.
            %   T is symmetric (M0 = Dt_phi*Dt_phi^T, M2 = eps^2*It_phi*It_phi^T)
            %   so eig gives real eigenvalues and cond = max|ev| / min|ev|.
            %   T's condition grows with eps because M1d(nt) = 1 - eps/dt
            %   decreases the smallest eigenvalue while M2 off-diagonals (∝eps^2)
            %   increase the largest.

            ops = prob.ops;
            zeros_x_f = zeros(nt, ny);
            zeros_y_f = zeros(nt, nx);

            rho_phi_in = ops.interp_t_at_phi(x_in.rho, prob.rho0, prob.rho1);
            nabla_in   = ops.deriv_x_at_phi(ops.deriv_x_at_m(rho_phi_in), zeros_x_f, zeros_x_f) ...
                       + ops.deriv_y_at_phi(ops.deriv_y_at_m(rho_phi_in), zeros_y_f, zeros_y_f);
            f_in = ops.deriv_t_at_phi(x_in.rho, prob.rho0, prob.rho1) ...
                 + ops.deriv_x_at_phi(x_in.mx,  zeros_x_f, zeros_x_f) ...
                 + ops.deriv_y_at_phi(x_in.my,  zeros_y_f, zeros_y_f) ...
                 - vareps * nabla_in;
            norm_f = norm(f_in(:));

            % Maximum condition number over all non-DC (kx,ky) modes.
            % T is symmetric so eig returns real eigenvalues; cond = max|ev|/min|ev|.
            max_cond = 0;
            for kx = 1:nx
                for ky = 1:ny
                    if kx == 1 && ky == 1, continue; end
                    a_k = bp.lower_all(:, kx, ky);
                    b_k = bp.main_all(:,  kx, ky);
                    c_k = bp.upper_all(:, kx, ky);
                    T_k = diag(b_k) + diag(c_k, 1) + diag(a_k, -1);
                    ev  = real(eig(T_k));
                    max_cond = max(max_cond, max(abs(ev)) / min(abs(ev)));
                end
            end

            predicted  = eps(1.0) * max_cond * norm_f;
            ratio_pred = fp_max / predicted;

            fprintf('%-4d  %-4d  %-10.2e  %-8.2f  %-12.3e  %-12.3e  %-10.3f  %-12.3e  %-12.3e  %-12.3e  %-10.2f  %-12.3e  %-12.3e  %-12.3e  %-10.3f\n', ...
                nt, nxy, vareps, eps_dt, min_S, min_Tnn, ratio, fp_max, fp_tT, fp_int, grow_ratio, ...
                norm_f, max_cond, predicted, ratio_pred);
        end
        fprintf('\n');
    end
end

% =========================================================================
%% Local helpers
% =========================================================================

function res = fp_residual(x, prob, vareps)
    ops     = prob.ops;
    rho0    = prob.rho0;
    rho1    = prob.rho1;
    nt      = prob.nt;
    nx      = prob.nx;
    ny      = prob.ny;
    zeros_x = zeros(nt, ny);
    zeros_y = zeros(nt, nx);

    rho_phi   = ops.interp_t_at_phi(x.rho, rho0, rho1);
    nabla_rho = ops.deriv_x_at_phi(ops.deriv_x_at_m(rho_phi),  zeros_x, zeros_x) ...
              + ops.deriv_y_at_phi(ops.deriv_y_at_m(rho_phi),  zeros_y, zeros_y);
    res = ops.deriv_t_at_phi(x.rho, rho0, rho1) ...
        + ops.deriv_x_at_phi(x.mx,  zeros_x, zeros_x) ...
        + ops.deriv_y_at_phi(x.my,  zeros_y, zeros_y) ...
        - vareps .* nabla_rho;
end
