%% test_projection.m
% Tests the projection step (proj_div_banded) inside sb1d_admm.
%
% Key question: is the T_k matrix (precomputed in sb1d_admm) correct?
% We compare the current code's T_k against the reference formula from
% the Python implementation (solve_phi in sb_admm_1d.py).

clear

nt = 16; nx = 16;
dt = 1/nt; dx = 1/nx;
ntm = nt-1; nxm = nx-1;
vareps = 1;

xx = ((1:nx) - 0.5) * dx;
Normal = @(x,mu,sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);
rho0 = Normal(xx, 1/3, 0.05); rho0 = rho0/sum(rho0);
rho1 = Normal(xx, 2/3, 0.05); rho1 = rho1/sum(rho1);

%% Build operators
It_phi = 0.5 * toeplitz([1 1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]);   % nt x ntm
It_rho = 0.5 * toeplitz([1 zeros(1,ntm-1)], [1 1 zeros(1,ntm-1)]);   % ntm x nt
Dt_phi = toeplitz([1 -1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]) / dt;   % nt x ntm
Dt_rho = toeplitz([-1 zeros(1,ntm-1)], [-1 1 zeros(1,ntm-1)]) / dt;  % ntm x nt
Dx_phi = toeplitz([1 -1 zeros(1,nxm-1)], [1 zeros(1,nxm-1)]) / dx;   % nx x nxm
Dx_m   = toeplitz([-1 zeros(1,nxm-1)], [-1 1 zeros(1,nxm-1)]) / dx;  % nxm x nx

%% ── Test A: Key algebraic identity ─────────────────────────────────────────
% Claim: Dt_rho = -Dt_phi'  (this determines the sign of M0_bnd)
fprintf('=== Test A: Dt_rho == -Dt_phi'' ? ===\n');
fprintf('  ||Dt_rho - (-Dt_phi'')||_F = %.2e  (should be 0)\n\n', ...
    norm(Dt_rho - (-Dt_phi'), 'fro'));

%% ── Test B: M0_bnd sign ─────────────────────────────────────────────────────
% Current code: M0_bnd = Dt_phi * Dt_rho  = -(Dt_phi * Dt_phi')
% Correct for T_k (from Python):  -(Dt_phi * Dt_rho) = Dt_phi * Dt_phi'
M0_code    = Dt_phi * Dt_rho;     % what the code uses
M0_correct = Dt_phi * Dt_phi';    % = -M0_code

fprintf('=== Test B: M0_bnd sign ===\n');
fprintf('  M0_code    interior diagonal = %+.4f  (= -2/dt^2 = %+.4f)\n', M0_code(2,2),    -2/dt^2);
fprintf('  M0_correct interior diagonal = %+.4f  (= +2/dt^2 = %+.4f)\n', M0_correct(2,2), +2/dt^2);
fprintf('  --> Code has WRONG sign; should use -M0_bnd in Tk.\n\n');

%% ── Test C: T_k eigenvalues ─────────────────────────────────────────────────
lambda_x = (2 - 2*cos(pi*dx*(0:nxm))) / dx^2;
M2_bnd   = vareps^2 * It_phi * It_rho;
M1d      = ones(nt, 1);
M1d(1)   = 1 + vareps/dt;
M1d(nt)  = 1 - vareps/dt;

fprintf('=== Test C: T_k min eigenvalue (k=2 and k=nx) ===\n');
for kk = [2, nx]
    lxk = lambda_x(kk);
    Tk_code = full( M0_code    + diag(lxk * M1d) + lxk^2 * M2_bnd);
    Tk_fix  = full(-M0_code    + diag(lxk * M1d) + lxk^2 * M2_bnd);
    ev_code = min(eig(Tk_code));
    ev_fix  = min(eig(Tk_fix));
    fprintf('  k=%2d (lxk=%.1f): code min_eig=%+.4f, fixed min_eig=%+.4f\n', ...
        kk, lxk, ev_code, ev_fix);
end
fprintf('  Projection requires T_k SPD (all min_eig > 0).\n\n');

%% ── Test D: T_k matches Python reference ────────────────────────────────────
% Python formula (from solve_phi docstring):
%   main_diag interior = 2/dt^2 + 0.5*eps^2*lk^2 + lk
%   off_diag            = -1/dt^2 + 0.25*eps^2*lk^2
%   corner first        = 1/dt^2 - eps*lk/dt + 0.25*eps^2*lk^2 + lk
%   corner last         = 1/dt^2 + eps*lk/dt + 0.25*eps^2*lk^2 + lk
fprintf('=== Test D: Fixed T_k vs Python reference formula ===\n');
for kk = [2, nx]
    lxk = lambda_x(kk);
    Tk_fix = -M0_code + diag(lxk * M1d) + lxk^2 * M2_bnd;

    main_d = (2/dt^2 + 0.5*vareps^2*lxk^2 + lxk) * ones(nt,1);
    main_d(1)   = 1/dt^2 - vareps*lxk/dt + 0.25*vareps^2*lxk^2 + lxk;
    main_d(end) = 1/dt^2 + vareps*lxk/dt + 0.25*vareps^2*lxk^2 + lxk;
    off_d  = -1/dt^2 + 0.25*vareps^2*lxk^2;
    Tk_py  = diag(main_d) + diag(off_d*ones(nt-1,1),1) + diag(off_d*ones(nt-1,1),-1);

    fprintf('  k=%2d: ||Tk_fix - Tk_python||_F = %.2e\n', kk, norm(Tk_fix - Tk_py,'fro'));
end
fprintf('  (should be ~0 if fix is correct)\n\n');

%% ── Test E: Projection residual ─────────────────────────────────────────────
% Build T_k LU factors for both versions, run one projection, check fp_res.
tt   = (1:ntm)'/nt;
rho_in = (1-tt).*rho0 + tt.*rho1;   % ntm x nx
mx_in  = zeros(nt, nxm);

% helper functions
itp = @(r) It_phi*r + [0.5*rho0; zeros(ntm-1,nx); 0.5*rho1];
itr = @(p) It_rho*p;
dtp = @(r) Dt_phi*r + [-rho0/dt; zeros(ntm-1,nx); rho1/dt];
dtr = @(p) Dt_rho*p;
dxp = @(m) m*Dx_phi';
dxm = @(p) p*Dx_m';
fp  = @(r,m) dtp(r) + dxp(m) - vareps*dxp(dxm(itp(r)));

lambda_t = (2 - 2*cos(pi*dt*(0:ntm)')) / dt^2;  % nt x 1



fprintf('=== Test E: fp_res after one projection ===\n');
% Variant 1: code (wrong M0, -vareps update)
[rh1, mh1] = run_proj(rho_in, mx_in, rho0, rho1, false, ...
    M0_code, M1d, M2_bnd, lambda_x, lambda_t, ...
    It_phi, It_rho, Dt_phi, Dt_rho, Dx_phi, Dx_m, vareps, ...
    nt, nx, ntm, nxm, dt, dx, -1);
f1 = fp(rh1, mh1);
fprintf('  Original code (+M0, -vareps update): ||fp_res||_inf = %.4e\n', max(abs(f1(:))));
if any(isnan(f1(:))), fprintf('  --> NaN detected!\n'); end

% Variant 2: fixed M0 only, -vareps update
[rh2, mh2] = run_proj(rho_in, mx_in, rho0, rho1, true, ...
    M0_code, M1d, M2_bnd, lambda_x, lambda_t, ...
    It_phi, It_rho, Dt_phi, Dt_rho, Dx_phi, Dx_m, vareps, ...
    nt, nx, ntm, nxm, dt, dx, -1);
f2 = fp(rh2, mh2);
fprintf('  Fixed M0 (-M0), -vareps update:      ||fp_res||_inf = %.4e\n', max(abs(f2(:))));

% Variant 3: fixed M0 + corrected update sign (+vareps)
[rh3, mh3] = run_proj(rho_in, mx_in, rho0, rho1, true, ...
    M0_code, M1d, M2_bnd, lambda_x, lambda_t, ...
    It_phi, It_rho, Dt_phi, Dt_rho, Dx_phi, Dx_m, vareps, ...
    nt, nx, ntm, nxm, dt, dx, +1);
f3 = fp(rh3, mh3);
fprintf('  Fixed M0 (-M0), +vareps update:      ||fp_res||_inf = %.4e\n', max(abs(f3(:))));
fprintf('  (best result = smallest fp_res)\n\n');

function [rh, mh] = run_proj(rho_in, mx_in, rho0, rho1, use_correct_M0, ...
        M0_code, M1d, M2_bnd, lambda_x, lambda_t, ...
        It_phi, It_rho, Dt_phi, Dt_rho, Dx_phi, Dx_m, vareps, ...
        nt, nx, ntm, nxm, dt, dx, sign_update)
    itp2 = @(r) It_phi*r + [0.5*rho0; zeros(ntm-1,nx); 0.5*rho1];
    itr2 = @(p) It_rho*p;
    dtp2 = @(r) Dt_phi*r + [-rho0/dt; zeros(ntm-1,nx); rho1/dt];
    dtr2 = @(p) Dt_rho*p;
    dxp2 = @(m) m*Dx_phi';
    dxm2 = @(p) p*Dx_m';
    fp2  = @(r,m) dtp2(r) + dxp2(m) - vareps*dxp2(dxm2(itp2(r)));

    f = fp2(rho_in, mx_in);
    f_hat_x = dct(f')';
    phi_hat_x = zeros(nt, nx);
    f1_hat_t = dct(f_hat_x(:,1));
    phi1_hat_t = zeros(nt,1);
    phi1_hat_t(2:end) = f1_hat_t(2:end) ./ lambda_t(2:end);
    phi_hat_x(:,1) = idct(phi1_hat_t);
    for kk = 2:nx
        lxk = lambda_x(kk);
        if use_correct_M0
            Tk = sparse(-M0_code + diag(lxk*M1d) + lxk^2*M2_bnd);
        else
            Tk = sparse( M0_code + diag(lxk*M1d) + lxk^2*M2_bnd);
        end
        [L,U,P] = lu(Tk);
        phi_hat_x(:,kk) = U \ (L \ (P * f_hat_x(:,kk)));
    end
    phi = idct(phi_hat_x')';
    dphi = dxm2(phi);
    nph  = itr2(dxp2(dphi));
    rh = rho_in + dtr2(phi) + sign_update * vareps * nph;
    mh = mx_in  + dphi;
end
