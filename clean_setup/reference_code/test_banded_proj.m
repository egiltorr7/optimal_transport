%% test_banded_proj.m
%
%  Numerical verification that:
%  (1) T_k (code formula) matches the exact k-th DCT-x block of AA*.
%  (2) The full banded projection gives fp_res(rho_out, m_out) ~ 0.
%
%  Run from the claude/ directory.

clear; fprintf('=== Banded projection diagnostic ===\n\n');

nt  = 8;   ntm = nt-1;
nx  = 6;   nxm = nx-1;
dt  = 1/nt;   dx  = 1/nx;
vareps = 1e-2;

%% Operators (no BCs)
It_phi = 0.5 * toeplitz([1 1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]);  % nt x ntm
It_rho = 0.5 * toeplitz([1 zeros(1,ntm-1)], [1 1 zeros(1,ntm-1)]); % ntm x nt
Dt_phi = toeplitz([1 -1 zeros(1,ntm-1)], [1 zeros(1,ntm-1)]) / dt;  % nt x ntm
Dt_rho = toeplitz([-1 zeros(1,ntm-1)], [-1 1 zeros(1,ntm-1)]) / dt; % ntm x nt

Dx_phi = toeplitz([1 -1 zeros(1,nxm-1)], [1 zeros(1,nxm-1)]) / dx;  % nx x nxm
Dx_m   = toeplitz([-1 zeros(1,nxm-1)], [-1 1 zeros(1,nxm-1)]) / dx; % nxm x nx

lambda_x = (2 - 2*cos(pi*dx*(0:nxm))) / dx^2;   % 1 x nx
lambda_t = (2 - 2*cos(pi*dt*(0:ntm)')) / dt^2;  % nt x 1

%% ── Build the exact AA* matrix ────────────────────────────────────────────
%
%  State: phi (nt x nx) stacked column-major, size N_phi = nt*nx.
%  A = [A_rho | A_mx]:
%    A_rho: (rho, ntm x nx) -> phi, using Dt_phi in t + eps*Lap_x*It_phi
%    A_mx : (mx, nt x nxm) -> phi, using Div_x = kron(Dx_phi, I_t)
%
%  Lap_x on phi rows: L = Dx_phi * Dx_m  (nx x nx, Neumann Laplacian,
%                                          negative semi-definite)

Lap_x = Dx_phi * Dx_m;   % nx x nx, eigenvalues are -(lambda_x)

% A_rho = kron(I_x, Dt_phi) + eps * kron(Lap_x', I_t) * kron(I_x, It_phi)
%       = kron(I_x, Dt_phi) + eps * kron(Lap_x', It_phi)
% [Because Lap_x acts on rows of phi (each row = x-profile):
%  (phi * Lap_x)(vec) = kron(Lap_x', I_t) * phi_vec]
A_rho = kron(eye(nx), Dt_phi) + vareps * kron(Lap_x', It_phi);  % N_phi x N_rho

% A_mx = kron(Dx_phi', I_t)
% [Div_x m = m * Dx_phi': kron(Dx_phi', I_t) acts on mx_vec]
A_mx  = kron(Dx_phi', eye(nt));   % N_phi x N_mx

A_full = [A_rho, A_mx];   % N_phi x (N_rho + N_mx)

% Adjoint:  A_rho^T = kron(I, Dt_phi^T) + eps*kron(Lap_x, It_phi^T)
%  Dt_phi^T = -Dt_rho  (note sign),  Lap_x^T = Lap_x (symmetric)
%  It_phi^T = It_rho
A_rho_T = kron(eye(nx), Dt_phi') + vareps * kron(Lap_x, It_phi');  % N_rho x N_phi
% A_mx^T  = kron(Dx_phi, I_t)
A_mx_T  = kron(Dx_phi, eye(nt));   % N_mx x N_phi

A_star = [A_rho_T; A_mx_T];        % (N_rho+N_mx) x N_phi

AAt = A_full * A_star;   % N_phi x N_phi

fprintf('AA* symmetry: ||AA* - (AA*)^T|| / ||AA*|| = %.2e\n', ...
    norm(AAt - AAt','fro') / norm(AAt,'fro'));
fprintf('Min eigenvalue of AA*: %.4g  (should be > 0)\n\n', min(real(eig(AAt))));

%% ── DCT-x block decomposition of AA* ─────────────────────────────────────
%  In col-major with nt fast index:  phi_vec(k*nt + t) = phi(t, k+1).
%  DCT in x: V_x (nx x nx) orthonormal DCT-II matrix.
%  Block diagonalisation:  kron(V_x, I_t) * AAt * kron(V_x', I_t).

V_x = dct(eye(nx));           % orthonormal DCT-II, nx x nx
V_x_kron = kron(V_x, eye(nt));
AAt_dct = V_x_kron * AAt * V_x_kron';   % should be block-diagonal

%% ── Code formula for T_k ─────────────────────────────────────────────────
M0_bnd = -Dt_phi * Dt_rho;
M2_bnd = vareps^2 * It_phi * It_rho;
M1d    = ones(nt, 1);
M1d(1) = 1 - vareps/dt;
M1d(nt)= 1 + vareps/dt;

fprintf('── T_k accuracy (code vs exact DCT-x block of AA*) ──\n');
max_err = 0;
for kk = 1:nx
    lxk = lambda_x(kk);
    Tk_code  = M0_bnd + diag(lxk * M1d) + lxk^2 * M2_bnd;

    % Extract k-th diagonal block (MATLAB 1-indexed, nt fast)
    idx = (kk-1)*nt + (1:nt);
    Tk_exact = AAt_dct(idx, idx);

    err = norm(Tk_code - Tk_exact,'fro') / max(norm(Tk_exact,'fro'),1e-14);
    if err > max_err, max_err = err; end
    fprintf('  k=%2d  lx=%8.3g  err = %.2e\n', kk, lxk, err);
end
fprintf('Max relative error: %.2e\n\n', max_err);

% Off-diagonal blocks
off = AAt_dct;
for kk = 1:nx
    idx = (kk-1)*nt + (1:nt);
    off(idx,idx) = 0;
end
fprintf('Off-diagonal block norm in DCT-x space: %.2e  (should be ~0)\n\n', ...
    norm(off,'fro'));

%% ── Full projection accuracy ──────────────────────────────────────────────
%
%  Pick random (mu, psi) on the interior grids.
%  Project using EXACT inverse (AAt\f) and via the banded T_k solve.
%  Check fp_res of the output.

rng(1);
mu  = randn(ntm, nx);   % rho-grid
psi = randn(nt, nxm);   % mx-grid

% BCs for this test
rho0 = abs(randn(1, nx)); rho0 = rho0/sum(rho0);
rho1 = abs(randn(1, nx)); rho1 = rho1/sum(rho1);

% fp_res: full constraint residual on phi-grid (nt x nx)
% f = Dt_phi*rho + Div_x*m + eps*Lap_x*It_phi*rho  [all BCs included]
zeros_x = zeros(nt, 1);

function out = It_phi_fn(r, b0, b1, It)
    out = It * r;
    out(1,:)   = out(1,:)   + 0.5 * b0;
    out(end,:) = out(end,:) + 0.5 * b1;
end
function out = Dt_phi_fn(r, b0, b1, Dt, dt)
    out = Dt * r;
    out(1,:)   = out(1,:)   - b0 / dt;
    out(end,:) = out(end,:) + b1 / dt;
end

It_phi_bc  = @(r) It_phi_fn(r, rho0, rho1, It_phi);
Dt_phi_bc  = @(r) Dt_phi_fn(r, rho0, rho1, Dt_phi, dt);
Div_x_fn   = @(m) m * Dx_phi';    % nt x nxm -> nt x nx  (zero-flux)
Grad_x_fn  = @(p) p * Dx_m';      % nt x nx  -> nt x nxm
Lap_x_fn   = @(p) p * Lap_x';     % nt x nx  -> nt x nx  (Lap_x' = Lap_x)
It_rho_fn  = @(p) It_rho * p;     % phi-grid -> rho-grid
Dt_rho_fn  = @(p) Dt_rho * p;

fp_res_fn = @(r, m) ...
    Dt_phi_bc(r) + Div_x_fn(m) + vareps * Lap_x_fn(It_phi_bc(r));

f = fp_res_fn(mu, psi);
f_vec = reshape(f', [], 1);   % col-major (x fast) ... hmm, need to be careful

% Actually, to use AAt, we need to use the SAME column-major ordering as A_full.
% Our vec ordering: phi_vec(j) = phi(mod(j-1,nt)+1, floor((j-1)/nt)+1)
% i.e., nt is FAST index (time changes fastest).
% But f is (nt x nx), so f_vec = f(:) is nt-fast. ✓

f_vec = f(:);

% Exact solve
phi_vec_exact = AAt \ f_vec;

% Apply A* (exact)
correction = A_star * phi_vec_exact;
rho_corr = reshape(correction(1:ntm*nx), ntm, nx);
mx_corr  = reshape(correction(ntm*nx+1:end), nt, nxm);

% Note: projection uses MINUS A^T (since [rho_out; m_out] = [mu; psi] - A^T phi)
rho_out_exact = mu - rho_corr;
m_out_exact   = psi - mx_corr;

f_after_exact = fp_res_fn(rho_out_exact, m_out_exact);
viol_exact = norm(f_after_exact(:)) / max(norm(f_vec), 1e-14);
fprintf('Exact projection residual (||f_after|| / ||f_before||): %.2e  (should be ~0)\n', viol_exact);

% ── Banded T_k solve ───────────────────────────────────────────────────────
f_hat_x = dct(f')';          % physical-t x DCT-x

phi_hat_x = zeros(nt, nx);
% k=1 (DC in x): T_1 = M0_bnd (time Neumann Laplacian), invert via DCT-t
f1_hat_t = dct(f_hat_x(:, 1));
phi1_hat_t = zeros(nt, 1);
phi1_hat_t(2:end) = f1_hat_t(2:end) ./ lambda_t(2:end);
phi_hat_x(:, 1) = idct(phi1_hat_t);
for kk = 2:nx
    lxk = lambda_x(kk);
    Tk  = M0_bnd + diag(lxk * M1d) + lxk^2 * M2_bnd;
    phi_hat_x(:, kk) = Tk \ f_hat_x(:, kk);   % simple \ for clarity
end

phi_banded = idct(phi_hat_x')';   % physical space

% Apply A* (same as in the code)
dphi_dx    = phi_banded * Dx_m';          % phi -> mx-grid (Grad_x)
Lap_phi    = phi_banded * Lap_x';         % Lap_x on phi
nablax_phi = It_rho_fn(Lap_phi);          % interpolate to rho-grid

rho_out_banded = mu  + Dt_rho_fn(phi_banded) - vareps * nablax_phi;
m_out_banded   = psi + dphi_dx;

f_after_banded = fp_res_fn(rho_out_banded, m_out_banded);
viol_banded = norm(f_after_banded(:)) / max(norm(f_vec), 1e-14);
fprintf('Banded projection residual (||f_after|| / ||f_before||): %.2e  (should be ~0)\n\n', viol_banded);

% Compare phi
phi_diff = norm(phi_banded(:) - reshape(phi_vec_exact,nt,nx)(:)) / ...
           max(norm(phi_vec_exact), 1e-14);
fprintf('||phi_banded - phi_exact|| / ||phi_exact|| = %.2e\n\n', phi_diff);

%% ── A^T sign check ────────────────────────────────────────────────────────
% Verify  A_rho^T = -(kron(I,Dt_rho)) + eps*kron(Lap_x, It_rho)
% The code uses: rho_out = mu + Dt_rho*phi - eps*It_rho*Lap_x*phi
% = mu - (-Dt_rho + eps*It_rho*Lap_x)*phi = mu - A_rho^T phi
% Check: A_rho^T phi_test == -(kron(I,Dt_rho) + ...)

phi_test = randn(nt, nx);
% Code's A^T applied to phi_test (rho component):
r1 = Dt_rho_fn(phi_test) - vareps * It_rho_fn(Lap_x_fn(phi_test));  % from code
% Matrix A_rho_T applied to phi_test:
r2 = reshape(A_rho_T * phi_test(:), ntm, nx);

err_AT = norm(r1(:) - r2(:)) / max(norm(r2(:)), 1e-14);
fprintf('A_rho^T check (code formula vs matrix): %.2e  (should be ~0)\n', err_AT);
fprintf('(If nonzero, the A* update formula in both proj functions is wrong)\n\n');

fprintf('=== Done ===\n');
