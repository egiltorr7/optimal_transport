% TEST_SPECTRAL_OPS  Verify spectral derivative operators from disc_spectral_1d.
%
%   Tests deriv_x and deriv_xx against exact derivatives of known periodic
%   functions.  For a smooth periodic function the FFT-based derivative is
%   exact to machine precision (no truncation error for band-limited inputs).

clear;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% Setup
cfg      = cfg_admm_spectral_fe();
cfg.nx   = 16;   % need nx >> max k_idx used in tests (k=3 for cos(6pi*x))
prob_def = prob_gaussian();
problem  = setup_problem_spectral(cfg, prob_def);   % only need the grid and ops

ops = problem.ops;
xx  = problem.xx;   % (1 x nx) node points  x_j = (j-1)/nx
nx  = problem.nx;
dx  = problem.dx;

tol_exact = 1e-10;   % spectral should be near machine precision for smooth inputs
tol_approx = 1e-3;   % for non-smooth inputs (Gaussian tails)

fprintf('=== test_spectral_ops ===\n\n');

%% --- Test 1: d/dx of sin(2*pi*x) = 2*pi*cos(2*pi*x) ---
u   = sin(2*pi*xx);           % (1 x nx)  -- exactly band-limited
du  = ops.deriv_x(u);
ref = 2*pi*cos(2*pi*xx);
err = max(abs(du - ref));
report('deriv_x( sin(2pi*x) )', err, tol_exact);



%% --- Test 2: d/dx of cos(4*pi*x) = -4*pi*sin(4*pi*x) ---
u   = cos(4*pi*xx);
du  = ops.deriv_x(u);
ref = -4*pi*sin(4*pi*xx);
err = max(abs(du - ref));
report('deriv_x( cos(4pi*x) )', err, tol_exact);

%% --- Test 3: d^2/dx^2 of sin(2*pi*x) = -(2*pi)^2 * sin(2*pi*x) ---
u    = sin(2*pi*xx);
ddu  = ops.deriv_xx(u);
ref  = -(2*pi)^2 * sin(2*pi*xx);
err  = max(abs(ddu - ref));
report('deriv_xx( sin(2pi*x) )', err, tol_exact);

%% --- Test 4: d^2/dx^2 of cos(4*pi*x) = -(4*pi)^2 * cos(4*pi*x) ---
u    = cos(4*pi*xx);
ddu  = ops.deriv_xx(u);
ref  = -(4*pi)^2 * cos(4*pi*xx);
err  = max(abs(ddu - ref));
report('deriv_xx( cos(4pi*x) )', err, tol_exact);

%% --- Test 5: deriv_x applied to a (nt x nx) matrix (all rows same) ---
nt  = problem.nt;
U   = repmat(sin(2*pi*xx), nt, 1);   % (nt x nx)
dU  = ops.deriv_x(U);
ref = repmat(2*pi*cos(2*pi*xx), nt, 1);
err = max(abs(dU(:) - ref(:)));
report('deriv_x on matrix (nt x nx)', err, tol_exact);

%% --- Test 6: linearity  d(a*f + b*g)/dx = a*df/dx + b*dg/dx ---
a = 3.7;  b = -1.2;
f = sin(2*pi*xx);  g = cos(6*pi*xx);
lhs = ops.deriv_x(a*f + b*g);
rhs = a*ops.deriv_x(f) + b*ops.deriv_x(g);
err = max(abs(lhs - rhs));
report('linearity of deriv_x', err, tol_exact);

%% --- Test 7: consistency  d^2/dx^2 = d/dx composed with d/dx ---
u    = sin(2*pi*xx) + 0.5*cos(4*pi*xx);
ddu1 = ops.deriv_xx(u);
ddu2 = ops.deriv_x(ops.deriv_x(u));
err  = max(abs(ddu1 - ddu2));
report('deriv_xx = deriv_x o deriv_x', err, tol_exact);

%% --- Test 8: wavenumber sign convention (imaginary part should vanish) ---
u_real = cos(2*pi*xx);
du     = ops.deriv_x(u_real);
err    = max(abs(imag(du)));
report('deriv_x output is real', err, tol_exact);

fprintf('\nDone.\n');

%% -----------------------------------------------------------------------
function report(name, err, tol)
    if err < tol
        fprintf('  PASS  %-45s  err = %.2e\n', name, err);
    else
        fprintf('  FAIL  %-45s  err = %.2e  (tol = %.2e)\n', name, err, tol);
    end
end
