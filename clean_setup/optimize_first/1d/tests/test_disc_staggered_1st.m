%% test_disc_staggered_1st.m
%
% Tests for the staggered-grid operators in disc_staggered_1st.m.
%
% Three check types per operator:
%   1. Adjoint   -- <Ax, y> == <x, A'y>
%   2. Constant  -- derivative of constant = 0, interpolation of constant = constant
%   3. Linear    -- derivative of linear function = expected slope

clear; clc;

% Add paths (run from tests/ or from 1d/)
base = fileparts(mfilename('fullpath'));
run(fullfile(base, '..', 'setup_paths.m'));

%% Setup minimal problem
problem.nt = 4;   ntm = problem.nt - 1;
problem.nx = 4;   nxm = problem.nx - 1;
problem.dt = 1 / problem.nt;
problem.dx = 1 / problem.nx;

ops = disc_staggered_1st(problem);

dt = problem.dt;
dx = problem.dx;
nt = problem.nt;
nx = problem.nx;

pass = 0; fail = 0;

% -------------------------------------------------------------------------
%% 1. ADJOINT TESTS
% -------------------------------------------------------------------------
% For each forward operator A: (domain -> range)
% check  <A*x, y>  ==  <x, A'*y>
% where A' is identified as its adjoint counterpart.
%
% Adjoint pairs:
%   interp_t_at_phi  <-->  interp_t_at_rho
%   interp_x_at_phi  <-->  interp_x_at_m
%   deriv_t_at_phi   <-->  -deriv_t_at_rho  (skew-adjoint, up to sign & BCs)
%   deriv_x_at_phi   <-->  -deriv_x_at_m

tol = 1e-10;
rng(42);

% -- interp_t_at_phi / interp_t_at_rho --
x = randn(ntm, nx);
y = randn(nt,  nx);
bc0 = zeros(1, nx);  bc1 = zeros(1, nx);
Ax  = ops.interp_t_at_phi(x, bc0, bc1);
Aty = ops.interp_t_at_rho(y);
err = abs(Ax(:)'*y(:) - x(:)'*Aty(:));
[pass, fail] = report('interp_t adjoint', err, tol, pass, fail);

% -- interp_x_at_phi / interp_x_at_m --
x = randn(nt, nxm);
y = randn(nt, nx);
bc0 = zeros(nt, 1);  bc1 = zeros(nt, 1);
Ax  = ops.interp_x_at_phi(x, bc0, bc1);
Aty = ops.interp_x_at_m(y);
err = abs(Ax(:)'*y(:) - x(:)'*Aty(:));
[pass, fail] = report('interp_x adjoint', err, tol, pass, fail);

% -- deriv_t_at_phi / deriv_t_at_rho (zero BCs, skew-adjoint: <Ax,y> = -<x,A'y>) --
x = randn(ntm, nx);
y = randn(nt,  nx);
bc0 = zeros(1, nx);  bc1 = zeros(1, nx);
Ax  = ops.deriv_t_at_phi(x, bc0, bc1);
Aty = ops.deriv_t_at_rho(y);
err = abs(Ax(:)'*y(:) + x(:)'*Aty(:));   % skew: should sum to 0
[pass, fail] = report('deriv_t skew-adjoint', err, tol, pass, fail);

% -- deriv_x_at_phi / deriv_x_at_m (zero BCs, skew-adjoint) --
x = randn(nt, nxm);
y = randn(nt, nx);
bc0 = zeros(nt, 1);  bc1 = zeros(nt, 1);
Ax  = ops.deriv_x_at_phi(x, bc0, bc1);
Aty = ops.deriv_x_at_m(y);
err = abs(Ax(:)'*y(:) + x(:)'*Aty(:));   % skew: should sum to 0
[pass, fail] = report('deriv_x skew-adjoint', err, tol, pass, fail);

% -------------------------------------------------------------------------
%% 2. CONSTANT FUNCTION TESTS
% -------------------------------------------------------------------------
% Derivative of a constant should be zero (ignoring boundary terms).
% Interpolation of a constant should remain constant.

tol_const = 1e-10;

% -- interp_t_at_phi: constant input, zero BCs -> constant output --
c = 3.7;
x = c * ones(ntm, nx);
bc0 = c * ones(1, nx);   bc1 = c * ones(1, nx);
out = ops.interp_t_at_phi(x, bc0, bc1);
err = max(abs(out(:) - c));
[pass, fail] = report('interp_t_at_phi constant', err, tol_const, pass, fail);

% -- interp_t_at_rho: constant input -> constant output --
x = c * ones(nt, nx);
out = ops.interp_t_at_rho(x);
err = max(abs(out(:) - c));
[pass, fail] = report('interp_t_at_rho constant', err, tol_const, pass, fail);

% -- interp_x_at_phi: constant input, zero BCs -> constant output --
x = c * ones(nt, nxm);
bc0 = c * ones(nt, 1);   bc1 = c * ones(nt, 1);
out = ops.interp_x_at_phi(x, bc0, bc1);
err = max(abs(out(:) - c));
[pass, fail] = report('interp_x_at_phi constant', err, tol_const, pass, fail);

% -- interp_x_at_m: constant input -> constant output --
x = c * ones(nt, nx);
out = ops.interp_x_at_m(x);
err = max(abs(out(:) - c));
[pass, fail] = report('interp_x_at_m constant', err, tol_const, pass, fail);

% -------------------------------------------------------------------------
%% 3. LINEAR FUNCTION TESTS
% -------------------------------------------------------------------------
% Derivative of a linear function f(t) = t should equal 1/dt * dt = 1.
% (i.e., d/dt [t] = 1)

tol_lin = 1e-10;

t_rho = (1:ntm) * dt;          % time locations of rho: midpoints
t_phi = ((1:nt)'-0.5) * dt;    % time locations of phi: full grid

x_space = linspace(dx/2, 1-dx/2, nx);   % space cell centers

% -- deriv_t_at_phi: f(t,x) = t at rho-locations, d/dt = 1 everywhere --
x   = repmat(t_rho', 1, nx);      % (ntm x nx)
bc0 = zeros(1, nx);               % f at t=0
bc1 = ones(1,  nx);               % f at t=1
out = ops.deriv_t_at_phi(x, bc0, bc1);
err = max(abs(out(:) - 1.0));
[pass, fail] = report('deriv_t_at_phi linear', err, tol_lin, pass, fail);

% -- deriv_t_at_rho: f(t,x) = t at phi-locations, d/dt = 1 everywhere --
x   = repmat(t_phi, 1, nx);       % (nt x nx)
out = ops.deriv_t_at_rho(x);
err = max(abs(out(:) - 1.0));
[pass, fail] = report('deriv_t_at_rho linear', err, tol_lin, pass, fail);

% -- deriv_x_at_phi: f(t,x) = x at m-locations, d/dx = 1 everywhere --
x_m = ((1:nxm)) * dx;      % space locations of mx
x   = repmat(x_m, nt, 1);         % (nt x nxm)
bc0 = zeros(nt, 1);               % f at x=0
bc1 = ones(nt,  1);               % f at x=1
out = ops.deriv_x_at_phi(x, bc0, bc1);
err = max(abs(out(:) - 1.0));
[pass, fail] = report('deriv_x_at_phi linear', err, tol_lin, pass, fail);

% -- deriv_x_at_m: f(t,x) = x at phi-locations, d/dx = 1 everywhere --
x   = repmat(x_space, nt, 1);     % (nt x nx)
out = ops.deriv_x_at_m(x);
err = max(abs(out(:) - 1.0));
[pass, fail] = report('deriv_x_at_m linear', err, tol_lin, pass, fail);

% -------------------------------------------------------------------------
%% Summary
% -------------------------------------------------------------------------
fprintf('\n--- Results: %d passed, %d failed ---\n', pass, fail);

% -------------------------------------------------------------------------
%% Helper
% -------------------------------------------------------------------------
function [pass, fail] = report(name, err, tol, pass, fail)
    if err < tol
        fprintf('  PASS  %-35s  err = %.2e\n', name, err);
        pass = pass + 1;
    else
        fprintf('  FAIL  %-35s  err = %.2e  (tol=%.2e)\n', name, err, tol);
        fail = fail + 1;
    end
end
