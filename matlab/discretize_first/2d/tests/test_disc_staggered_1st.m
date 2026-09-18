%% test_disc_staggered_1st.m
%
% Tests for the 2D staggered-grid operators in shared/2d/discretization/disc_staggered_1st.m.
%
% Four check types:
%   1. Shape    -- each operator outputs the expected array size
%   2. Adjoint  -- <Ax, y> == <x, A'y> with zero BCs (so affine part vanishes)
%   3. Constant -- derivative of constant = 0, interpolation of constant = constant
%   4. Linear   -- derivative of linear function f(t)=t, f(x)=x, f(y)=y equals 1

clear; clc;

% Add paths directly (setup_paths.m not yet complete)
base    = fileparts(mfilename('fullpath'));
sh_base = fullfile(base, '..', '..', '..', 'shared');
addpath(fullfile(sh_base, '2d', 'discretization'));

%% --- Minimal problem struct (odd sizes to catch transposition bugs) ---
problem.nt = 5;   ntm = problem.nt - 1;
problem.nx = 6;   nxm = problem.nx - 1;
problem.ny = 4;   nym = problem.ny - 1;
problem.dt = 1 / problem.nt;
problem.dx = 1 / problem.nx;
problem.dy = 1 / problem.ny;

ops = disc_staggered_1st(problem);

dt = problem.dt;  dx = problem.dx;  dy = problem.dy;
nt = problem.nt;  nx = problem.nx;  ny = problem.ny;

pass = 0; fail = 0;
tol  = 1e-10;
rng(42);

% BCs are zero arrays of the shape each operator expects:
%   time operators : (nx x ny)  -- boundary densities at t=0,1
%   x operators    : (nt x ny)  -- momentum at x=0,1
%   y operators    : (nt x nx)  -- momentum at y=0,1
bc_t0 = zeros(nx, ny);
bc_x0 = zeros(nt, ny);
bc_y0 = zeros(nt, nx);

% -------------------------------------------------------------------------
%% 1. SHAPE TESTS
% -------------------------------------------------------------------------
% Every operator should map to the expected array size.

rho = randn(ntm, nx,  ny );   % density (staggered)
mx  = randn(nt,  nxm, ny );   % x-momentum (staggered)
my  = randn(nt,  nx,  nym);   % y-momentum (staggered)
phi = randn(nt,  nx,  ny );   % cell-centre / phi grid

[pass,fail] = check_size('interp_t_at_phi ', ops.interp_t_at_phi(rho, bc_t0, bc_t0), [nt,  nx,  ny ], pass, fail);
[pass,fail] = check_size('interp_t_at_rho ', ops.interp_t_at_rho(phi),               [ntm, nx,  ny ], pass, fail);
[pass,fail] = check_size('interp_x_at_phi ', ops.interp_x_at_phi(mx, bc_x0, bc_x0),  [nt,  nx,  ny ], pass, fail);
[pass,fail] = check_size('interp_x_at_m   ', ops.interp_x_at_m(phi),                 [nt,  nxm, ny ], pass, fail);
[pass,fail] = check_size('interp_y_at_phi ', ops.interp_y_at_phi(my, bc_y0, bc_y0),  [nt,  nx,  ny ], pass, fail);
[pass,fail] = check_size('interp_y_at_m   ', ops.interp_y_at_m(phi),                 [nt,  nx,  nym], pass, fail);
[pass,fail] = check_size('deriv_t_at_phi  ', ops.deriv_t_at_phi(rho, bc_t0, bc_t0),  [nt,  nx,  ny ], pass, fail);
[pass,fail] = check_size('deriv_t_at_rho  ', ops.deriv_t_at_rho(phi),                [ntm, nx,  ny ], pass, fail);
[pass,fail] = check_size('deriv_x_at_phi  ', ops.deriv_x_at_phi(mx, bc_x0, bc_x0),   [nt,  nx,  ny ], pass, fail);
[pass,fail] = check_size('deriv_x_at_m    ', ops.deriv_x_at_m(phi),                  [nt,  nxm, ny ], pass, fail);
[pass,fail] = check_size('deriv_y_at_phi  ', ops.deriv_y_at_phi(my, bc_y0, bc_y0),   [nt,  nx,  ny ], pass, fail);
[pass,fail] = check_size('deriv_y_at_m    ', ops.deriv_y_at_m(phi),                  [nt,  nx,  nym], pass, fail);

% -------------------------------------------------------------------------
%% 2. NOTE ON ADJOINTS
% -------------------------------------------------------------------------
% interp_t_at_rho is NOT the exact adjoint of interp_t_at_phi(x, 0, 0).
% With zero BCs, interp_t_at_phi zeros the first and last output rows, so
% y(1) and y(nt) never appear in <Ax,y>. But interp_t_at_rho(y) includes
% 0.5*y(1) and 0.5*y(nt) at the boundary entries of its output, causing a
% mismatch of 0.5*(x(1)*y(1) + x(ntm)*y(nt)).
%
% The same applies to the derivative pairs. These operators are the adjoint
% of the linear part without BC correction -- an approximation used in the
% ADMM linearised gradient step, not an exact adjoint.
%
% Correctness is instead verified by the constant and linear tests below.

% -------------------------------------------------------------------------
%% 3. CONSTANT FUNCTION TESTS
% -------------------------------------------------------------------------
% Interpolation of a constant field = constant everywhere.
% Derivative of a constant field = 0.

c = 3.7;

% interp_t_at_phi: constant interior + matching BCs -> constant output
x = c * ones(ntm, nx, ny);
bc = c * ones(nx, ny);
out = ops.interp_t_at_phi(x, bc, bc);
[pass,fail] = report('interp_t_at_phi const ', max(abs(out(:)-c)), tol, pass, fail);

% interp_t_at_rho: constant -> constant
out = ops.interp_t_at_rho(c * ones(nt, nx, ny));
[pass,fail] = report('interp_t_at_rho const ', max(abs(out(:)-c)), tol, pass, fail);

% interp_x_at_phi: constant interior + matching BCs -> constant output
x = c * ones(nt, nxm, ny);
bc = c * ones(nt, ny);
out = ops.interp_x_at_phi(x, bc, bc);
[pass,fail] = report('interp_x_at_phi const ', max(abs(out(:)-c)), tol, pass, fail);

% interp_x_at_m: constant -> constant
out = ops.interp_x_at_m(c * ones(nt, nx, ny));
[pass,fail] = report('interp_x_at_m const   ', max(abs(out(:)-c)), tol, pass, fail);

% interp_y_at_phi: constant interior + matching BCs -> constant output
x = c * ones(nt, nx, nym);
bc = c * ones(nt, nx);
out = ops.interp_y_at_phi(x, bc, bc);
[pass,fail] = report('interp_y_at_phi const ', max(abs(out(:)-c)), tol, pass, fail);

% interp_y_at_m: constant -> constant
out = ops.interp_y_at_m(c * ones(nt, nx, ny));
[pass,fail] = report('interp_y_at_m const   ', max(abs(out(:)-c)), tol, pass, fail);

% -------------------------------------------------------------------------
%% 4. LINEAR FUNCTION TESTS
% -------------------------------------------------------------------------
% d/dt [t] = 1, d/dx [x] = 1, d/dy [y] = 1

% Grid coordinates
t_rho = (1:ntm)' * dt;             % time at rho-locations  (ntm x 1 x 1)
t_phi = ((1:nt)' - 0.5) * dt;         % time at phi-locations  (nt  x 1 x 1)  [interior faces: k*dt]
x_m   = (1:nxm)' * dx;         % x at mx-locations      (nxm x 1)
x_phi = ((1:nx)'  - 0.5)  * dx;        % x at phi-locations     (nx  x 1)
y_m   = (1:nym)            * dy;        % y at my-locations      (1 x nym)
y_phi = ((1:ny)   - 0.5)  * dy;        % y at phi-locations     (1 x ny)

% -- deriv_t_at_phi: f(t,x,y)=t at rho-locs, d/dt=1 --
% rho-grid: t goes 0.5*dt, 1.5*dt, ..., (ntm-0.5)*dt
% BCs: f(t=0,.) = 0,  f(t=1,.) = 1
x = repmat(t_rho, 1, nx, ny);
bc0 = zeros(nx, ny);   bc1 = ones(nx, ny);
out = ops.deriv_t_at_phi(x, bc0, bc1);
[pass,fail] = report('deriv_t_at_phi linear ', max(abs(out(:)-1)), tol, pass, fail);

% -- deriv_t_at_rho: f(t,x,y)=t at phi-locs, d/dt=1 --
x = repmat(t_phi, 1, nx, ny);
out = ops.deriv_t_at_rho(x);
[pass,fail] = report('deriv_t_at_rho linear ', max(abs(out(:)-1)), tol, pass, fail);

% -- deriv_x_at_phi: f(t,x,y)=x at mx-locs, d/dx=1 --
x = repmat(reshape(x_m, 1, nxm, 1), nt, 1, ny);
bc0 = zeros(nt, ny);   bc1 = ones(nt, ny);
out = ops.deriv_x_at_phi(x, bc0, bc1);
[pass,fail] = report('deriv_x_at_phi linear ', max(abs(out(:)-1)), tol, pass, fail);

% -- deriv_x_at_m: f(t,x,y)=x at phi-locs, d/dx=1 --
x = repmat(reshape(x_phi, 1, nx, 1), nt, 1, ny);
out = ops.deriv_x_at_m(x);
[pass,fail] = report('deriv_x_at_m linear   ', max(abs(out(:)-1)), tol, pass, fail);

% -- deriv_y_at_phi: f(t,x,y)=y at my-locs, d/dy=1 --
x = repmat(reshape(y_m, 1, 1, nym), nt, nx, 1);
bc0 = zeros(nt, nx);   bc1 = ones(nt, nx);
out = ops.deriv_y_at_phi(x, bc0, bc1);
[pass,fail] = report('deriv_y_at_phi linear ', max(abs(out(:)-1)), tol, pass, fail);

% -- deriv_y_at_m: f(t,x,y)=y at phi-locs, d/dy=1 --
x = repmat(reshape(y_phi, 1, 1, ny), nt, nx, 1);
out = ops.deriv_y_at_m(x);
[pass,fail] = report('deriv_y_at_m linear   ', max(abs(out(:)-1)), tol, pass, fail);

% -------------------------------------------------------------------------
%% Summary
% -------------------------------------------------------------------------
fprintf('\n--- Results: %d passed, %d failed ---\n', pass, fail);

% -------------------------------------------------------------------------
%% Helpers
% -------------------------------------------------------------------------
function [pass, fail] = report(name, err, tol, pass, fail)
    if err < tol
        fprintf('  PASS  %-30s  err = %.2e\n', name, err);
        pass = pass + 1;
    else
        fprintf('  FAIL  %-30s  err = %.2e  (tol=%.2e)\n', name, err, tol);
        fail = fail + 1;
    end
end

function [pass, fail] = check_size(name, out, expected, pass, fail)
    actual = size(out);
    % Pad to same length for comparison
    n = max(numel(actual), numel(expected));
    a = [actual,   ones(1, n-numel(actual))];
    e = [expected, ones(1, n-numel(expected))];
    if isequal(a, e)
        fprintf('  PASS  size %-26s  got [%s]\n', name, num2str(actual));
        pass = pass + 1;
    else
        fprintf('  FAIL  size %-26s  expected [%s], got [%s]\n', ...
            name, num2str(expected), num2str(actual));
        fail = fail + 1;
    end
end
