%% test_prox_ke_cc.m
%
% Tests for discretize_first/2d/prox/prox_ke_cc.m.
%
% Tests:
%   1. Output shapes
%   2. KKT / optimality conditions (pointwise, exact formula)
%   3. Non-negativity  rho_out >= 0
%   4. my=0 in => my=0 out  (symmetry)
%   5. Consistency with 1D prox_ke_cc at each (t,x) slice when ny=1

clear; clc;

base    = fileparts(mfilename('fullpath'));
addpath(fullfile(base, '..', 'prox'));
addpath(fullfile(base, '..', '..', '..', 'shared', 'utils'));

nt = 8;  nx = 6;  ny = 5;
sigma = 0.05;
problem = [];   % not used by prox_ke_cc

pass = 0; fail = 0;
tol  = 1e-10;
rng(42);

% --- Random positive-density input ---
x_in.rho = abs(randn(nt, nx, ny)) + 0.1;
x_in.mx  = 0.1 * randn(nt, nx, ny);
x_in.my  = 0.1 * randn(nt, nx, ny);

% -------------------------------------------------------------------------
%% 1. OUTPUT SHAPES
% -------------------------------------------------------------------------
x_out = prox_ke_cc(x_in, sigma, problem);

[pass,fail] = check_size('rho shape', x_out.rho, [nt, nx, ny], pass, fail);
[pass,fail] = check_size('mx  shape', x_out.mx,  [nt, nx, ny], pass, fail);
[pass,fail] = check_size('my  shape', x_out.my,  [nt, nx, ny], pass, fail);

% -------------------------------------------------------------------------
%% 2. KKT CONDITIONS (pointwise)
%
%   Stationarity wrt mx:  mx_out/rho_out + (mx_out - mx_in)/sigma = 0
%   Stationarity wrt my:  my_out/rho_out + (my_out - my_in)/sigma = 0
%   Stationarity wrt rho: -(mx_out^2+my_out^2)/(2*rho_out^2) + (rho_out - rho_in)/sigma = 0
% -------------------------------------------------------------------------
r  = x_out.rho;
mx = x_out.mx;
my = x_out.my;

kkt_mx  = mx ./ r + (mx - x_in.mx)  / sigma;
kkt_my  = my ./ r + (my - x_in.my)  / sigma;
kkt_rho = -(mx.^2 + my.^2) ./ (2*r.^2) + (r - x_in.rho) / sigma;

[pass,fail] = report('KKT wrt mx',  max(abs(kkt_mx(:))),  tol, pass, fail);
[pass,fail] = report('KKT wrt my',  max(abs(kkt_my(:))),  tol, pass, fail);
[pass,fail] = report('KKT wrt rho', max(abs(kkt_rho(:))), tol, pass, fail);

% -------------------------------------------------------------------------
%% 3. NON-NEGATIVITY
% -------------------------------------------------------------------------
[pass,fail] = report('rho_out >= 0', max(-min(x_out.rho(:), 0)), tol, pass, fail);

% -------------------------------------------------------------------------
%% 4. my = 0 SYMMETRY
% -------------------------------------------------------------------------
x_in0     = x_in;
x_in0.my  = zeros(nt, nx, ny);
x_out0    = prox_ke_cc(x_in0, sigma, problem);
[pass,fail] = report('my=0 in => my=0 out', max(abs(x_out0.my(:))), tol, pass, fail);

% -------------------------------------------------------------------------
%% 5. CONSISTENCY WITH 1D AT EACH y-SLICE
%    With ny=1, result should equal applying prox_ke_cc independently.
% -------------------------------------------------------------------------
x_in1.rho = x_in.rho(:,:,1);
x_in1.mx  = x_in.mx(:,:,1);
x_in1.my  = x_in.my(:,:,1);

x_out1 = prox_ke_cc(x_in1, sigma, problem);

x_in3.rho = x_in.rho(:,:,1:1);
x_in3.mx  = x_in.mx(:,:,1:1);
x_in3.my  = x_in.my(:,:,1:1);
x_out3 = prox_ke_cc(x_in3, sigma, problem);

err_rho = max(abs(x_out1.rho(:) - x_out3.rho(:)));
err_mx  = max(abs(x_out1.mx(:)  - x_out3.mx(:)));
[pass,fail] = report('2D slice matches 2D with ny=1 (rho)', err_rho, tol, pass, fail);
[pass,fail] = report('2D slice matches 2D with ny=1 (mx)',  err_mx,  tol, pass, fail);

% -------------------------------------------------------------------------
fprintf('\n--- Results: %d passed, %d failed ---\n', pass, fail);

% =========================================================================
function [pass, fail] = report(name, err, tol, pass, fail)
    if err < tol
        fprintf('  PASS  %-45s  err = %.2e\n', name, err);
        pass = pass + 1;
    else
        fprintf('  FAIL  %-45s  err = %.2e  (tol=%.2e)\n', name, err, tol);
        fail = fail + 1;
    end
end

function [pass, fail] = check_size(name, out, expected, pass, fail)
    if isequal(size(out), expected)
        fprintf('  PASS  %-45s  [%s]\n', name, num2str(size(out)));
        pass = pass + 1;
    else
        fprintf('  FAIL  %-45s  expected [%s], got [%s]\n', name, num2str(expected), num2str(size(out)));
        fail = fail + 1;
    end
end
