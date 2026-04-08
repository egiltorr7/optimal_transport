function x = solve_cubic(a, b, c, d)
%SOLVE_CUBIC  Element-wise real root of a cubic via Cardano's formula.
%
%   X = SOLVE_CUBIC(A, B, C, D) returns, for each element, the largest real
%   root of  a*x^3 + b*x^2 + c*x + d = 0.  All inputs must be the same size
%   (or scalar).

% Normalise to monic form  x^3 + b x^2 + c x + d
b = b ./ a;
c = c ./ a;
d = d ./ a;

% Depress to  t^3 + p t + q  via the substitution x = t - b/3
p = c - b.^2 / 3;
q = 2*b.^3 / 27 - b.*c / 3 + d;

delta = q.^2 / 4 + p.^3 / 27;   % discriminant

x = zeros(size(b));

% Case 1: p = 0  =>  t = -q^(1/3)
ind = (p == 0);
x(ind) = -nthroot(q(ind), 3);

% Case 2: delta = 0, p ~= 0  =>  two distinct real roots; pick the larger
ind = (p ~= 0) & (delta == 0);
x(ind) = max(3*q(ind) ./ p(ind), -3*q(ind) ./ p(ind) / 2);

% Case 3: delta > 0  =>  one real root (Cardano)
ind = (p ~= 0) & (delta > 0);
s = sqrt(delta(ind));
x(ind) = nthroot(-q(ind)/2 - s, 3) + nthroot(-q(ind)/2 + s, 3);

% Case 4: delta < 0  =>  three real roots; pick the largest (trigonometric method)
ind = (p ~= 0) & (delta < 0);
r     = 2 * sqrt(-p(ind) / 3);
theta = real(acos(3*q(ind) ./ (p(ind) .* r)) / 3);
x(ind) = r .* cos(theta);

% Undo the depression substitution
x = x - b / 3;

end
