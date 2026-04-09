function prob = prob_gaussian()
% PROB_GAUSSIAN  Transport a Gaussian from x=1/3 to x=2/3 (uniform in y).
%
%   rho0_func(xx, yy) and rho1_func(xx, yy) accept broadcasting inputs
%   xx (nx x 1) and yy (1 x ny) and return (nx x ny).

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'gaussian';
    prob.rho0_func = @(xx, yy) Normal(xx, 1/3, 0.05) .* ones(size(yy));
    prob.rho1_func = @(xx, yy) Normal(xx, 2/3, 0.05) .* ones(size(yy));
end
