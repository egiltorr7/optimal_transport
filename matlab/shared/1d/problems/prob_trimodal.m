function prob = prob_trimodal()
% PROB_TRIMODAL  Transport a trimodal distribution to a single broad Gaussian.
%
%   rho0 = (1/3)[ N(0.2, 0.05^2) + N(0.5, 0.05^2) + N(0.8, 0.05^2) ]
%   rho1 = N(0.5, 0.12^2)
%
%   For eps -> 0 (pure OT) the optimal plan merges the three sharp peaks into
%   one broad mode.  The left and right peaks travel distance 0.3 to the centre
%   while the central peak stays.  No analytical OT/SB solution is available;
%   use Sinkhorn as reference.
%
%   This problem tests:
%     - Multi-modal to unimodal mass transport (topology change).
%     - Performance at moderate eps where diffusion assists the merging.
%     - Robustness of the FP projection when the density is highly concentrated.

    Normal = @(x, mu, sig) exp(-0.5*((x - mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'trimodal';
    prob.rho0_func = @(xx) (Normal(xx, 0.20, 0.05) + ...
                             Normal(xx, 0.50, 0.05) + ...
                             Normal(xx, 0.80, 0.05)) / 3;
    prob.rho1_func = @(xx)  Normal(xx, 0.50, 0.12);
end
