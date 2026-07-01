function prob = prob_four_peaks()
% PROB_FOUR_PEAKS  Aggregate four corner peaks into a central Gaussian.
%
%   rho0: four equal narrow Gaussians at the corners of the square [0.25, 0.75]^2
%
%         (0.25, 0.75)          (0.75, 0.75)
%              *                    *
%
%              *                    *
%         (0.25, 0.25)          (0.75, 0.25)
%
%         sigma_0 = 0.05,  equal weight 1/4 each.
%
%   rho1: a single broad Gaussian centred at (0.5, 0.5), sigma_1 = 0.14.
%         The width is chosen so that the bulk of the mass overlaps the
%         transport paths from all four corners.
%
%   Optimal transport (eps -> 0):
%     Each corner peak (mass 1/4) travels distance d = 0.25*sqrt(2) ≈ 0.354
%     to the centre.  By symmetry all four peaks follow identical radial paths,
%     giving  W_2^2 ≈ 4 * (1/4) * d^2 = 0.5 * 0.25^2 * 2 = 0.125.
%
%   Interest:
%     - Topology change: four distinct modes collapse to one.
%     - For small eps the trajectories remain narrow and well-separated;
%       for large eps they overlap early and the path is qualitatively different.
%     - No analytical solution available: use 2D Sinkhorn as reference.

    sig0 = 0.05;
    sig1 = 0.14;

    G0 = @(xx, yy, mx, my) exp(-((xx - mx).^2 + (yy - my).^2) / (2*sig0^2)) ...
                            / (2*pi*sig0^2);
    G1 = @(xx, yy) exp(-((xx - 0.5).^2 + (yy - 0.5).^2) / (2*sig1^2)) ...
                   / (2*pi*sig1^2);

    prob.rho0_func = @(xx, yy) ( G0(xx, yy, 0.25, 0.25) + ...
                                  G0(xx, yy, 0.75, 0.25) + ...
                                  G0(xx, yy, 0.75, 0.75) + ...
                                  G0(xx, yy, 0.25, 0.75) ) / 4;
    prob.rho1_func = @(xx, yy) G1(xx, yy);

    prob.name   = 'four_peaks';
    prob.sigma0 = sig0;
    prob.sigma1 = sig1;
    prob.w2_ot_approx = 0.125;   % approximate OT cost (symmetry argument)
end
