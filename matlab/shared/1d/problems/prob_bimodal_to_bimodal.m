function prob = prob_bimodal_to_bimodal()
% PROB_BIMODAL_TO_BIMODAL  Two symmetric peaks moving inward toward each other.
%
%   rho0 = N(0.2, 0.04^2) + N(0.8, 0.04^2)   (far apart, symmetric)
%   rho1 = N(0.35, 0.04^2) + N(0.65, 0.04^2) (closer together)
%
%   For eps=0 (OT) the optimal plan is NON-CROSSING:
%     left  peak:  0.2  -> 0.35   (shift = +0.15)
%     right peak:  0.8  -> 0.65   (shift = -0.15)
%   KE = W_2^2 = (1/2)*0.15^2 + (1/2)*0.15^2 = 0.0225
%
%   The crossing plan (left->right, right->left, shift=0.45 each) has
%   KE = (1/2)*0.45^2 + (1/2)*0.45^2 = 0.2025 >> 0.0225, so the
%   solver should strongly prefer the non-crossing plan.
%
%   The displacement interpolation at time t is:
%     rho(t,x) = (1/2)*N(0.2+0.15*t, 0.04^2) + (1/2)*N(0.8-0.15*t, 0.04^2)
%
%   Analytical OT solution available via analytical_ot_bimodal_to_bimodal.

    Normal = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);

    prob.name      = 'bimodal_to_bimodal';
    prob.rho0_func = @(xx) Normal(xx, 0.20, 0.04) + Normal(xx, 0.80, 0.04);
    prob.rho1_func = @(xx) Normal(xx, 0.35, 0.04) + Normal(xx, 0.65, 0.04);

    % Parameters for analytical solution
    prob.mu_L0  = 0.20;   prob.mu_L1  = 0.35;
    prob.mu_R0  = 0.80;   prob.mu_R1  = 0.65;
    prob.sigma  = 0.04;
    prob.ke_ot  = 0.0225;   % exact OT kinetic energy
end
