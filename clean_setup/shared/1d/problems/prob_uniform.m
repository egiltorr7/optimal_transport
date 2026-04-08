function prob = prob_uniform()
% PROB_UNIFORM  Transport uniform[0.2, 0.4] to uniform[0.6, 0.8].
%
%   Both distributions have the same width (0.2), so the OT map is a
%   pure translation by 0.4 and the velocity is constant everywhere in
%   the support.
%
%   For eps=0 (OT), the exact solution is available via analytical_ot_uniform:
%     rho(t,x) = Uniform[0.2 + 0.4*t, 0.4 + 0.4*t]
%     m(t,x)   = rho(t,x) * 0.4
%     KE       = W_2^2 = 0.4^2 = 0.16
%
%   For eps>0 (SB), no closed-form exists; use fine-grid ADMM as reference.
%
%   Note: the discontinuous boundaries are a useful stress test for the
%   solver's ability to handle non-smooth densities.

    prob.name      = 'uniform';
    prob.rho0_func = @(xx) double(xx >= 0.2 & xx <= 0.4);
    prob.rho1_func = @(xx) double(xx >= 0.6 & xx <= 0.8);

    % Store parameters for use with analytical_ot_uniform
    prob.a = 0.2;   prob.b = 0.4;
    prob.c = 0.6;   prob.d = 0.8;
end
