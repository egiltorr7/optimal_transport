% Checkpoint: version BEFORE the interior-only diffusion fix in fp_res
% (i.e., the version that was working for eps=1e-2 but oscillating for eps=1e-1)
%
% Key state at this checkpoint:
%   M0_bnd = -Dt_phi * Dt_rho        (negated — user's fix)
%   M1d(1) = 1 + vareps/dt           (swapped — user's change)
%   M1d(nt)= 1 - vareps/dt           (swapped — user's change)
%   proj update: +vareps * nablax_phi (user's change)
%   fp_res diffusion: interp_t_at_phi(rho_in, bc0, bc1)  <-- WITH boundary BCs
%   calc_constraint_viol: -vareps     (my earlier fix)
%
% The fp_res block at that point read:
%
%     function f = fp_res(rho_in, m_in, bc0, bc1)
%         nablax_r = interp_t_at_phi(rho_in, bc0, bc1);
%         nablax_r = deriv_x_at_phi(deriv_x_at_m(nablax_r), zeros_x, zeros_x);
%         f = deriv_t_at_phi(rho_in, bc0, bc1) + ...
%             deriv_x_at_phi(m_in, zeros_x, zeros_x) - vareps * nablax_r;
%     end
%
% The NEW version (just applied) changes line 1 of the body to:
%         nablax_r = It_phi * rho_in;   % interior nodes only — no bc0/bc1
%
% To revert: replace the new fp_res body with the block above.
