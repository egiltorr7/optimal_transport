function bp = precomp_banded_proj_spike2(problem, vareps)
% PRECOMP_BANDED_PROJ_SPIKE2  Precompute for the Spike p=2 tridiagonal solver.
%
%   bp = precomp_banded_proj_spike2(problem, vareps)
%
%   Extends precomp_banded_proj with two additional fields needed by
%   proj_fokker_planck_spike2:
%
%     bp.spike_pivots  (nt-1 x nx x ny)
%         Thomas-modified pivots for T1 = T[1:nt-1, 1:nt-1], precomputed
%         once so the per-iteration solve only needs to sweep the RHS.
%
%     bp.spike_v  (nt-1 x nx x ny)
%         Spike vector v = T1^{-1} * c, where c = up(nt-1) * e_{nt-1}.
%         Captures how the last coupling entry propagates back through Block 1.
%         Used to form the Schur complement and correct the Block 1 solution.
%
%   Background — Spike p=2 partition:
%     The (nt x nt) tridiagonal T is split into two blocks:
%       Block 1: T1 = T[1:nt-1, 1:nt-1]  — well-conditioned (M1d(1..nt-1) >= 1)
%       Block 2: T[nt,nt]                 — positive scalar for all eps
%     and coupled via:
%       T(nt-1, nt) = up(nt-1)  (Block 1 last row -> Block 2)
%       T(nt,   nt-1) = lo(nt-1)  (Block 2 -> Block 1 last row)
%
%   Why Block 2 is always safe:
%     T(nt,nt) = M0(nt,nt) + lxy*M1d(nt) + lxy^2*M2(nt)
%     The discriminant of this quadratic in lxy is negative for all physically
%     relevant parameters, so T(nt,nt) > 0 for all modes regardless of eps.
%
%   All output arrays are regular (CPU) doubles.  Cast spike_pivots and
%   spike_v to gpuArray in discretize_then_optimize if cfg.use_gpu is set.

    % --- Standard diagonal precomputation (reuse existing function) ---
    bp = precomp_banded_proj(problem, vareps);

    nt  = problem.nt;
    nx  = problem.nx;
    ny  = problem.ny;
    ntm = nt - 1;   % = number of rows in Block 1

    % Diagonals of T1 = T[1:nt-1, 1:nt-1]
    %   lo1(i) = T(i+1, i)  for i = 1..nt-2  ->  bp.lower_all(1:nt-2, :, :)
    %   ma1(i) = T(i,   i)  for i = 1..nt-1  ->  bp.main_all( 1:nt-1, :, :)
    %   up1(i) = T(i, i+1)  for i = 1..nt-2  ->  bp.upper_all(1:nt-2, :, :)
    lo1 = bp.lower_all(1:nt-2, :, :);   % (nt-2 x nx x ny)
    ma1 = bp.main_all( 1:ntm,  :, :);   % (nt-1 x nx x ny)
    up1 = bp.upper_all(1:nt-2, :, :);   % (nt-2 x nx x ny)

    % --- Thomas forward sweep on T1: compute modified pivots ---
    % b(1) = ma1(1)
    % b(i) = ma1(i) - (lo1(i-1) / b(i-1)) * up1(i-1)   for i = 2..nt-1
    % These depend only on the matrix, not the RHS, so they can be precomputed.
    b = ma1;   % copy; will modify in place
    for i = 2:ntm
        b(i,:,:) = b(i,:,:) - (lo1(i-1,:,:) ./ b(i-1,:,:)) .* up1(i-1,:,:);
    end
    bp.spike_pivots = b;   % (nt-1 x nx x ny)

    % --- Spike vector: v = T1^{-1} * c  where c = up(nt-1) * e_{nt-1} ---
    %
    % c has a single nonzero at position nt-1:  c(nt-1) = bp.upper_all(ntm,:,:)
    %
    % Forward sweep of Thomas with this RHS:
    %   d(1..ntm-1) are all 0, so d(i) -= w(i)*d(i-1) leaves them at 0.
    %   d(ntm) = up(nt-1) stays unchanged.
    %
    % Back-substitution:
    %   v(ntm) = d(ntm) / b(ntm) = up(nt-1) / b(ntm)
    %   v(i)   = (0 - up1(i)*v(i+1)) / b(i)   for i = ntm-1 down to 1
    spike_v          = zeros(ntm, nx, ny);
    spike_v(ntm,:,:) = bp.upper_all(ntm,:,:) ./ b(ntm,:,:);
    for i = ntm-1:-1:1
        spike_v(i,:,:) = -(up1(i,:,:) .* spike_v(i+1,:,:)) ./ b(i,:,:);
    end
    bp.spike_v = spike_v;   % (nt-1 x nx x ny)
end
