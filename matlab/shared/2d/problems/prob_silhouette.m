function prob = prob_silhouette(opts)
% PROB_SILHOUETTE  Transport one mollified silhouette onto another.
%
%   prob = prob_silhouette(img0=..., img1=..., ...)
%
%   Same contract as prob_gaussian: rho0_func/rho1_func take broadcasting
%   xx (nx x 1), yy (1 x ny) and return (nx x ny), so the shapes are
%   resampled onto whatever grid setup_problem asks for and the same file
%   serves nx=128 and nx=256.
%
%   The silhouette comes from the image's ALPHA channel, not from
%   thresholding its colours: a sprite on a transparent background already
%   carries an exact mask, so there is no threshold to tune and no halo
%   from anti-aliased edges being mistaken for interior.
%
%   MOLLIFICATION.  A raw indicator 1_Omega is discontinuous, which the
%   staggered-grid operators resolve as a one-cell jump and which makes the
%   transport grid-dependent. This convolves it with the standard C^infinity
%   BUMP FUNCTION
%
%       phi(r) = exp(-1/(1-r^2))   for |r| < 1,   0 otherwise
%
%   rather than with a Gaussian, and the distinction is the whole point: a
%   Gaussian is smooth but has INFINITE support, so the density would be
%   nonzero (if tiny) everywhere, including against the wall. The bump is
%   smooth AND compactly supported, so 1_Omega * phi_w is C^infinity, equals
%   1 well inside Omega, and is EXACTLY zero more than `mollify_w` away from
%   it -- zero as in the bit pattern, not as in 1e-12.
%
%   Combined with `margin` (the shape is placed inside
%   [margin, 1-margin]^2 with margin > mollify_w), the density is therefore
%   identically zero on the domain boundary, which is what the no-flux wall
%   BCs want.
%
%   Name-value options:
%     img0, img1   paths to two RGBA images (alpha = the silhouette)
%     mollify_w    smoothing radius, in DOMAIN units (not pixels or cells),
%                  so the continuum density is resolution-independent
%     margin       keep-out band at the domain edge; must exceed mollify_w
%     floor_frac   uniform density floor as a fraction of the peak.
%                  DEFAULT 0 -- i.e. genuinely zero outside the shape.
%                  See the note below before raising it.
%     name         label used in output filenames
%
%   ON THE FLOOR.  The kinetic-energy prox minimises |m|^2/(2 rho), so
%   rho = 0 is a degenerate point of the fluid formulation, and the image
%   literature routinely floors the density for that reason. It is NOT
%   needed here and defaults to 0: rho = 0 only at t = 0 and t = 1, where
%   the density is a fixed boundary condition rather than an unknown, and
%   the interior rho(t) that the solver actually optimises is free to be
%   positive wherever mass has to flow. If the ADMM misbehaves on this
%   problem, a small floor (1e-3) is the first knob to try -- but reach for
%   it as a diagnostic, not as a default.

    arguments
        opts.img0       (1,:) char
        opts.img1       (1,:) char
        opts.mollify_w  (1,1) double = 0.02
        opts.margin     (1,1) double = 0.06
        opts.floor_frac (1,1) double = 0.0
        opts.name       (1,:) char   = 'silhouette'
    end

    if opts.margin <= opts.mollify_w
        error('prob_silhouette:marginTooSmall', ...
              ['margin (%.4g) must exceed mollify_w (%.4g), or the smoothed ' ...
               'shape reaches the wall and the density is not zero there'], ...
              opts.margin, opts.mollify_w);
    end

    [F0, x0, y0] = local_load(opts.img0, opts.margin);
    [F1, x1, y1] = local_load(opts.img1, opts.margin);

    prob.name      = opts.name;
    prob.rho0_func = @(xx, yy) local_sample(F0, x0, y0, xx, yy, opts);
    prob.rho1_func = @(xx, yy) local_sample(F1, x1, y1, xx, yy, opts);
end


function [F, xs, ys] = local_load(path, margin)
% Alpha channel -> binary mask, cropped to its bounding box and given the
% source coordinates that place it inside [margin, 1-margin]^2.
    if exist(path, 'file') ~= 2
        error('prob_silhouette:missingImage', 'image not found: %s', path);
    end
    [~, ~, alpha] = imread(path);
    if isempty(alpha)
        error('prob_silhouette:noAlpha', ...
              ['%s has no alpha channel; prob_silhouette needs a sprite on a ' ...
               'transparent background (use prob_images for a photograph)'], path);
    end
    mask = double(alpha) > 0.5 * double(max(alpha(:)));

    % Crop to the bounding box so `margin` means what it says regardless of
    % how much blank frame the source image happened to carry.
    r = find(any(mask, 2));  c = find(any(mask, 1));
    mask = mask(r(1):r(end), c(1):c(end));

    % (row, col) with row 1 at the top -> F(i,j) with i along x, j along y
    % increasing upward.
    F = double(flipud(mask)).';          % (W x H)
    [W, H] = size(F);

    % Fit inside [margin, 1-margin]^2, preserving aspect ratio.
    span = 1 - 2*margin;
    if W >= H
        xs = linspace(margin, 1 - margin, W);
        half = 0.5 * span * H / W;
        ys = linspace(0.5 - half, 0.5 + half, H);
    else
        ys = linspace(margin, 1 - margin, H);
        half = 0.5 * span * W / H;
        xs = linspace(0.5 - half, 0.5 + half, W);
    end
end


function rho = local_sample(F, xs, ys, xx, yy, opts)
% Resample onto the solver's cell centres, then mollify on that grid.
    [Xq, Yq] = meshgrid(xx, yy);                      % (ny x nx)
    V = interp2(xs, ys, F.', Xq, Yq, 'linear', 0);     % 0 outside the shape
    V = max(V.', 0);                                   % back to (nx x ny)

    % Mollify with a compactly supported bump, radius mollify_w in DOMAIN
    % units converted to cells. conv2 'same' zero-pads, which is the correct
    % boundary behaviour here: outside the shape there is nothing.
    dx = 1 / numel(xx);
    K = local_bump_kernel(opts.mollify_w / dx);   % width in CELLS, not rounded
    V = conv2(V, K, 'same');

    peak = max(V(:));
    if peak <= 0
        error('prob_silhouette:blankImage', 'silhouette sampled to all zeros');
    end
    V = V / peak;

    if opts.floor_frac > 0
        rho = (1 - opts.floor_frac) * V + opts.floor_frac;
    else
        rho = V;                                        % exactly 0 off-shape
    end
end


function K = local_bump_kernel(rad_cells)
% Normalised C^infinity bump whose support is a disc of radius EXACTLY
% `rad_cells` cells (a real number, not an integer):
%   phi(r) = exp(-1/(1-r^2)),  |r| < 1,   r = |offset| / rad_cells
% Entries at r >= 1 are exactly 0, so the kernel -- and hence the mollified
% density -- has genuinely compact support.
%
% `rad_cells` is deliberately NOT rounded, and the normalising radius is NOT
% inflated by a cell. An earlier version did both (round(), then /(rad+1)),
% which made the effective support 4 cells at nx=128 and 6 at nx=256 -- i.e.
% 0.031 and 0.023 in domain units, rather than the 0.02 requested at both.
% The two grids were then discretizing different continuum densities, which
% defeats the point of specifying the width in domain units at all. The
% giveaway was that refining 128 -> 256 cut the peak cell-to-cell jump by
% only 6/4 = 1.5x instead of the 2x a fixed-width ramp must give.
    R0 = ceil(rad_cells);
    [I, J] = meshgrid(-R0:R0, -R0:R0);
    R = sqrt(I.^2 + J.^2) / rad_cells;
    K = zeros(size(R));
    in = R < 1;
    K(in) = exp(-1 ./ (1 - R(in).^2));
    if ~any(in(:))      % mollify_w below the grid scale: nothing to smooth with
        K = 1;
        return;
    end
    K = K / sum(K(:));
end
