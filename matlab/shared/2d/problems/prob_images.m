function prob = prob_images(opts)
% PROB_IMAGES  Transport one greyscale image onto another.
%
%   prob = prob_images(img0=..., img1=..., ...)
%
%   Returns the same contract as prob_gaussian and friends: rho0_func and
%   rho1_func accept broadcasting xx (nx x 1) and yy (1 x ny) and return
%   (nx x ny).  The images are therefore RESOLUTION-INDEPENDENT -- the same
%   prob works at nx=ny=128 or 256 without touching this file, because the
%   picture is resampled onto whatever grid setup_problem asks for rather
%   than being pre-resized to one.
%
%   Name-value options:
%     img0, img1   paths to the two images (anything imread handles)
%     invert0/1    true = dark pixels become HIGH density.  Needed for
%                  ink-on-white artwork (a lithograph), where the raw
%                  intensity would put all the mass in the white background
%                  and leave the subject at zero.  Photographs with a bright
%                  subject on a darker ground want false.
%     floor_frac   delta below; the density floor, as a fraction of the peak
%     name         label used in output filenames
%
%   THE FLOOR IS NOT COSMETIC.  The kinetic-energy prox minimises
%   |m|^2/(2 rho), so a cell with rho = 0 makes the fluid formulation
%   degenerate, and photographs are full of near-black regions.  This
%   returns
%
%       rho = (1 - delta) * img/max(img) + delta
%
%   so rho >= delta > 0 everywhere, and setup_problem's own normalisation
%   then scales it to unit mass.  delta = 1e-2 is the usual choice in the
%   Benamou-Brenier image-interpolation literature: large enough to keep the
%   prox well posed, small enough that the picture is still the picture.
%
%   Aspect ratio is preserved: the image is centred in [0,1]^2 with the
%   short axis letterboxed, rather than stretched to fill the square, so the
%   portraits are not distorted.  Outside the letterbox the density is the
%   floor.

    arguments
        opts.img0       (1,:) char
        opts.img1       (1,:) char
        opts.invert0    (1,1) logical = false
        opts.invert1    (1,1) logical = false
        opts.floor_frac (1,1) double  = 1e-2
        opts.name       (1,:) char    = 'images'
    end

    % Load once here, not inside the closures: setup_problem calls each
    % rho*_func once, but a sweep builds the problem repeatedly.
    [F0, x0, y0] = local_load(opts.img0, opts.invert0);
    [F1, x1, y1] = local_load(opts.img1, opts.invert1);

    prob.name      = opts.name;
    prob.rho0_func = @(xx, yy) local_sample(F0, x0, y0, xx, yy, opts.floor_frac);
    prob.rho1_func = @(xx, yy) local_sample(F1, x1, y1, xx, yy, opts.floor_frac);
end


function [F, xs, ys] = local_load(path, invert)
% Read, convert to greyscale in [0,1], orient, and return with the source
% coordinate vectors that letterbox it into [0,1]^2.
    if exist(path, 'file') ~= 2
        error('prob_images:missingImage', 'image not found: %s', path);
    end
    G = imread(path);
    G = double(G);
    if ndims(G) == 3
        % Rec. 601 luma. Written out rather than calling rgb2gray so this
        % does not require the Image Processing Toolbox.
        G = 0.299*G(:,:,1) + 0.587*G(:,:,2) + 0.114*G(:,:,3);
    end
    G = G / max(G(:));
    if invert
        G = 1 - G;
    end

    % imread gives (row, col) with row 1 at the TOP. The solver wants
    % F(i,j) with i indexing x (left to right) and j indexing y (bottom to
    % top), so flip the rows and transpose.
    F = flipud(G).';            % (W x H), F(x, y)
    [W, H] = size(F);

    % Letterbox into [0,1]^2 preserving aspect ratio.
    if W >= H
        xs = linspace(0, 1, W);
        half = 0.5 * H / W;
        ys = linspace(0.5 - half, 0.5 + half, H);
    else
        ys = linspace(0, 1, H);
        half = 0.5 * W / H;
        xs = linspace(0.5 - half, 0.5 + half, W);
    end
end


function rho = local_sample(F, xs, ys, xx, yy, delta)
% Resample onto the solver's cell centres and apply the positivity floor.
    [Xq, Yq] = meshgrid(xx, yy);          % (ny x nx)
    % interp2 wants V indexed (y, x); F is (x, y), hence the transpose.
    V = interp2(xs, ys, F.', Xq, Yq, 'linear', 0);   % 0 outside the letterbox
    V = V.';                                          % back to (nx x ny)

    V = max(V, 0);
    peak = max(V(:));
    if peak <= 0
        error('prob_images:blankImage', 'image sampled to all zeros');
    end
    rho = (1 - delta) * (V / peak) + delta;
end
