function [rho, mx, my, outs] = sb2d_admm(rho0, rho1, opts)
%SB2D_ADMM  Solve the 2-D Schrödinger bridge problem via ADMM.
%
%   [RHO, MX, MY, OUTS] = SB2D_ADMM(RHO0, RHO1, OPTS) finds the density RHO
%   (size (NT-1) x NX x NY), x-momentum MX (NT x (NX-1) x NY), and
%   y-momentum MY (NT x NX x (NY-1)) that solve:
%
%       min   int_0^1 int_Omega  (mx^2 + my^2) / rho  dA dt
%       s.t.  d_t rho + d_x mx + d_y my + eps * (Delta_x + Delta_y) rho = 0
%             rho(0, .) = rho0,   rho(1, .) = rho1
%             (mx, my) . n = 0  on domain boundary
%
%   Grid layout (arrays: time x x-space x y-space):
%     rho : (NT-1) x NX     x NY     -- time faces,    space cell-centres
%     mx  : NT     x (NX-1) x NY     -- time centres,  x-faces,   y-centres
%     my  : NT     x NX     x (NY-1) -- time centres,  x-centres, y-faces
%     phi : NT     x NX     x NY     -- time centres,  space cell-centres (dual)
%
%   Operator implementation:
%     All staggered-grid operators (interpolation, derivative) are implemented
%     as diff/average of adjacent array slices with appropriate BC padding.
%     No explicit matrices are formed or stored -- every operator is a single
%     cat + diff or cat + average, mapping to a handful of GPU kernels.
%
%   Woodbury projection:
%     Full 3-D DCT/IDCT: mirt_dctn/mirt_idctn (CPU) or dct/idct (GPU).
%     Partial transforms (time-only IDCT, xy-only IDCT): always dct/idct
%     with the dim argument, which supports gpuArray since R2022a.
%
%   OPTS fields:
%     nt          - number of time intervals
%     maxIter     - maximum ADMM iterations
%     gamma       - ADMM penalty parameter
%     vareps      - Schrödinger regularisation coefficient
%     use_gpu     - (optional, default false) use GPU  [requires R2022a+]
%     postprocess - (optional, default false) track cost and constraint viol

    %% Grid
    nt  = opts.nt;
    [nx, ny] = size(rho0);
    ntm = nt - 1;
    nxm = nx - 1;
    nym = ny - 1;
    dx  = 1 / nx;
    dy  = 1 / ny;
    dt  = 1 / nt;

    %% Parameters
    gamma   = opts.gamma;
    maxIter = opts.maxIter;
    vareps  = opts.vareps;
    sigma   = 1 / gamma;

    use_gpu = isfield(opts, 'use_gpu') && opts.use_gpu;

    track_postprocess = isfield(opts, 'postprocess') && opts.postprocess;
    if track_postprocess
        cost_history    = zeros(maxIter, 1);
        constraint_viol = zeros(maxIter, 1);
    end

    %% DCT eigenvalues for the Woodbury projection
    %
    %  In 3-D (time x x x y), AA* is diagonalised by the 3-D separable DCT
    %  with eigenvalue:
    %
    %    lambda(l, kx, ky) = lambda_t(l) + lambda_xy(kx,ky)
    %                       + eps^2 * lambda_xy(kx,ky)^2 * sigma_t(l)
    %
    %  where lambda_xy = lambda_x + lambda_y  (from the 2-D spatial Laplacian)
    %  and   sigma_t(l) = cos(pi*l*dt/2)^2   (from the It_phi * It_rho term).
    %
    %  The rank-2 correction in time (from the time-BC asymmetry) is handled
    %  by a 2x2 Woodbury system per (kx,ky) spatial mode.

    lambda_x = (2 - 2*cos(pi*dx*(0:nxm))) / dx^2;    % 1 x nx
    lambda_y = (2 - 2*cos(pi*dy*(0:nym))) / dy^2;    % 1 x ny
    lambda_t = (2 - 2*cos(pi*dt*(0:ntm)')) / dt^2;   % nt x 1

    % Broadcast to nt x nx x ny
    lx3  = reshape(lambda_x, 1, nx, 1);
    ly3  = reshape(lambda_y, 1, 1, ny);
    lt3  = reshape(lambda_t, nt, 1, 1);
    sig3 = reshape(cos(pi*dt*(0:ntm)'/2).^2, nt, 1, 1);

    lxy3          = lx3 + ly3;                                    % 1  x nx x ny
    lambda_biharm = lt3 + lxy3 + vareps^2 * lxy3.^2 .* sig3;    % nt x nx x ny
    lambda_biharm(1,1,1) = 1.0;    % gauge fix: DC mode -> 1 (zeroed later)

    %% Woodbury precomputation  (CPU, done once at O(N log N))
    %
    %  D1_sphys(:,kx,ky)   = P_DCT^{-1} e_1   in (physical-t, x-DCT, y-DCT) space
    %  D_nt_sphys(:,kx,ky) = P_DCT^{-1} e_nt
    %  c_k(kx,ky)          = -eps*(lambda_x(kx)+lambda_y(ky))/dt
    %  M                   = 2x2 per-(kx,ky) Woodbury matrix
    e1_t  = [1; zeros(nt-1,1)];
    ent_t = [zeros(nt-1,1); 1];
    q_e1  = dct(e1_t);
    q_ent = dct(ent_t);

    D1_hat          = reshape(q_e1,  nt,1,1) ./ lambda_biharm;
    D1_hat(1,1,1)   = 0;
    D1_sphys        = idct(D1_hat, [], 1);    % IDCT in time only

    D_nt_hat        = reshape(q_ent, nt,1,1) ./ lambda_biharm;
    D_nt_hat(1,1,1) = 0;
    D_nt_sphys      = idct(D_nt_hat, [], 1);

    c_k   = -vareps * lxy3 / dt;                  % 1 x nx x ny

    M11   =  1 + c_k .* D1_sphys(1,:,:);
    M12   = -c_k .* D_nt_sphys(1,:,:);
    M21   =  c_k .* D1_sphys(nt,:,:);
    M22   =  1 - c_k .* D_nt_sphys(nt,:,:);
    det_M = M11.*M22 - M12.*M21;
    det_M(abs(det_M) < eps) = 1;

    %% Move Woodbury arrays to GPU (if requested)
    if use_gpu
        lambda_biharm = gpuArray(lambda_biharm);
        D1_sphys      = gpuArray(D1_sphys);
        D_nt_sphys    = gpuArray(D_nt_sphys);
        c_k           = gpuArray(c_k);
        M11 = gpuArray(M11); M12 = gpuArray(M12);
        M21 = gpuArray(M21); M22 = gpuArray(M22);
        det_M         = gpuArray(det_M);
    end

    %% Initialisation
    tt = reshape(linspace(0,1,nt+1)', nt+1, 1, 1);
    tt = tt(2:end-1);                              % ntm x 1 x 1

    rho0_3d = reshape(rho0, 1, nx, ny);            % 1 x nx x ny (time BC at t=0)
    rho1_3d = reshape(rho1, 1, nx, ny);            % 1 x nx x ny (time BC at t=1)

    rho = (1-tt).*rho0_3d + tt.*rho1_3d;          % ntm x nx x ny
    mx  = zeros(nt, nxm, ny);
    my  = zeros(nt, nx,  nym);

    if use_gpu
        rho0_3d = gpuArray(rho0_3d);
        rho1_3d = gpuArray(rho1_3d);
        rho = gpuArray(rho);
        mx  = gpuArray(mx);
        my  = gpuArray(my);
    end

    rho_tilde = rho;  mx_tilde = mx;  my_tilde = my;
    delta_rho = zeros(size(rho), 'like', rho);
    delta_mx  = zeros(size(mx),  'like', mx);
    delta_my  = zeros(size(my),  'like', my);

    %% ADMM main loop
    residual_diff = zeros(maxIter, 1);

    for iter = 1:maxIter

        % ---- Step 1: Proximal update (pointwise kinetic energy) ----
        %
        %  Interpolate consensus + dual to cell-centres (phi-grid).
        %  At each (t,x,y) cell, solve:
        %    min_{r,ux,uy}  (ux^2+uy^2)/r  +  (1/2sigma)||(r,ux,uy)-(r0,ux0,uy0)||^2
        %  r -> largest real root of a cubic (GPU-compatible vectorised solver);
        %  ux, uy -> closed form once r is known.

        tmp_rho = interp_t_at_phi(rho_tilde + delta_rho/gamma, rho0_3d, rho1_3d);
        tmp_mx  = interp_x_at_phi(mx_tilde  + delta_mx /gamma);
        tmp_my  = interp_y_at_phi(my_tilde  + delta_my /gamma);

        m2 = tmp_mx.^2 + tmp_my.^2;   % nt x nx x ny, |m|^2 at cell centres

        rho_new = solve_cubic_vec(2*sigma - tmp_rho, ...
                                  sigma^2 - 2*sigma*tmp_rho, ...
                                  -sigma*(sigma*tmp_rho + 0.5*m2));

        neg          = rho_new <= 1e-14;
        rho_new(neg) = 0;

        scale           = rho_new ./ (rho_new + sigma);
        mx_new_phi      = scale .* tmp_mx;
        my_new_phi      = scale .* tmp_my;
        mx_new_phi(neg) = 0;
        my_new_phi(neg) = 0;

        % Interpolate back to staggered grids
        rho_new = interp_t_at_rho(rho_new);
        mx_new  = interp_x_at_m(mx_new_phi);
        my_new  = interp_y_at_m(my_new_phi);

        % ---- Step 2: Project onto the Fokker-Planck constraint ----
        [rho_tilde_new, mx_tilde_new, my_tilde_new] = proj_div( ...
            rho_new - delta_rho/gamma, ...
            mx_new  - delta_mx /gamma, ...
            my_new  - delta_my /gamma);

        % ---- Convergence monitor ----
        d2 = sum((rho_tilde_new(:)-rho_tilde(:)).^2) ...
           + sum((mx_tilde_new(:) - mx_tilde(:)).^2)  ...
           + sum((my_tilde_new(:) - my_tilde(:)).^2);
        residual_diff(iter) = gather(sqrt(dt*dx*dy * d2));

        % ---- Step 3: Dual update ----
        delta_rho = delta_rho - gamma*(rho_new - rho_tilde_new);
        delta_mx  = delta_mx  - gamma*(mx_new  - mx_tilde_new);
        delta_my  = delta_my  - gamma*(my_new  - my_tilde_new);

        rho = rho_new;  mx = mx_new;  my = my_new;
        rho_tilde = rho_tilde_new;
        mx_tilde  = mx_tilde_new;
        my_tilde  = my_tilde_new;

        if mod(iter,50) == 0
            viol = calc_constraint_viol(rho_tilde, mx_tilde, my_tilde);
            fprintf('iter %4d: residual=%.3e  viol=%.3e\n', ...
                    iter, residual_diff(iter), gather(viol));
        end

        if track_postprocess
            cost_history(iter)    = gather(calc_cost(rho, mx, my));
            constraint_viol(iter) = gather(calc_constraint_viol(rho, mx, my));
        end
    end

    rho = gather(rho);
    mx  = gather(mx);
    my  = gather(my);

    outs.residual_diff = residual_diff;
    if track_postprocess
        outs.cost            = cost_history;
        outs.constraint_viol = constraint_viol;
    end


    %% ================================================================
    %% Staggered-grid operators via array slicing
    %%
    %%  Every operator is either:
    %%    - average of adjacent elements:  0.5*(u(1:end-1) + u(2:end))
    %%    - difference of adjacent elements:  diff(u)/h
    %%  Boundary conditions are enforced by concatenating the BC value
    %%  before applying the diff/average.  No matrices are formed or stored.
    %%  diff, cat, and slicing all map to a small number of GPU kernels.
    %% ================================================================

    % Helper: zero-BC slabs for x and y boundaries
    function z = zx(arr)
        % Column of zeros matching nt x 1 x ny, same type as arr
        z = zeros(size(arr,1), 1, size(arr,3), 'like', arr);
    end
    function z = zy(arr)
        % Slice of zeros matching nt x nx x 1, same type as arr
        z = zeros(size(arr,1), size(arr,2), 1, 'like', arr);
    end

    % ---- Interpolation (average adjacent staggered values) ----

    % rho-grid (ntm x nx x ny) -> phi-grid (nt x nx x ny), time BCs
    function out = interp_t_at_phi(in, bc0, bc1)
        ext = cat(1, bc0, in, bc1);                          % (nt+1) x nx x ny
        out = 0.5 * (ext(1:end-1,:,:) + ext(2:end,:,:));    % nt x nx x ny
    end

    % phi-grid (nt x nx x ny) -> rho-grid (ntm x nx x ny)
    function out = interp_t_at_rho(in)
        out = 0.5 * (in(1:end-1,:,:) + in(2:end,:,:));      % ntm x nx x ny
    end

    % mx-grid (nt x nxm x ny) -> phi-grid (nt x nx x ny), zero-flux BCs in x
    function out = interp_x_at_phi(in)
        ext = cat(2, zx(in), in, zx(in));                   % nt x (nxm+2) x ny
        out = 0.5 * (ext(:,1:end-1,:) + ext(:,2:end,:));    % nt x nx x ny
    end

    % phi-grid (nt x nx x ny) -> mx-grid (nt x nxm x ny)
    function out = interp_x_at_m(in)
        out = 0.5 * (in(:,1:end-1,:) + in(:,2:end,:));      % nt x nxm x ny
    end

    % my-grid (nt x nx x nym) -> phi-grid (nt x nx x ny), zero-flux BCs in y
    function out = interp_y_at_phi(in)
        ext = cat(3, zy(in), in, zy(in));                   % nt x nx x (nym+2)
        out = 0.5 * (ext(:,:,1:end-1) + ext(:,:,2:end));    % nt x nx x ny
    end

    % phi-grid (nt x nx x ny) -> my-grid (nt x nx x nym)
    function out = interp_y_at_m(in)
        out = 0.5 * (in(:,:,1:end-1) + in(:,:,2:end));      % nt x nx x nym
    end

    % ---- Derivatives (finite differences of adjacent staggered values) ----

    % rho-grid (ntm x nx x ny) -> phi-grid (nt x nx x ny), time BCs
    function out = deriv_t_at_phi(in, bc0, bc1)
        ext = cat(1, bc0, in, bc1);
        out = diff(ext, 1, 1) / dt;                          % nt x nx x ny
    end

    % phi-grid (nt x nx x ny) -> rho-grid (ntm x nx x ny)
    function out = deriv_t_at_rho(in)
        out = diff(in, 1, 1) / dt;                           % ntm x nx x ny
    end

    % mx-grid (nt x nxm x ny) -> phi-grid (nt x nx x ny), zero-flux BCs
    function out = deriv_x_at_phi(in)
        ext = cat(2, zx(in), in, zx(in));
        out = diff(ext, 1, 2) / dx;                          % nt x nx x ny
    end

    % phi-grid (nt x nx x ny) -> mx-grid (nt x nxm x ny)
    function out = deriv_x_at_m(in)
        out = diff(in, 1, 2) / dx;                           % nt x nxm x ny
    end

    % my-grid (nt x nx x nym) -> phi-grid (nt x nx x ny), zero-flux BCs
    function out = deriv_y_at_phi(in)
        ext = cat(3, zy(in), in, zy(in));
        out = diff(ext, 1, 3) / dy;                          % nt x nx x ny
    end

    % phi-grid (nt x nx x ny) -> my-grid (nt x nx x nym)
    function out = deriv_y_at_m(in)
        out = diff(in, 1, 3) / dy;                           % nt x nx x nym
    end


    %% ================================================================
    %% Fokker-Planck residual and projection
    %% ================================================================

    % FP residual  A(rho_in, mx_in, my_in)  on the phi-grid (nt x nx x ny).
    % Computes: d_t rho + d_x mx + d_y my + eps*(Delta_x + Delta_y)*rho
    function f = fp_res(rho_in, mx_in, my_in, bc0, bc1)
        rho_phi = interp_t_at_phi(rho_in, bc0, bc1);
        lap_rho = deriv_x_at_phi(deriv_x_at_m(rho_phi)) ...
                + deriv_y_at_phi(deriv_y_at_m(rho_phi));
        f = deriv_t_at_phi(rho_in, bc0, bc1) ...
          + deriv_x_at_phi(mx_in) ...
          + deriv_y_at_phi(my_in) ...
          + vareps * lap_rho;
    end

    % Invert AA* phi = f via the Woodbury identity.
    %
    %  AA* = P_DCT + E  where E is rank-2 in time (boundary asymmetry).
    %
    %  Steps:
    %    1. 3-D DCT of f                            [mirt_dctn / dct-per-dim]
    %    2. Pointwise divide by lambda_biharm        [element-wise]
    %    3. IDCT in time only -> (phys-t, kx, ky)   [idct(g,[],1)]
    %    4. 2x2 Woodbury system per (kx,ky) mode    [element-wise broadcast]
    %    5. IDCT in x and y -> physical space        [idct-per-dim]
    %
    %  All steps are element-wise or single large ops -> GPU-friendly.
    %  Steps 1/5: mirt_dctn/mirt_idctn used on CPU (faster, FFT-based ND DCT).
    %             On GPU, dct/idct with dim arg is used instead (native GPU).
    function phi = invert_biharmonic(f)
        % Step 1: 3-D DCT
        if use_gpu
            f_hat = dct(dct(dct(f,[],1),[],2),[],3);
        else
            f_hat = mirt_dctn(f);
        end

        % Step 2: solve P_DCT^{-1} f in DCT space
        g_hat        = f_hat ./ lambda_biharm;
        g_hat(1,1,1) = 0;              % gauge fix (DC mode)

        % Step 3: IDCT in time only -> (physical-t, x-DCT, y-DCT)
        g_sphys = idct(g_hat, [], 1);

        % Step 4: Woodbury 2x2 correction per (kx,ky) mode
        %   M * [a; b] = [g_sphys(1,:,:); g_sphys(nt,:,:)]
        rhs1   = g_sphys(1,:,:);
        rhs_nt = g_sphys(nt,:,:);
        a_hat  = ( M22.*rhs1 - M12.*rhs_nt) ./ det_M;   % 1 x nx x ny
        b_hat  = (-M21.*rhs1 + M11.*rhs_nt) ./ det_M;

        phi_sphys = g_sphys ...
                  - D1_sphys   .* (c_k .* a_hat) ...
                  + D_nt_sphys .* (c_k .* b_hat);        % nt x nx x ny

        % Step 5: IDCT in x and y -> physical space.
        % mirt_idctn transforms ALL dimensions so it cannot be used here for a
        % partial transform.  dct/idct with the dim argument works on both CPU
        % and GPU (R2022a+) and is the natural choice for partial transforms.
        phi = idct(idct(phi_sphys,[],2),[],3);
    end

    % Project (mu, psi_x, psi_y) onto the Fokker-Planck constraint set.
    %
    %  A*(phi) = (Dt_rho phi - eps * It_rho * Lap phi,  Dx_m phi,  Dy_m phi)
    function [rho_out, mx_out, my_out] = proj_div(mu, psi_x, psi_y)
        f    = fp_res(mu, psi_x, psi_y, rho0_3d, rho1_3d);
        viol = sqrt(dt*dx*dy * sum(f(:).^2, 'all'));
        if viol < 1e-13
            rho_out = mu;  mx_out = psi_x;  my_out = psi_y;
            return;
        end

        phi = invert_biharmonic(f);

        dphi_dx    = deriv_x_at_m(phi);                % nt x nxm x ny
        dphi_dy    = deriv_y_at_m(phi);                % nt x nx  x nym
        lap_phi    = deriv_x_at_phi(dphi_dx) + deriv_y_at_phi(dphi_dy);
        nablax_phi = interp_t_at_rho(lap_phi);         % ntm x nx x ny

        rho_out = mu    + deriv_t_at_rho(phi) - vareps * nablax_phi;
        mx_out  = psi_x + dphi_dx;
        my_out  = psi_y + dphi_dy;
    end


    %% ================================================================
    %% GPU-compatible vectorised cubic solver
    %%
    %%  Solves monic cubic  x^3 + b*x^2 + c*x + d = 0  element-wise.
    %%  Uses vectorised case selection (mask multiplication, not scatter
    %%  indexing) so every operation is element-wise -> GPU-friendly.
    %% ================================================================
    function x = solve_cubic_vec(b, c, d)
        p = c - b.^2 / 3;
        q = 2*b.^3 / 27 - b.*c / 3 + d;
        disc = q.^2/4 + p.^3/27;

        cbrt = @(v) sign(v) .* abs(v).^(1/3);

        % Avoid p=0 division in case2 (masked out, but prevents NaN on GPU)
        p_safe = p + (abs(p) < 1e-30) .* 1e-30;

        % Case 1: p ~ 0
        t1 = cbrt(-q);

        % Case 2: disc ~ 0, p ~= 0  (two coincident roots; take larger)
        t2 = max(3*q./p_safe, -1.5*q./p_safe);

        % Case 3: disc > 0  (one real root, Cardano)
        sq = sqrt(max(disc, 0));
        t3 = cbrt(-q/2 - sq) + cbrt(-q/2 + sq);

        % Case 4: disc < 0  (three real roots; take largest via trig method)
        r4     = 2*sqrt(max(-p/3, 0));
        denom4 = p_safe.*r4 + (abs(p_safe.*r4) < 1e-30).*1e-30;
        t4     = r4 .* cos(acos(min(max(3*q./denom4, -1), 1)) / 3);

        % Vectorised case selection (no scatter indexing)
        is1 = abs(p)    < 1e-28;
        is2 = ~is1 & (abs(disc) < 1e-28);
        is3 = ~is1 & ~is2 & (disc > 0);

        t = t1.*is1 + t2.*is2 + t3.*is3 + t4.*(~is1 & ~is2 & ~is3);
        x = t - b/3;
    end


    %% ================================================================
    %% Post-processing metrics
    %% ================================================================

    function cost = calc_cost(rho_in, mx_in, my_in)
        rho_phi = interp_t_at_phi(rho_in, rho0_3d, rho1_3d);
        mx_phi  = interp_x_at_phi(mx_in);
        my_phi  = interp_y_at_phi(my_in);
        m2      = mx_phi.^2 + my_phi.^2;

        integrand      = zeros(size(rho_phi), 'like', rho_phi);
        pos            = rho_phi > 1e-8;
        integrand(pos) = m2(pos) ./ rho_phi(pos);
        cost           = sum(integrand(:)) * dt * dx * dy;
    end

    function viol = calc_constraint_viol(rho_in, mx_in, my_in)
        f    = fp_res(rho_in, mx_in, my_in, rho0_3d, rho1_3d);
        viol = sqrt(dt * dx * dy * sum(f(:).^2, 'all'));
    end

end
