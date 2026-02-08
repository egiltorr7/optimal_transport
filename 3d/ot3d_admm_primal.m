% ADMM on the primal problem of the dynamic formulation of OT - 3D version
function [rho,mx,my,mz,outs] = ot3d_admm_primal(rho0, rho1, opts)

    % Grid Parameters
    nt = opts.nt;
    ntp = nt+1;

    nx = opts.nx;
    nxp = nx+1;

    ny = opts.ny;
    nyp = ny+1;

    nz = opts.nz;
    nzp = nz+1;

    dx = 1/nxp;
    dy = 1/nyp;
    dz = 1/nzp;
    dt = 1/ntp;

    ex = ones(nxp,1);
    ey = ones(nyp,1);
    ez = ones(nzp,1);
    et = ones(ntp,1);

    % More parameters
    tol = opts.tol;
    maxIter = opts.maxIter;
    tau = opts.tau;
    sigma = 1/tau;
    residual_diff = zeros(maxIter,1);
    residual_numerical = zeros(maxIter,1);
    gpu = opts.GPUcomputing;

    % Initialize variables
    % Initialize rho as linear interpolation in time from rho0 to rho1
    t_vec = reshape((0:nt-1)' * dt, nt, 1, 1, 1);  % Shape (nt, 1, 1, 1)
    rho = (1 - t_vec) .* reshape(rho0, 1, nxp, nyp, nzp) + t_vec .* reshape(rho1, 1, nxp, nyp, nzp);
    rho_new = rho;
    mx  = ones(ntp,nx,nyp,nzp); mx_new  = ones(ntp,nx,nyp,nzp);
    my  = ones(ntp,nxp,ny,nzp); my_new  = ones(ntp,nxp,ny,nzp);
    mz  = ones(ntp,nxp,nyp,nz); mz_new  = ones(ntp,nxp,nyp,nz);
    rho_tilde = ones(nt,nxp,nyp,nzp); a = ones(nt,nxp,nyp,nzp);
    mx_tilde = ones(ntp,nx,nyp,nzp);  my_tilde = ones(ntp,nxp,ny,nzp);
    mz_tilde = ones(ntp,nxp,nyp,nz);
    bx = ones(ntp,nx,nyp,nzp); by = ones(ntp,nxp,ny,nzp); bz = ones(ntp,nxp,nyp,nz);
    bx_interior = ones(nt,nx,ny,nz); by_interior = ones(nt,nx,ny,nz); bz_interior = ones(nt,nx,ny,nz);
    b_interior_sqrd = ones(nt,nx,ny,nz); a_interior = ones(nt,nx,ny,nz);

    phit = zeros(nt,nxp,nyp,nzp);
    phix = zeros(ntp,nx,nyp,nzp);
    phiy = zeros(ntp,nxp,ny,nzp);
    phiz = zeros(ntp,nxp,nyp,nz);

    % Create Derivative Matrix
    Dx = create_div_1d(nx);
    Dy = create_div_1d(ny);
    Dz = create_div_1d(nz);
    Dt = create_div_1d(nt);

    % Eigenvalues for laplacian
    [lambda_x, Vx] = Bmtx_eig(nx);
    iVx = Vx';
    [lambda_y, Vy] = Bmtx_eig(ny);
    iVy = Vy';
    [lambda_z, Vz] = Bmtx_eig(nz);
    iVz = Vz';
    [lambda_t, Vt] = Bmtx_eig(nt);
    iVt = Vt';

    % Boundary Conditions
    bc0 = rho0./dt;
    bc1 = -rho1./dt;

    if (opts.GPUcomputing)
        ex = gpuArray(ex); ey = gpuArray(ey); ez = gpuArray(ez); et = gpuArray(et);
        residual_diff = gpuArray(residual_diff);
        residual_numerical = gpuArray(residual_numerical);
        rho = gpuArray(rho); rho_new = gpuArray(rho_new);
        mx  = gpuArray(mx); mx_new  = gpuArray(mx_new);
        my  = gpuArray(my); my_new  = gpuArray(my_new);
        mz  = gpuArray(mz); mz_new  = gpuArray(mz_new);
        rho_tilde = gpuArray(rho_tilde);
        mx_tilde = gpuArray(mx_tilde);
        my_tilde = gpuArray(my_tilde);
        mz_tilde = gpuArray(mz_tilde);
        phix = gpuArray(phix); phiy = gpuArray(phiy); phiz = gpuArray(phiz);
        phit = gpuArray(phit);
        Dx = gpuArray(Dx); Dy = gpuArray(Dy); Dz = gpuArray(Dz); Dt = gpuArray(Dt);
        lambda_x = gpuArray(lambda_x); Vx = gpuArray(Vx);
        lambda_y = gpuArray(lambda_y); Vy = gpuArray(Vy);
        lambda_z = gpuArray(lambda_z); Vz = gpuArray(Vz);
        lambda_t = gpuArray(lambda_t); Vt = gpuArray(Vt);
        bc0 = gpuArray(bc0); bc1 = gpuArray(bc1);
        bx = gpuArray(bx); by = gpuArray(by); bz = gpuArray(bz);
        bx_interior = gpuArray(bx_interior);
        by_interior = gpuArray(by_interior);
        bz_interior = gpuArray(bz_interior);
        b_interior_sqrd = gpuArray(b_interior_sqrd);
        a_interior = gpuArray(a_interior);
        Device = opts.Device;
    end

    % Compute Laplacian eigenvalues: lambda_t + lambda_x + lambda_y + lambda_z
    if (opts.GPUcomputing)
        lambda_lap = gpuArray(squeeze(tensorprod(lambda_t, ex*ey'*ez')) + ...
                               squeeze(tensorprod(et, lambda_x*ey'*ez')) + ...
                               squeeze(tensorprod(et, ex*lambda_y'*ez')) + ...
                               squeeze(tensorprod(et, ex*ey'*lambda_z')));
    else
        lambda_lap = squeeze(tensorprod(lambda_t, ex*ey'*ez')) + ...
                     squeeze(tensorprod(et, lambda_x*ey'*ez')) + ...
                     squeeze(tensorprod(et, ex*lambda_y'*ez')) + ...
                     squeeze(tensorprod(et, ex*ey'*lambda_z'));
    end

    tic
    for iter = 1:maxIter

        % Prox for quadratic cost
        a = rho_tilde + sigma*phit;
        a_interior = a(:,2:nxp,2:nyp,2:nzp);
        bx = mx_tilde + sigma*phix;
        by = my_tilde + sigma*phiy;
        bz = mz_tilde + sigma*phiz;
        bx_interior = bx(2:ntp,:,2:nyp,2:nzp);
        by_interior = by(2:ntp,2:nxp,:,2:nzp);
        bz_interior = bz(2:ntp,2:nxp,2:nyp,:);
        b_interior_sqrd  = bx_interior.^2 + by_interior.^2 + bz_interior.^2;

        % Interior nodes
        rho_new(:,2:nxp,2:nyp,2:nzp) = solve_cubic(1, 2*sigma - a_interior, ...
            sigma^2 - 2*a_interior*sigma, ...
            -sigma*(sigma*a_interior+(b_interior_sqrd./2)), gpu);
        mx_new(2:ntp,:,2:nyp,2:nzp)  = (rho_new(:,2:nxp,2:nyp,2:nzp).*bx_interior)./...
            (rho_new(:,2:nxp,2:nyp,2:nzp)+sigma);
        my_new(2:ntp,2:nxp,:,2:nzp)  = (rho_new(:,2:nxp,2:nyp,2:nzp).*by_interior)./...
            (rho_new(:,2:nxp,2:nyp,2:nzp)+sigma);
        mz_new(2:ntp,2:nxp,2:nyp,:)  = (rho_new(:,2:nxp,2:nyp,2:nzp).*bz_interior)./...
            (rho_new(:,2:nxp,2:nyp,2:nzp)+sigma);

        % Corner boundaries (x=0,y=0,z=0)
        rho_new(:,1,1,1) = a(:,1,1,1);

        % Edge boundaries at t=0
        mx_new(1,:,1,1) = (rho0(2:nxp,1,1)'.*squeeze(bx(1,:,1,1)))./(rho0(2:nxp,1,1)'+sigma);
        my_new(1,1,:,1) = (rho0(1,2:nyp,1)'.*squeeze(by(1,1,:,1)))./(rho0(1,2:nyp,1)'+sigma);
        mz_new(1,1,1,:) = (rho0(1,1,2:nzp)'.*squeeze(bz(1,1,1,:)))./(rho0(1,1,2:nzp)'+sigma);

        % Face boundaries at t=0
        mx_new(1,:,2:nyp,2:nzp) = (rho0(2:nxp,2:nyp,2:nzp).*squeeze(bx(1,:,2:nyp,2:nzp)))./...
            (rho0(2:nxp,2:nyp,2:nzp)+sigma);
        my_new(1,2:nxp,:,2:nzp) = (rho0(2:nxp,2:nyp,2:nzp).*squeeze(by(1,2:nxp,:,2:nzp)))./...
            (rho0(2:nxp,2:nyp,2:nzp)+sigma);
        mz_new(1,2:nxp,2:nyp,:) = (rho0(2:nxp,2:nyp,2:nzp).*squeeze(bz(1,2:nxp,2:nyp,:)))./...
            (rho0(2:nxp,2:nyp,2:nzp)+sigma);

        % x=0 face (for t>0)
        a_interior = a(:,1,2:nyp,2:nzp);
        b_interior_sqrd = by(2:ntp,1,:,2:nzp).^2 + bz(2:ntp,1,2:nyp,:).^2;
        rho_new(:,1,2:nyp,2:nzp) = solve_cubic(1, 2*sigma - a_interior, ...
            sigma^2 - 2*a_interior*sigma, ...
            -sigma*(sigma*a_interior+(b_interior_sqrd./2)), gpu);
        my_new(2:ntp,1,:,2:nzp) = (squeeze(rho_new(:,1,2:nyp,2:nzp)).*squeeze(by(2:ntp,1,:,2:nzp)))./...
            (squeeze(rho_new(:,1,2:nyp,2:nzp))+sigma);
        mz_new(2:ntp,1,2:nyp,:) = (squeeze(rho_new(:,1,2:nyp,2:nzp)).*squeeze(bz(2:ntp,1,2:nyp,:)))./...
            (squeeze(rho_new(:,1,2:nyp,2:nzp))+sigma);

        % y=0 face (for t>0)
        a_interior = a(:,2:nxp,1,2:nzp);
        b_interior_sqrd = bx(2:ntp,:,1,2:nzp).^2 + bz(2:ntp,2:nxp,1,:).^2;
        rho_new(:,2:nxp,1,2:nzp) = solve_cubic(1, 2*sigma - a_interior, ...
            sigma^2 - 2*a_interior*sigma, ...
            -sigma*(sigma*a_interior+(b_interior_sqrd./2)), gpu);
        mx_new(2:ntp,:,1,2:nzp) = (rho_new(:,2:nxp,1,2:nzp).*squeeze(bx(2:ntp,:,1,2:nzp)))./...
            (rho_new(:,2:nxp,1,2:nzp)+sigma);
        mz_new(2:ntp,2:nxp,1,:) = (squeeze(rho_new(:,2:nxp,1,2:nzp)).*squeeze(bz(2:ntp,2:nxp,1,:)))./...
            (squeeze(rho_new(:,2:nxp,1,2:nzp))+sigma);

        % z=0 face (for t>0)
        a_interior = a(:,2:nxp,2:nyp,1);
        b_interior_sqrd = bx(2:ntp,:,2:nyp,1).^2 + by(2:ntp,2:nxp,:,1).^2;
        rho_new(:,2:nxp,2:nyp,1) = solve_cubic(1, 2*sigma - a_interior, ...
            sigma^2 - 2*a_interior*sigma, ...
            -sigma*(sigma*a_interior+(b_interior_sqrd./2)), gpu);
        mx_new(2:ntp,:,2:nyp,1) = (rho_new(:,2:nxp,2:nyp,1).*squeeze(bx(2:ntp,:,2:nyp,1)))./...
            (rho_new(:,2:nxp,2:nyp,1)+sigma);
        my_new(2:ntp,2:nxp,:,1) = (rho_new(:,2:nxp,2:nyp,1).*squeeze(by(2:ntp,2:nxp,:,1)))./...
            (rho_new(:,2:nxp,2:nyp,1)+sigma);

        % Handle edges at x=0, y=0
        a_interior = a(:,1,1,2:nzp);
        b_interior_sqrd = bz(2:ntp,1,1,:).^2;
        rho_new(:,1,1,2:nzp) = solve_cubic(1, 2*sigma - a_interior, ...
            sigma^2 - 2*a_interior*sigma, ...
            -sigma*(sigma*a_interior+(b_interior_sqrd./2)), gpu);
        mz_new(2:ntp,1,1,:) = (squeeze(rho_new(:,1,1,2:nzp)).*squeeze(bz(2:ntp,1,1,:)))./...
            (squeeze(rho_new(:,1,1,2:nzp))+sigma);

        % Handle edges at x=0, z=0
        a_interior = a(:,1,2:nyp,1);
        b_interior_sqrd = by(2:ntp,1,:,1).^2;
        rho_new(:,1,2:nyp,1) = solve_cubic(1, 2*sigma - a_interior, ...
            sigma^2 - 2*a_interior*sigma, ...
            -sigma*(sigma*a_interior+(b_interior_sqrd./2)), gpu);
        my_new(2:ntp,1,:,1) = (squeeze(rho_new(:,1,2:nyp,1)).*squeeze(by(2:ntp,1,:,1)))./...
            (squeeze(rho_new(:,1,2:nyp,1))+sigma);

        % Handle edges at y=0, z=0
        a_interior = a(:,2:nxp,1,1);
        b_interior_sqrd = bx(2:ntp,:,1,1).^2;
        rho_new(:,2:nxp,1,1) = solve_cubic(1, 2*sigma - a_interior, ...
            sigma^2 - 2*a_interior*sigma, ...
            -sigma*(sigma*a_interior+(b_interior_sqrd./2)), gpu);
        mx_new(2:ntp,:,1,1) = (rho_new(:,2:nxp,1,1).*squeeze(bx(2:ntp,:,1,1)))./...
            (rho_new(:,2:nxp,1,1)+sigma);

        % Handle negative densities
        neg_ind = (rho_new <= 1e-12);
        neg_ind_rho0 = (rho0 <= 1e-12);
        rho_new(neg_ind) = 0.0;
        mx_new(2:ntp,:,:,:) = mx_new(2:ntp,:,:,:) .* ~neg_ind(:,2:nxp,:,:);
        mx_new(1,:,:,:) = squeeze(mx_new(1,:,:,:)) .* ~neg_ind_rho0(2:nxp,:,:);
        my_new(2:ntp,:,:,:) = my_new(2:ntp,:,:,:) .* ~neg_ind(:,:,2:nyp,:);
        my_new(1,:,:,:) = squeeze(my_new(1,:,:,:)) .* ~neg_ind_rho0(:,2:nyp,:);
        mz_new(2:ntp,:,:,:) = mz_new(2:ntp,:,:,:) .* ~neg_ind(:,:,:,2:nzp);
        mz_new(1,:,:,:) = squeeze(mz_new(1,:,:,:)) .* ~neg_ind_rho0(:,:,2:nzp);

        % Projection to the set Au=b
        a = rho_new - sigma*phit;
        bx = mx_new - sigma*phix;
        by = my_new - sigma*phiy;
        bz = mz_new - sigma*phiz;
        [rho_tilde, mx_tilde, my_tilde, mz_tilde] = proj_div_free(a,bx,by,bz,bc0,bc1);

        % Last Step of ADMM
        phit = phit - tau*(rho_new-rho_tilde);
        phix = phix - tau*(mx_new-mx_tilde);
        phiy = phiy - tau*(my_new-my_tilde);
        phiz = phiz - tau*(mz_new-mz_tilde);

        % Calculate residual
        drho = rho_new - rho;
        dmx  = mx_new - mx;
        dmy  = my_new - my;
        dmz  = mz_new - mz;

        drho_ref = rho_new - opts.rho_numerical;
        dmx_ref  = mx_new - opts.mx_numerical;
        dmy_ref  = my_new - opts.my_numerical;
        dmz_ref  = mz_new - opts.mz_numerical;
        residual_diff(iter) = sqrt(dt*dx*dy*dz*(norm(drho,'fro')^2 + norm(dmx,'fro')^2 + ...
            norm(dmy,'fro')^2 + norm(dmz,'fro')^2));
        residual_numerical(iter) = sqrt(dt*dx*dy*dz*(norm(drho_ref,'fro')^2 + ...
            norm(dmx_ref,'fro')^2 + norm(dmy_ref,'fro')^2 + norm(dmz_ref,'fro')^2));

        if residual_diff(iter) < 1e-14
            opts.maxIter = iter;
            break
        end

        % update
        rho = rho_new;
        mx  = mx_new;
        my  = my_new;
        mz  = mz_new;

        % GPU Clearing memory
        if gpu && (mod(iter,50)==0) && (Device.AvailableMemory < 0.25*Device.TotalMemory)
            reset(Device)
        end

    end
    toc

    % Data to be saved for output
    rho = gather(rho);
    mx = gather(mx);
    my = gather(my);
    mz = gather(mz);

    outs.residual_diff = gather(residual_diff);
    outs.residual_numerical = gather(residual_numerical);

    %% Auxiliary Functions
    function D = create_div_1d(n)
        h = 1/(n+1);
        e=ones(n,1);
        D=spdiags([-e 1*e], [-1 0], n+1, n);
        D = full(D);
        D=D/h;
    end

    function [lambda,V] = Bmtx_eig(n)
        np = n+1;
        h = 1/np;
        xx = [0.5*h:h:1-h/2]';
        lambda=(2*ones(np,1)-2*cos([0:n]'*pi*h))/h^2;
        V=cos(xx*pi*[0:n]);
        V=V*diag(sqrt(1./diag(V'*V)));
    end

    function [rho_out,mx_out,my_out,mz_out] = proj_div_free(rho_in, mx_in, my_in, mz_in, bc0, bc1)

        % b - Au: compute divergence
        tmp1 = squeeze(-tensorprod(Dt,rho_in,2,1)); % t
        tmp1 = tmp1 - pagetranspose(squeeze(tensorprod(Dx,mx_in,2,2)));   % x
        tmp1 = tmp1 - tensorprod(my_in,Dy',3,1);  % y
        tmp1 = tmp1 - tensorprod(mz_in,Dz',4,1);  % z
        tmp1(1,:,:,:) = squeeze(tmp1(1,:,:,:)) + bc0;
        tmp1(ntp,:,:,:) = squeeze(tmp1(ntp,:,:,:)) + bc1;

        % Inverse Transform (apply inverse eigenvalue decomposition)
        tmp2 = squeeze(tensorprod(iVt,tmp1,2,1));  % t
        tmp2 = pagetranspose(squeeze(tensorprod(iVx,tmp2,2,2)));  % x
        tmp2 = squeeze(tensorprod(tmp2,iVy',3,1)); % y
        tmp2 = squeeze(tensorprod(tmp2,iVz',4,1)); % z

        % Divide by laplacian
        tmp2 = tmp2./lambda_lap;
        tmp2(1,1,1,1) = 0.0;

        % Forward transform
        tmp2 = squeeze(tensorprod(Vt,tmp2,2,1));  % t
        tmp2 = pagetranspose(squeeze(tensorprod(Vx,tmp2,2,2)));  % x
        tmp2 = squeeze(tensorprod(tmp2,Vy',3,1)); % y
        tmp2 = squeeze(tensorprod(tmp2,Vz',4,1)); % z

        rho_out = rho_in + squeeze(tensorprod(Dt,tmp2,1,1));
        mx_out  = mx_in  + pagetranspose(squeeze(tensorprod(Dx,tmp2,1,2)));
        my_out  = my_in  + squeeze(tensorprod(tmp2,Dy',3,2));
        mz_out  = mz_in  + squeeze(tensorprod(tmp2,Dz',4,2));

    end

end
